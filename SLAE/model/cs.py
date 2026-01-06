from SLAE.model.decoder import AllAtomDecoder
import torch, torch.nn as nn
from torch.nn import functional as F
from typing import Tuple

# --------------------------------------------------------------------- #
#  Decoder backbone with flexible UN-FREEZE                             #
# --------------------------------------------------------------------- #
class DecoderBackbone(nn.Module):
    """
    Wrap projection + TransformerStack of `AllAtomDecoder` and expose
    trainable control over the *last N* transformer blocks.
    """
    def __init__(self,
                 allatom_decoder: AllAtomDecoder,
                 unfreeze_n_blocks: int = 1):   # 1 = old behaviour
        super().__init__()
        self.proj  = allatom_decoder.proj
        self.stack = allatom_decoder.decoder_stack   # list-like .blocks
        self.d_model = self.stack.d_model            # handy for caller

        # 1) freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

        # 2) un-freeze last N blocks (or all if −1)
        if unfreeze_n_blocks != 0:
            n = len(self.stack.blocks) if unfreeze_n_blocks < 0 else unfreeze_n_blocks
            for blk in self.stack.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)

    def forward(self, x, mask):
        x = self.proj(x)
        x, *_ = self.stack(
            x,
            mask=mask,
            pairwise_mask=None,
            pairwise_repr=None,
            affine=None,
            affine_mask=None,
            sequence_id=None,
            chain_id=None,
            res_idx=None,
        )
        return x



# ------------------------------------------------------------------ #
#  Building blocks                                                   #
# ------------------------------------------------------------------ #
class GEGLU(nn.Module):
    """Gated GELU: (xW₁) ⊙ GELU(xW₂) — PaLM style"""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class ResidualFF(nn.Module):
    """LayerNorm → GEGLU → Dropout → Linear → Dropout + residual"""
    def __init__(self, width: int, hidden: int, drop: float):
        super().__init__()
        self.norm   = nn.LayerNorm(width)
        self.ff_g   = GEGLU(width, hidden)
        self.linear = nn.Linear(hidden, width)
        self.drop   = nn.Dropout(drop)

    def forward(self, x):
        y = self.ff_g(self.norm(x))
        y = self.drop(self.linear(self.drop(y)))
        return x + y


class TinyContext(nn.Module):
    """2-layer Transformer encoder (optional) for long-range context."""
    def __init__(self, width: int = 256, heads: int = 4, layers: int = 2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x, mask):                       # x: [B,L,W]
        key_pad = ~mask                              # True = pad
        return self.enc(x, src_key_padding_mask=key_pad)


# ------------------------------------------------------------------ #
#  Main network                                                      #
# ------------------------------------------------------------------ #


class ShiftNet(nn.Module):
    """
    Predict backbone-15N chemical shifts (ppm) per residue.

    Parameters
    ----------
    emb_dim : input embedding dimension.
    width   : hidden width for residual MLP blocks.
    depth   : number of ResidualFF blocks.
    drop    : dropout after each linear.
    ppm_window : (lo, hi) expected shift range; sets Huber δ.
    frac_delta : δ as a fraction of window (default 0.10 → 2 ppm).
    analytic_centering : subtract mean residual *per protein*.
    use_ctx_conv : add depth-wise conv for ±2-residue mixing.
    use_ctx_transformer : add TinyContext for long-range attention.
    """
    def __init__(
        self,
        emb_dim: int = 128,
        width: int = 256,
        depth: int = 4,
        conv_kernel_size: int = 5, # 5 for ±2-residue mixing, 11 for ±5-residue mixing
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        drop: float = 0.2,
        ppm_window: Tuple[float, float] = (110., 130.),
        bias: float = 119.0,
        frac_delta: float = 0.10,
        analytic_centering: bool = True,
        use_ctx_conv: bool = True,
        use_ctx_transformer: bool = True,
    ):
        super().__init__()
        self.analytic_centering = analytic_centering

        # -- input projection ------------------------------------------
        self.in_proj = nn.Linear(emb_dim, width) if emb_dim != width else nn.Identity()

        # -- optional depth-wise conv (local context) ------------------
        if use_ctx_conv:
            padding  = conv_kernel_size // 2
            self.ctx_conv = nn.Conv1d(width, width, kernel_size=conv_kernel_size,
                                      padding=padding, groups=width)
        else:
            self.ctx_conv = None

        # -- optional tiny Transformer (global context) ---------------
        self.ctx_tr = TinyContext(width = width, heads = transformer_heads, layers = transformer_layers) if use_ctx_transformer else None

        # -- residual MLP stack ---------------------------------------
        self.blocks = nn.Sequential(
            *[ResidualFF(width, width * 2, drop) for _ in range(depth)]
        )
        self.bias = bias
        # -- output head ----------------------------------------------
        self.out_lin = nn.Linear(width, 1)
        nn.init.constant_(self.out_lin.bias, bias)      # prior ≈ μ₁₅ᴺ

        # -- Huber loss -----------------------------------------------
        δ = (ppm_window[1] - ppm_window[0]) * frac_delta
        self.huber = nn.SmoothL1Loss(beta=δ, reduction="none")

    # ----------------------------------------------------------------
    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        emb  : [B, L, emb_dim]   (float32)
        mask : [B, L]            (bool)  True where shift exists
        returns pred [B, L]      (float32, ppm)
        """
        x = self.in_proj(emb)                       # [B,L,W]

        # depth-wise conv (local ±2)
        if self.ctx_conv is not None:
            x = self.ctx_conv(x.transpose(1, 2)).transpose(1, 2)

        # tiny Transformer (global)
        if self.ctx_tr is not None:
            x = self.ctx_tr(x, mask)

        x = self.blocks(x)                          # residual MLP stack
        return self.out_lin(x).squeeze(-1)          # [B,L]

    # ----------------------------------------------------------------
    def loss(
        self,
        pred: torch.Tensor,            # [B,L]
        target: torch.Tensor,          # [B,L]
        mask: torch.Tensor,            # [B,L]  True = valid shift
    ) -> torch.Tensor:
        """Huber loss with optional per-protein centring."""
        if self.analytic_centering:
            denom = mask.sum(-1, keepdim=True).clamp(min=1)  # [B,1]
            mean_err = ((pred - target) * mask).sum(-1, keepdim=True) / denom
            pred = pred - mean_err                           # remove Δref

        loss = self.huber(pred, target)
        return loss[mask].mean()

