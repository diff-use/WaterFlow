import functools
import einops
import torch
import torch.nn.functional as F
from torch import nn

from esm.layers.rotary import RotaryEmbedding
from esm.layers.geom_attention import GeometricReasoningOriginalImpl
from esm.utils.structure.affine3d import Affine3D

from SLAE.nn.pos_enc import PositionalEmbedding

from loguru import logger

import math

def softclamp(t, value):
    return (t / value).tanh() * value


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        bias: bool = False, 
        qk_layernorm: bool = True, 
        pos_enc: bool = True,
        pos_enc_type: str = "rotary",
        dim_pairwise: int = 0,
        enable_attn_softclamp = True,
        attn_softclamp_value = 50.,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads

        # QKV
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Q, K layernorm
        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.pos_enc = pos_enc
        self.pos_enc_type = pos_enc_type
        if self.pos_enc_type == "rotary":
            self.rotary = RotaryEmbedding(d_model // n_heads)
        elif self.pos_enc_type == "rotary_idx":
            self.rotary = PositionalEmbedding(dim=d_model // n_heads, type="rotary", n_heads = n_heads, true_idx = True)
        elif self.pos_enc_type == "rel":
            self.rel_pos = PositionalEmbedding(dim=dim_pairwise, type="rel", rmax = 32)
        elif self.pos_enc_type == "rel_bert":
            self.rel_pos = PositionalEmbedding(dim=dim_pairwise, type="rel_bert", rmax = 32, n_heads = n_heads)
            self.rel_gain = nn.Parameter(torch.ones(n_heads))
        elif self.pos_enc_type == "rel_lookup":
            self.rel_pos = PositionalEmbedding(dim=dim_pairwise, type="rel_lookup", rmax = 32, n_heads = n_heads)
        
            self.rel_gain = nn.Parameter(torch.ones(n_heads))

        self.dim_pairwise = dim_pairwise
        if dim_pairwise != 0:
            self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
            # linear from dim_pairwise -> n_heads, no bias
            self.to_attn_bias = nn.Linear(dim_pairwise, n_heads, bias=bias)
                
            # We'll define the rearrange in forward
        else:
            self.to_attn_bias_norm = None
            self.to_attn_bias = None
        

        self.attn_softclamp = enable_attn_softclamp
        self.attn_softclamp_value = attn_softclamp_value
    def _apply_rotary(self, 
                      q: torch.Tensor, 
                      k: torch.Tensor,
                      res_idx: torch.Tensor = None):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        if self.pos_enc_type == "rotary_idx":
            q, k = self.rotary(q, k, res_idx=res_idx)
        else:
            q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, 
                x: torch.Tensor, 
                seq_id: torch.Tensor = None, 
                mask: torch.Tensor = None, 
                pairwise_mask: torch.Tensor = None, #[B, L, L]
                pairwise_repr: torch.Tensor = None, # [B, L, L, dim_pairwise]
                res_idx: torch.Tensor = None, # [B, L]
                ):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            seq_id: [batch_size, seq_len], if provided, ensures attention only within same sequence_id positions.
            mask: [batch_size, seq_len], True for valid tokens, False for padded tokens. If provided, 
                  will be combined with seq_id-based mask to form the final attention mask.
        """

        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD = self.q_ln(query_BLD).to(query_BLD.dtype)
        key_BLD = self.k_ln(key_BLD).to(key_BLD.dtype)


        rel_pos_emb = None
        if self.pos_enc and self.pos_enc_type == "rotary":
            query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)
        elif self.pos_enc and self.pos_enc_type == "rotary_idx":
            query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD, res_idx=res_idx)
        elif self.pos_enc and self.pos_enc_type == "rel":
            rel_pos_emb = self.rel_pos(query_BLD, key_BLD, pairwise_mask)
        elif self.pos_enc and self.pos_enc_type == "rel_bert":
            rel_pos_emb = self.rel_pos(query_BLD, key_BLD)
        elif self.pos_enc and self.pos_enc_type == "rel_lookup":
            rel_pos_emb = self.rel_pos(query_BLD, key_BLD)
        if pairwise_repr is None:
            pairwise_repr = rel_pos_emb
        else:
            # TODO 
            pairwise_repr = pairwise_repr + rel_pos_emb
            #raise NotImplementedError("pairwise_repr is not None")

        n_heads = self.n_heads
        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

        query_BHLD = reshaper(query_BLD)
        key_BHLD = reshaper(key_BLD)
        value_BHLD = reshaper(value_BLD)

        attn_mask = None
        if seq_id is not None:
            # seq_id-based mask: True where same seq_id, False otherwise
            # mask_BLL: [B, L, L]
            seq_id = seq_id.squeeze(-1) if seq_id.ndim == 3 else seq_id  
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)  # [B, 1, L, L]
            assert mask_BHLL.shape[2] == mask_BHLL.shape[3]


            if mask is not None:
                mask = mask.squeeze(-1) if mask.ndim == 3 else mask
                # mask: True=valid, shape [B, L]
                # We need to ensure keys at invalid positions are masked.
                # Combine with seq_id mask: only positions allowed by seq_id AND valid according to mask remain True.
                # mask_BHLL is True where allowed by seq_id.
                # We also need to ensure no queries attend to invalid keys:
                # Expand mask to [B, 1, L] for keys and broadcast:
                expanded_mask = mask.unsqueeze(1)  # [B, 1, L]
                # Combine with seq_id mask: now allowed = mask_BHLL & expanded_mask along key dimension
                # We'll treat this as (B, 1, L) broadcast over queries dimension automatically.
                # mask_BHLL: [B,1,L,L] and expanded_mask: [B,1,L]
                # To combine them, we need to ensure that for each query position,
                # the key must also be valid:
                combined_mask = mask_BHLL & expanded_mask.unsqueeze(-2)  # [B,1,L,L]
                assert combined_mask.shape[2] == combined_mask.shape[3], "combined_mask.shape: {combined_mask.shape}"
                assert combined_mask.ndim == 4, f"combined_mask.ndim: {combined_mask.ndim}"
                # attn_mask expects True where we want to mask (invalidate):
                attn_mask = ~combined_mask
                #logger.info(f"attn_mask.shape: {attn_mask.shape}")
            else:
                # No external mask, use seq_id mask directly
                attn_mask = ~mask_BHLL
        else:
            # No seq_id, only external mask if given
            if mask is not None:
                mask = mask.squeeze(-1) if mask.ndim == 3 else mask
                # mask [B, L]: True=valid
                # Convert to an attention mask. We must ensure that invalid positions are masked.
                # With no seq_id, we rely solely on mask to define validity.
                # We'll use it as a key_padding_mask kind of scenario:
                # For scaled_dot_product_attention, attn_mask shape can be [B,1,L] broadcasted over queries.
                expanded_mask = mask.unsqueeze(1)  # [B, 1, L] True=valid
                # True means valid, we need True=masked_out in attn_mask
                attn_mask = ~expanded_mask.unsqueeze(2)  # [B, 1, 1, L] will broadcast to Q dimension automatically.


        # TODO add pairwise_repr
        pair_bias = None
        if pairwise_repr is not None and self.to_attn_bias is not None:
            if self.pos_enc_type == "rel":
            # apply LN => [B,L,L,dim_pairwise]
            #normed = self.to_attn_bias_norm(pairwise_repr)
            # => shape [B,L,L,nHeads]
            # we do a manual call: out = self.to_attn_bias(normed)
            # but we have to expand the rearrange pattern
            # By default we have:
            #   out => [B,L,L,nHeads] after LinearNoBias
            # but we have the rearrange which is "b ... h -> b h ..."
            # so let's do it manually:
            # 1) linear => shape [B,L,L,nHeads]
                out = self.to_attn_bias(pairwise_repr) #normed) #F.linear(normed, self.to_attn_bias[0].weight)  # no bias
                # => shape [B,L,L,nHeads]
                # 2) rearrange => "b (l l) h -> b h l l"? Actually we do "b n n h -> b h n n"
                # let's do out = out.permute(0, 3, 1, 2)
                out = out.permute(0, 3, 1, 2)
                # => [B,nHeads,L,L]
                # done
                pair_bias = out  # rename for clarity
                assert not torch.isnan(pairwise_repr).any(), f"NaN detected in pairwise_repr: {pairwise_repr}"
                assert not torch.isnan(pair_bias).any(), f"NaN detected in pair_bias: {pair_bias}"

            elif self.pos_enc_type == "rel_bert" or self.pos_enc_type == "rel_lookup":
                pair_bias = pairwise_repr*self.rel_gain[:, None, None]
            # TODO 
            scores = torch.einsum("bhld,bhmd->bhlm", query_BHLD, key_BHLD) / math.sqrt(self.d_head)
            if pair_bias is not None:
                scores = scores + pair_bias
            if attn_mask is not None:
                scores.masked_fill_(attn_mask, float("-1e9")) 
            if self.attn_softclamp:
                scores = softclamp(scores, self.attn_softclamp_value) 
            attn_weights = F.softmax(scores, dim=-1)
            assert not torch.isnan(attn_weights).any(), f"attention weights contains NaN"
            context_BHLD = torch.einsum("bhlm,bhmd->bhld", attn_weights, value_BHLD)
            context_BLD = einops.rearrange(context_BHLD, "b h l d -> b l (h d)")
            assert not torch.isnan(context_BLD).any(), f"Transformer context_BLD NaN after pairwise_repr {pairwise_repr} and resulting bias {pair_bias}"
           

        else:
            # Perform attention
            if attn_mask is not None:
                # scaled_dot_product_attention expects a boolean mask where True indicates positions to ignore.
                # We have already prepared attn_mask accordingly.
                context_BHLD = F.scaled_dot_product_attention(
                    query_BHLD, key_BHLD, value_BHLD, attn_mask=attn_mask
                )
            else:
                # No mask provided at all
                context_BHLD = F.scaled_dot_product_attention(query_BHLD, key_BHLD, value_BHLD)

            context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
        return self.out_proj(context_BLD)

def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimension to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        v_heads: int | None = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        pos_enc: bool = True,
        pos_enc_type: str = "rotary",
        dim_pairwise: int | None = None,

    ):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            self.attn = MultiHeadAttention(
                d_model, 
                n_heads, 
                bias, 
                qk_layernorm=qk_layernorm, 
                pos_enc=pos_enc,
                pos_enc_type = pos_enc_type,
                dim_pairwise = dim_pairwise
            )
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
            )

        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
        frames: Affine3D,
        frames_mask: torch.Tensor,
        chain_id: torch.Tensor,
        mask: torch.Tensor = None,
        pairwise_repr: torch.Tensor = None,
        pairwise_mask: torch.Tensor = None, #[B, L, L]
        res_idx: torch.Tensor = None, # [B, L]
    ) -> torch.Tensor:
        """
        Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            [batch_size, seq_len, d_model] Input embeddings.
        sequence_id : torch.Tensor[int]
            [batch_size, seq_len] Sequence IDs.
        frames : Affine3D
            Geometric frame information for geometric attention.
        frames_mask : torch.Tensor[bool]
            [batch_size, seq_len] Mask for valid frames.
        chain_id : torch.Tensor[int]
            [batch_size, seq_len] Chain IDs.
        mask : torch.Tensor[bool], optional
            [batch_size, seq_len] Padding mask where True = valid, False = padded.

        Returns
        -------
        torch.Tensor[float]
            [batch_size, seq_len, d_model] Output of the transformer block.
        """
        key_padding_mask = None
        if mask is not None:
            # MultiHeadAttention expects key_padding_mask with True indicating positions to ignore.
            # If mask is True for valid tokens, invert it:
            key_padding_mask = ~mask  # Now True means padded/ignore.

        # Plain attention
        if self.use_plain_attn:
            r1 = self.attn(x, sequence_id, mask=key_padding_mask, pairwise_repr=pairwise_repr, pairwise_mask=pairwise_mask, res_idx=res_idx)
            x = x + r1 / self.scaling_factor

        # Geometric attention (frames_mask is already passed)
        if self.use_geom_attn:
            r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor

        # Feed-forward network
        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x
    


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks used in the ESM-3 model. Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
        pos_enc: bool = True,
        pos_enc_type: str = "rotary", # "rel"
        dim_pairwise: int | None = None,

    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=i < n_layers_geom,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    mask_and_zero_frameless=mask_and_zero_frameless,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                    pos_enc=pos_enc,
                    pos_enc_type = pos_enc_type,
                    dim_pairwise = dim_pairwise,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        affine: Affine3D | None = None,
        affine_mask: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        mask: torch.Tensor | None = None, # TODO check this
        pairwise_mask: torch.Tensor = None, #[B, L, L]
        pairwise_repr: torch.Tensor = None, # [B, L, L, dim_pairwise]
        res_idx: torch.Tensor = None, # [B, L]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
            affine (Affine3D | None): The affine transformation tensor or None.
            affine_mask (torch.Tensor | None): The affine mask tensor or None.
            chain_id (torch.Tensor): The protein chain tensor of shape (batch_size, sequence_length).
                Only used in geometric attention.

        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
        """
        *batch_dims, _ = x.shape
        #logger.info(f"x.shape: {x.shape}  mask.shape: {mask.shape}")
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id, affine, affine_mask, chain_id, mask, pairwise_repr=pairwise_repr, pairwise_mask=pairwise_mask, res_idx=res_idx)
            hiddens.append(x)
        hiddens = torch.stack(hiddens, dim=0)
        return self.norm(x), x, hiddens
