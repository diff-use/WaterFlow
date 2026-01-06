from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from loguru import logger
import einx

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb_batch(x, cos, sin):
    """
    x   : (B, L, H, D)
    cos : (B, L, D/2)
    sin : (B, L, D/2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], "headdim too small for requested RoPE dim"

    # make cos/sin broadcastable to heads
    cos = repeat(cos, "b l d -> b l 1 (2 d)")
    sin = repeat(sin, "b l d -> b l 1 (2 d)")

    x_ro, x_pass = x[..., :ro_dim], x[..., ro_dim:]
    x_rot = x_ro * cos + rotate_half(x_ro) * sin
    return torch.cat((x_rot, x_pass), dim=-1)


def apply_rotary_emb_torch(x, cos, sin):
    """
    x: (batch_size, seqlen, nheads, headdim)
    #cos, sin: (seqlen, rotary_dim / 2)
    """
    logger.info(f"Shape of q/k before rotation {x.shape}")
    logger.info(f"Shape of cos/sin {cos.shape}")
    ro_dim = cos.shape[-1] * 2
    logger.info(f"Ro_dim is {ro_dim}")
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )

class PositionalEmbedding(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        base=10000.0,
        rmax = 32, 
        type = "rotary", # "rel", "rel_bert"  # one‑hot  →  MLP  →  per‑head scalar bias
        true_idx = False,
        n_heads = 16,
        scaling_factor = 1.0
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.type = type
        self.rmax = rmax
        self.true_idx = true_idx
        self.scaling_factor = scaling_factor
        if type == "rotary":
            inv_freq = self._compute_inv_freq()
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            #logger.info(f"Registered buffer for rotary pos enc")
        elif type == "rel":
            # For 'rel', we build an Embedding for [0..2*rmax + 1], i.e. chain break => index=2*rmax+1
            # We'll produce a per-pair embedding of dimension 'dim'
            self.rel_embedding = nn.Linear(2*rmax + 2, dim, bias=False)
        elif type == "rel_bert":
            assert n_heads is not None, "`n_heads` must be set for rel_bert"
            self.n_heads   = n_heads
            self.n_bins    = 2 * rmax + 2                       # 0…rmax, >rmax
            self.mlp = nn.Sequential(
                nn.Linear(self.n_bins, dim, bias=False),
                nn.GELU(),
                nn.Linear(dim, n_heads, bias=False)
            )
            # BERT/T5 initialise the last projection to zeros
            nn.init.zeros_(self.mlp[-1].weight)

        elif type == "rel_lookup":
            assert n_heads is not None, "`n_heads` must be set for rel_lookup"
            self.n_heads = n_heads
            self.n_bins  = 2 * rmax + 2                     # 0..rmax, >rmax
            # Parameter table:   (n_heads, n_bins)
            self.bias_table = nn.Parameter(torch.zeros(n_heads, self.n_bins))
            # zero‑init → identical to “no‑pos-enc” at step 0, like T5/BERT

        else:
            raise ValueError(f"Unknown positional embedding type '{type}'")

    def _distance_bucket(self, seq_len: int, device):
        idx    = torch.arange(seq_len, device=device)
        delta  = idx[:, None] - idx[None, :]
        delta  = delta.clamp(-self.rmax, self.rmax) + self.rmax
        long   = (idx[:, None] - idx[None, :]).abs() > self.rmax
        delta[long] = 2 * self.rmax + 1
        return delta  # (L,L)


    def _compute_inv_freq(self):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32)
                / self.dim
            )
        )
    
    def construct_relative_positions(self, res_idx, chain_mask, device=None):
        """
        Computes adjusted relative positions with chain break handling.

        Args:
            res_idx: (B, L) Residue indices.
            chain_mask: (B, L, L) Binary mask indicating same-chain membership.

        Returns:
            adjusted_positions: (B, L, L) Adjusted relative positions.
        """

        # Compute raw relative positions
        relative_positions = res_idx.unsqueeze(-1) - res_idx.unsqueeze(-2)  # (B, L, L)
        #logger.info(f"Chain mask shape {chain_mask.size()}")
        #logger.info(f"RelPos shape {relative_positions.size()}")
        # Apply clipping: Keep normal positions within a chain, reset at chain breaks
        adjusted_rel_pos = torch.where(
            chain_mask,
            torch.clamp(relative_positions + self.rmax, 0, 2 * self.rmax),  # Normal positions
            2 * self.rmax + 1  # Large reset value at chain breaks
        )
        #logger.info(f"adjusted rel pos is {adjusted_rel_pos}")


        return adjusted_rel_pos

    def rot_compute_cos_sin(self, t, device=None, dtype=None):
        """
        For each (batch, i, j), compute the cos/sin used to rotate Q[i], K[j].
        
        Args:
            t: torch.arange(seq_len, device=device, dtype=torch.float32)
            device: Device to use.
            dtype: Data type to use.
            
        Returns:
            cos, sin: Cosine and sine for relative positions for the batch. (B, L, L, dim/2)
            """

        # We want to recompute self.inv_freq if it was not loaded in fp32
        if self.inv_freq.dtype != torch.float32:
            inv_freq = self.inv_freq.to(torch.float32)
        else:
            inv_freq = self.inv_freq
 
        # Don't do einsum, it converts fp32 to fp16 under AMP
        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, inv_freq)
        logger.info(f"computing cos_sin freq: {freqs.shape}")
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        
        return cos, sin
    
    def rot_compute_cos_sin_idx(self, res_idx: torch.Tensor, dtype=None):
        """
        Build cos/sin when each element in the batch has its own residue indices.

        Args
        ----
        res_idx : (B, L)  float / int
            Arbitrary residue numbers per token *per* sample.
        dtype   : torch.dtype  –  normally q.dtype / k.dtype

        Returns
        -------
        cos : (B, L, dim/2)
        sin : (B, L, dim/2)
        """
        # (dim/2,)  → (1,1,dim/2) for broadcasting
        inv_freq = self.inv_freq.to(res_idx.device).unsqueeze(0).unsqueeze(0)

        # freqs[b,i,j] = res_idx[b,i] * inv_freq[j]
        freqs = res_idx.unsqueeze(-1) * inv_freq        # (B, L, dim/2)

        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        return cos, sin


    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        res_idx: torch.Tensor = None,
        chain_mask : torch.Tensor = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        chain_mask: (batch, seqlen, seqlen) binary mask indicating same-chain membership.  
        res_idx: (batch, seqlen) residue indices
        """

        device = q.device
        B = q.size(0)
        seq_len = q.size(1)

        # We want fp32 here as well since inv_freq will be multiplied with t, and the output
        # will be large. Having it in bf16 will lose a lot of precision and cause the
        # cos & sin output to change significantly.
        
        if self.type == "rotary":
            if not self.true_idx:
                res_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
                res_idx /= self.scaling_factor
            
            
                cos, sin = self.rot_compute_cos_sin(
                    res_idx, device=device, dtype=q.dtype
                )
                return (
                    apply_rotary_emb_torch(
                        q,
                        cos,
                        sin
                    ),
                    apply_rotary_emb_torch(
                        k,
                        cos,
                        sin
                    ),
                )
            elif self.true_idx and res_idx is not None:
                #logger.info(f"Using true_idx and res_idx")
                res_idx = res_idx.to(torch.float32).to(device) / self.scaling_factor
                #logger.info(f"Res idx is {res_idx}")
                cos, sin = self.rot_compute_cos_sin_idx(res_idx, dtype=q.dtype)
                #logger.info(f"Cos and sin are of shape {cos.shape} and is {cos}")
                q = apply_rotary_emb_batch(q, cos, sin)
                k = apply_rotary_emb_batch(k, cos, sin)
                return q, k

            else:
                raise ValueError("res_idx must be provided if true_idx is True")
        elif self.type == "rel":
            if chain_mask is None: 
                # initialize chain mask as all True
                chain_mask = torch.ones(q.size(0), seq_len, seq_len, device=device, dtype=torch.bool)
            # TODO 
            base_idx_1d = torch.arange(seq_len, device=device, dtype=torch.float32)
            res_idx = base_idx_1d.unsqueeze(0).expand(B, -1)     
            adjusted_rel_pos = self.construct_relative_positions(res_idx, chain_mask, device=device) # (B, L, L)
            dtype = self.rel_embedding.weight.dtype
            # convert to one-hot
            def onehot(x, bins):
                dist_from_bins = einx.subtract('... i, j -> ... i j', x, bins)
                indices = dist_from_bins.abs().min(dim = -1, keepdim = True).indices
                one_hots = F.one_hot(indices.long(), num_classes = len(bins))
                return one_hots.type(dtype)

            r_arange = torch.arange(2*self.rmax + 2, device = device)
            rel_pos_onehot = onehot(adjusted_rel_pos, r_arange).squeeze()


            rel_embedding = self.rel_embedding(rel_pos_onehot)
            #logger.info(f"Rel embedding is {rel_embedding}") 
            return rel_embedding
        
        elif self.type == "rel_bert":
            bins = self._distance_bucket(seq_len, device=device)         # (L,L)
            one_hot = F.one_hot(bins, num_classes=self.n_bins).float()  # (L,L,C)
            bias = self.mlp(one_hot)                                # (L,L,H)
            bias = bias.permute(2, 0, 1)                            # (H,L,L)
            bias = bias.unsqueeze(0).expand(B, -1, -1, -1)          # (B,H,L,L)
            return bias
        
        elif self.type == "rel_lookup":
            bucket = self._distance_bucket(seq_len, device=device)          # (L,L)
            # gather per‑head scalar bias
            #   bias_table: (H, C)   bucket: (L,L) -> (H,L,L)
            bias   = self.bias_table[:, bucket]                       # (H,L,L)
            bias   = bias.unsqueeze(0).expand(B, -1, -1, -1)          # (B,H,L,L)
            return bias


        else:
            raise NotImplementedError(f"Unknown positional embedding type: {self.type}")
