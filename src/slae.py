# slae.py
"""
SLAE (Strictly Local All-Atom Environment) base encoder implementation.

This encoder reads pre-computed SLAE embeddings from the data and returns
them directly as scalar features with zero vector channels. Downstream
GVP message-passing layers (including protein-protein edges) provide all
geometric processing.
"""

from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData

from .encoder_base import BaseProteinEncoder, register_encoder


@register_encoder('slae')
class SLAEEncoder(BaseProteinEncoder):
    """
    SLAE encoder that reads cached embeddings from data.

    Returns (slae_embedding, empty_vectors) with output_dims = (slae_dim, 0).
    No learnable parameters — all geometric processing happens in the
    downstream ProteinWaterUpdate layers (pp, wp, pw, ww edges).
    """

    def __init__(self, slae_dim: int = 128):
        super().__init__()
        self._slae_dim = slae_dim

    @property
    def output_dims(self) -> Tuple[int, int]:
        """Return (slae_dim, 0) — scalars only."""
        return (self._slae_dim, 0)

    @property
    def encoder_type(self) -> str:
        return 'slae'

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read cached SLAE embeddings and return (s, V).

        Args:
            data: HeteroData with data['protein'].slae_embedding

        Returns:
            s: (N, slae_dim) — raw SLAE embeddings
            V: (N, 0, 3)    — empty vector features
        """
        if 'slae_embedding' not in data['protein']:
            raise NotImplementedError(
                "SLAE encoder requires cached embeddings. "
                "Please provide pre-computed slae_embedding in data['protein']. "
                "Run scripts/precompute_slae_embeddings.py first."
            )

        embeddings = data['protein'].slae_embedding
        V = embeddings.new_empty(embeddings.size(0), 0, 3)
        return embeddings, V

    @classmethod
    def from_config(cls, config: Dict, device: torch.device) -> 'SLAEEncoder':
        """
        Construct SLAEEncoder from config dict.

        Args:
            config: Configuration dictionary with:
                - slae_dim: SLAE embedding dimension (default: 128)
            device: Device to place the encoder on

        Returns:
            Instantiated SLAEEncoder
        """
        slae_dim = config.get('slae_dim', 128)
        return cls(slae_dim=slae_dim).to(device)
