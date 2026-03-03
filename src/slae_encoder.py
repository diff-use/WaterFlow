"""
SLAE (Strictly Local All-Atom Environment) base encoder implementation.

This encoder reads pre-computed SLAE embeddings from the data and returns
them directly as scalar features with zero vector channels. Downstream
GVP message-passing layers (including protein-protein edges) provide all
geometric processing.
"""
from __future__ import annotations

import torch

from src.encoder_base import CachedEmbeddingEncoder, register_encoder


@register_encoder('slae')
class SLAEEncoder(CachedEmbeddingEncoder):
    """
    SLAE encoder that reads cached embeddings from data.

    Returns (slae_embedding, empty_vectors) with output_dims = (slae_dim, 0).
    No learnable parameters — all geometric processing happens in the
    downstream ProteinWaterUpdate layers (pp, wp, pw, ww edges).
    """

    def __init__(self, slae_dim: int = 128):
        """
        Initialize SLAEEncoder.

        Args:
            slae_dim: Dimension of SLAE embeddings (default: 128)
        """
        super().__init__(embedding_dim=slae_dim, embedding_key='slae_embedding')

    @property
    def encoder_type(self) -> str:
        return 'slae'

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> SLAEEncoder:
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
