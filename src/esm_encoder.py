# esm_encoder.py
"""
ESM embeddings wrapper.

This encoder reads pre-computed ESM3 embeddings from data and returns
them directly as scalar features with zero vector channels. Downstream
GVP message-passing layers (including protein-protein edges) provide all
geometric processing.
"""

from __future__ import annotations

import torch

from src.encoder_base import CachedEmbeddingEncoder, register_encoder


@register_encoder("esm")
class ESMEncoder(CachedEmbeddingEncoder):
    """
    ESM encoder that reads cached embeddings from data.

    Returns (esm_embedding, empty_vectors) with output_dims = (esm_dim, 0).
    No learnable parameters — all geometric processing happens in the
    downstream ProteinWaterUpdate layers (pp, wp, pw, ww edges).
    """

    def __init__(self, esm_dim: int = 1536):
        """
        Initialize ESMEncoder.

        Args:
            esm_dim: Dimension of ESM embeddings (default: 1536 for ESM3-open)
        """
        super().__init__(embedding_dim=esm_dim, embedding_key="esm_embedding")

    @property
    def encoder_type(self) -> str:
        return "esm"

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> ESMEncoder:
        """
        Construct ESMEncoder from config dict.

        Args:
            config: Configuration dictionary with:
                - esm_dim: ESM embedding dimension (default: 1536)
            device: Device to place the encoder on

        Returns:
            Instantiated ESMEncoder
        """
        esm_dim = config.get("esm_dim", 1536)
        return cls(esm_dim=esm_dim).to(device)
