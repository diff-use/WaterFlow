# esm_encoder.py
from __future__ import annotations

"""
ESM embeddings wrapper.

This encoder reads pre-computed ESM3 embeddings from data and returns
them directly as scalar features with zero vector channels. Downstream
GVP message-passing layers (including protein-protein edges) provide all
geometric processing.
"""

import torch
from torch_geometric.data import HeteroData

from src.encoder_base import BaseProteinEncoder, register_encoder


@register_encoder('esm')
class ESMEncoder(BaseProteinEncoder):
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
        super().__init__()
        self._esm_dim = esm_dim

    @property
    def output_dims(self) -> tuple[int, int]:
        """Return (esm_dim, 0) — scalars only."""
        return (self._esm_dim, 0)

    @property
    def encoder_type(self) -> str:
        return 'esm'

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Read cached ESM embeddings and return (s, V, None).

        Args:
            data: HeteroData with data['protein'].esm_embedding

        Returns:
            s: (N, esm_dim) — raw ESM embeddings
            V: (N, 0, 3)    — empty vector features
            pp_edge_attr: None — ESM doesn't process edges
        """
        if 'esm_embedding' not in data['protein']:
            raise NotImplementedError(
                "ESM encoder requires cached embeddings. "
                "Please provide pre-computed esm_embedding in data['protein']. "
                "Run scripts/generate_esm_embeddings.py first."
            )

        embeddings = data['protein'].esm_embedding
        V = embeddings.new_empty(embeddings.size(0), 0, 3)
        return embeddings, V, None

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
        esm_dim = config.get('esm_dim', 1536)
        return cls(esm_dim=esm_dim).to(device)
