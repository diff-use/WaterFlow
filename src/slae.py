# slae.py
"""
SLAE (Strictly Local All-Atom Environment) base encoder implementation.

This encoder uses pre-computed SLAE embeddings stored in the data
and projects them to the (scalar, vector) format expected by the flow model.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from .encoder_base import BaseProteinEncoder, register_encoder


class SLAEProjection(nn.Module):
    """
    Projects SLAE scalar embeddings to (scalar, vector) tuple format.

    SLAE outputs scalar embeddings only (N_atoms, slae_dim).
    This module:
    1. Projects SLAE scalars to target scalar dimension
    2. Generates zero vectors (downstream GVP layers will create vectors from scalars)
    """

    def __init__(self, slae_dim: int = 128, out_dims: Tuple[int, int] = (256, 32)):
        """
        Args:
            slae_dim: SLAE embedding dimension (default: 128)
            out_dims: (s_dim, v_dim) for output format
        """
        super().__init__()
        s_dim, v_dim = out_dims

        self.scalar_proj = nn.Sequential(
            nn.Linear(slae_dim, s_dim),
            nn.LayerNorm(s_dim),
        )

        self.v_dim = v_dim

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert SLAE embeddings to (scalar, vector) tuple format.

        Args:
            embeddings: (N_atoms, slae_dim) SLAE node embeddings

        Returns:
            s: (N_atoms, s_dim) - projected scalar features
            V: (N_atoms, v_dim, 3) - zero vectors
        """
        s = self.scalar_proj(embeddings)

        V = torch.zeros(
            embeddings.size(0), self.v_dim, 3,
            device=embeddings.device,
            dtype=embeddings.dtype
        )

        return s, V


@register_encoder('slae')
class SLAEEncoder(BaseProteinEncoder):
    """
    SLAE encoder that uses cached embeddings from data.

    This encoder expects pre-computed SLAE embeddings stored in
    data['protein'].slae_embedding and projects them to the
    (scalar, vector) format expected by downstream flow layers.
    """

    def __init__(
        self,
        slae_dim: int = 128,
        output_dims: Tuple[int, int] = (256, 32),
        freeze: bool = False,
    ):
        """
        Args:
            slae_dim: SLAE embedding dimension
            output_dims: (scalar_dim, vector_dim) output dimensions
            freeze: If True, projection layers are frozen (default: True)
        """
        super().__init__()
        self._output_dims = output_dims
        self._slae_dim = slae_dim
        self.projection = SLAEProjection(slae_dim, output_dims)

        if freeze:
            for p in self.projection.parameters():
                p.requires_grad = False

    @property
    def output_dims(self) -> Tuple[int, int]:
        """Return (scalar_dim, vector_dim)."""
        return self._output_dims

    @property
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        return 'slae'

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get (s, V) features from cached SLAE embeddings.

        Args:
            data: HeteroData with data['protein'].slae_embedding

        Returns:
            Tuple of (s, V) where:
                s: (N, scalar_dim) scalar features
                V: (N, vector_dim, 3) vector features
        """
        if 'slae_embedding' not in data['protein']:
            raise NotImplementedError(
                "SLAE encoder requires cached embeddings. "
                "Please provide pre-computed slae_embedding in data['protein']. "
                "Run scripts/precompute_slae_embeddings.py first."
            )

        embeddings = data['protein'].slae_embedding
        return self.projection(embeddings)

    @classmethod
    def from_config(cls, config: Dict, device: torch.device) -> 'SLAEEncoder':
        """
        Construct SLAEEncoder from config dict.

        Args:
            config: Configuration dictionary with:
                - slae_dim: SLAE embedding dimension (default: 128)
                - slae_adapter_dims: "s,v" string or None (uses hidden_s, hidden_v)
                - hidden_s, hidden_v: flow hidden dimensions
                - freeze_encoder: whether to freeze projection (default: True)
            device: Device to place the encoder on

        Returns:
            Instantiated SLAEEncoder
        """
        slae_dim = config.get('slae_dim', 128)

        # Parse output dimensions
        slae_adapter_dims = config.get('slae_adapter_dims')
        if slae_adapter_dims is not None:
            s_dim, v_dim = map(int, slae_adapter_dims.split(','))
        else:
            s_dim = config.get('hidden_s', 256)
            v_dim = config.get('hidden_v', 32)

        freeze = config.get('freeze_encoder', True)

        encoder = cls(
            slae_dim=slae_dim,
            output_dims=(s_dim, v_dim),
            freeze=freeze,
        )

        return encoder.to(device)


# Backward compatibility alias
SLAEToGVPAdapter = SLAEProjection
