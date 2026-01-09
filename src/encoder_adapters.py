"""
Adapter modules to convert between different encoder output formats.
"""

import torch
import torch.nn as nn


class SLAEToGVPAdapter(nn.Module):
    """
    Adapter from SLAE encoder embeddings to GVP format.

    SLAE outputs scalar embeddings only (N_atoms, 128).
    GVP architecture expects (scalar, vector) tuples: ((N, s_dim), (N, v_dim, 3)).

    This adapter:
    1. Projects SLAE scalars to target scalar dimension
    2. Generates zero vectors (encoder_to_flow GVP will create vectors from scalars)

    The adapter's scalar projection and subsequent encoder_to_flow GVP layer
    are both trainable, allowing the model to learn how to map frozen SLAE
    embeddings into effective features for the flow model.
    """

    def __init__(self, slae_dim=128, out_dims=(256, 32)):
        """
        Args:
            slae_dim: SLAE embedding dimension (default: 128)
            out_dims: (s_dim, v_dim) for GVP format
        """
        super().__init__()
        s_dim, v_dim = out_dims

        # project scalars from SLAE dimension to target dimension
        self.scalar_proj = nn.Sequential(
            nn.Linear(slae_dim, s_dim),
            nn.LayerNorm(s_dim),
        )

        self.v_dim = v_dim

    def forward(self, embeddings):
        """
        Convert SLAE embeddings to (scalar, vector) tuple format.

        Args:
            embeddings: (N_atoms, slae_dim) SLAE node embeddings

        Returns:
            s: (N_atoms, s_dim) - projected scalar features
            V: (N_atoms, v_dim, 3) - zero vectors (encoder_to_flow will generate real vectors)
        """
        s = self.scalar_proj(embeddings)

        # initialize with zero vectors
        # the encoder_to_flow GVP layer will generate non-zero vectors from scalars
        V = torch.zeros(
            embeddings.size(0), self.v_dim, 3,
            device=embeddings.device,
            dtype=embeddings.dtype
        )

        return s, V
