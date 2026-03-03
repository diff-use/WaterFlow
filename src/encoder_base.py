"""
Base encoder and registry for modular encoders.

This module provides:
- BaseProteinEncoder: Abstract base class that all encoders must implement
- Registry pattern: Decorator-based registration and build_encoder() function
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

class BaseProteinEncoder(ABC, nn.Module):
    """
    Abstract base class for protein encoders.

    All encoder implementations must inherit from this class and implement
    the required properties and methods.
    """

    @property
    @abstractmethod
    def output_dims(self) -> tuple[int, int]:
        """Return (scalar_dim, vector_dim) output dimensions."""
        raise NotImplementedError("Subclasses must implement output_dims")

    @property
    @abstractmethod
    def encoder_type(self) -> str:
        """Return encoder type identifier ('gvp', 'slae', 'esm', etc.)."""
        raise NotImplementedError("Subclasses must implement encoder_type")

    @abstractmethod
    def forward(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """
        Encode protein data.

        Args:
            data: HeteroData with protein nodes

        Returns:
            tuple of (s, V, pp_edge_attr) where:
                s: (N, scalar_dim) scalar features
                V: (N, vector_dim, 3) vector features
                pp_edge_attr: tuple (s_edge, V_edge) for PP edges, or None if encoder
                    doesn't process edges (e.g., SLAE, ESM)
        """
        raise NotImplementedError("Subclasses must implement forward")

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, device: torch.device) -> BaseProteinEncoder:
        """
        Construct encoder from config dict.

        Args:
            config: Configuration dictionary with encoder parameters
            device: Device to place the encoder on

        Returns:
            Instantiated encoder
        """
        raise NotImplementedError("Subclasses must implement from_config")


class CachedEmbeddingEncoder(BaseProteinEncoder):
    """
    Base class for encoders that read pre-computed embeddings from data.

    Subclasses only need to define:
    - encoder_type property (return string like 'slae', 'esm')
    - from_config class method
    """

    def __init__(self, embedding_dim: int, embedding_key: str):
        """
        Initialize CachedEmbeddingEncoder.

        Args:
            embedding_dim: Dimension of the cached embeddings
            embedding_key: Key to look up embeddings in data['protein']
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._embedding_key = embedding_key

    @property
    def output_dims(self) -> tuple[int, int]:
        """Return (embedding_dim, 0) — scalars only."""
        return self._embedding_dim, 0

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, None]:
        """
        Read cached embeddings and return (s, V, None).

        Args:
            data: HeteroData with cached embeddings in data['protein']

        Returns:
            s: (N, embedding_dim) — raw embeddings
            V: (N, 0, 3)         — empty vector features
            pp_edge_attr: None   — cached embedding encoders don't process edges
        """
        if self._embedding_key not in data['protein']:
            raise NotImplementedError(
                f"{self.encoder_type.upper()} encoder requires cached embeddings. "
                f"Please provide pre-computed {self._embedding_key} in data['protein']."
            )

        embeddings = data['protein'][self._embedding_key]
        V = embeddings.new_empty(embeddings.size(0), 0, 3)
        return embeddings, V, None

# global encoder registry
_ENCODER_REGISTRY: dict[str, BaseProteinEncoder] = {}

def register_encoder(name: str):
    """
    Decorator to register an encoder class.

    Usage:
        @register_encoder('my_encoder')
        class MyEncoder(BaseProteinEncoder):
            ...
    """

    def decorator(cls: BaseProteinEncoder) -> BaseProteinEncoder:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered")
        _ENCODER_REGISTRY[name] = cls
        return cls

    return decorator

def get_encoder_class(name: str) -> BaseProteinEncoder:
    """
    Get encoder class by name.

    Args:
        name: Encoder type identifier

    Returns:
        Encoder class

    Raises:
        KeyError: If encoder name is not registered
    """
    if name not in _ENCODER_REGISTRY:
        available = list(_ENCODER_REGISTRY.keys())
        raise KeyError(f"Unknown encoder type '{name}'. Available: {available}")
    return _ENCODER_REGISTRY[name]

def build_encoder(config: dict, device: torch.device) -> BaseProteinEncoder:
    """
    Build encoder from configuration dict.

    Args:
        config: Configuration dictionary containing:
            - encoder_type: 'gvp' or 'slae' (required)
            - Other encoder-specific parameters
        device: Device to place the encoder on

    Returns:
        Instantiated encoder implementing BaseProteinEncoder
    """
    if "encoder_type" not in config:
        raise ValueError("'encoder_type' must be specified in config")
    encoder_type = config["encoder_type"]
    encoder_cls = get_encoder_class(encoder_type)
    return encoder_cls.from_config(config, device)
