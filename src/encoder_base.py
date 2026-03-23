"""
Base encoder and registry for modular encoders.

This module provides:
- BaseProteinEncoder: Abstract base class that all encoders must implement
- Registry pattern: Decorator-based registration and build_encoder() function
- CachedEmbeddingEncoder: Concrete encoder for pre-computed embeddings (ESM, SLAE)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn


if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

# global encoder registry
_ENCODER_REGISTRY: dict[str, type["BaseProteinEncoder"]] = {}


def register_encoder(name: str):
    """
    Decorator to register an encoder class.

    Usage:
        @register_encoder('my_encoder')
        class MyEncoder(BaseProteinEncoder):
            ...
    """

    def decorator(cls: type[BaseProteinEncoder]) -> type[BaseProteinEncoder]:
        if name in _ENCODER_REGISTRY:
            raise KeyError(f"Encoder '{name}' is already registered")
        _ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_encoder_class(name: str) -> type[BaseProteinEncoder]:
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
            - encoder_type: 'gvp', 'slae', or 'esm' (required)
            - Other encoder-specific parameters
        device: Device to place the encoder on

    Returns:
        Instantiated encoder implementing BaseProteinEncoder
    """
    if "encoder_type" not in config:
        raise KeyError("'encoder_type' must be specified in config")
    encoder_type = config["encoder_type"]
    encoder_cls = get_encoder_class(encoder_type)
    return encoder_cls.from_config(config, device)


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


@register_encoder("esm")
@register_encoder("slae")
class CachedEmbeddingEncoder(BaseProteinEncoder):
    """
    Encoder for pre-computed protein embeddings (ESM, SLAE, etc.).

    This pass-through encoder reads embeddings stored in HeteroData under a
    specified key and returns them as scalar features. No neural network
    computation occurs; all geometric processing happens in downstream layers.

    Supported embedding types:
    - ESM: Evolutionary Scale Modeling embeddings (https://github.com/evolutionaryscale/esm)
    - SLAE: Strictly Local Atom-level Environment Embeddings (https://www.biorxiv.org/content/10.1101/2025.10.03.680398v1)

    Embedding dimension is inferred from the data on first forward pass.
    Accessing output_dims before forward() raises RuntimeError.

    Memory: Embeddings are NOT loaded at initialization. The encoder stores
    only the key name; actual embeddings are read from data at forward time,
    allowing standard PyTorch batching/streaming.

    Note: Returns empty vector features (shape Nx0x3) since cached embeddings
    are scalar-only.
    """

    def __init__(
        self, embedding_key: str, encoder_type: str, embedding_dim: int | None = None
    ):
        """
        Initialize CachedEmbeddingEncoder.

        Args:
            embedding_key: Key to look up embeddings in data['protein']
            encoder_type: Encoder type identifier ('esm' or 'slae')
            embedding_dim: Optional embedding dimension. If provided, output_dims is
                available immediately. If None, dimension is inferred on first forward.
        """
        super().__init__()
        self._embedding_dim: int | None = embedding_dim
        self._embedding_key = embedding_key
        self._encoder_type = encoder_type

    @property
    def output_dims(self) -> tuple[int, int]:
        """Return (embedding_dim, 0) — scalars only.

        Raises:
            RuntimeError: If accessed before first forward pass (dimension not yet inferred)
        """
        if self._embedding_dim is None:
            raise RuntimeError(
                f"{self._encoder_type.upper()} encoder dimension not yet known. "
                "Run a forward pass first to infer dimension from data."
            )
        return self._embedding_dim, 0

    @property
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        return self._encoder_type

    def forward(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """
        Read cached embeddings and return (s, V, None).

        On first call, infers embedding dimension from the data.

        Args:
            data: HeteroData with cached embeddings in data['protein']

        Returns:
            s: (N, embedding_dim) — raw embeddings
            V: (N, 0, 3)         — empty vector features
            pp_edge_attr: None   — cached embedding encoders don't process edges
        """
        if self._embedding_key not in data["protein"]:
            raise KeyError(
                f"{self._encoder_type.upper()} encoder requires cached embeddings. "
                f"Please provide pre-computed '{self._embedding_key}' in data['protein']."
            )

        embeddings = data["protein"][self._embedding_key]

        # Infer dimension on first forward
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.size(-1)

        V = embeddings.new_empty(embeddings.size(0), 0, 3)
        return embeddings, V, None

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> CachedEmbeddingEncoder:
        """
        Construct CachedEmbeddingEncoder from config dict.

        Args:
            config: Configuration dictionary with:
                - encoder_type: 'esm' or 'slae' (required)
                - embedding_key: Optional key name (defaults to 'embedding')
                - embedding_dim: Optional embedding dimension (if known upfront)
            device: Device to place the encoder on

        Returns:
            Instantiated CachedEmbeddingEncoder
        """
        encoder_type = config["encoder_type"]  # "esm" or "slae"
        embedding_key = config.get("embedding_key", "embedding")
        embedding_dim = config.get("embedding_dim")
        return cls(embedding_key, encoder_type, embedding_dim).to(device)
