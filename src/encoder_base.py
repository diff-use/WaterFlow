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

from src.constants import NODE_FEATURE_DIM


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
    Fusion encoder for pre-computed protein embeddings (ESM, SLAE, etc.).

    Each protein-type node is described by two modalities, both already present
    on data['protein']:
      - a cached sequence/structure embedding (`embedding`, width embedding_dim).
        ASU protein atoms carry the real ESM/SLAE vector; symmetry mates and
        ligand atoms are zero-padded (they have no residue embedding).
      - a per-atom element one-hot (`x`, width NODE_FEATURE_DIM).

    The two are each projected to fusion_dim, concatenated, and passed through a
    small MLP to produce per-node scalar features. This gives every atom -- not
    just ligands -- its own element identity (ESM is per-residue, so all atoms of
    a residue otherwise share an identical vector), and handles ligands/mates
    uniformly: their zero-ESM rows simply contribute nothing from esm_proj, so
    their fused features are element-driven. No ligand/mate special-casing.

    Supported embedding types:
    - ESM: Evolutionary Scale Modeling embeddings (https://github.com/evolutionaryscale/esm)
    - SLAE: Strictly Local Atom-level Environment Embeddings (https://www.biorxiv.org/content/10.1101/2025.10.03.680398v1)

    Memory: Cached embeddings are NOT loaded at initialization. The encoder
    stores only the key name; actual embeddings are read from data at forward
    time, allowing standard PyTorch batching/streaming.

    Note: Returns empty vector features (shape Nx0x3); fused output is scalar-only.
    """

    def __init__(
        self,
        embedding_key: str,
        encoder_type: str,
        embedding_dim: int,
        fusion_dim: int,
    ):
        """
        Initialize CachedEmbeddingEncoder.

        Args:
            embedding_key: Key to look up embeddings in data['protein']
            encoder_type: Encoder type identifier ('esm' or 'slae')
            embedding_dim: Width of the cached embeddings (e.g. 1536 for ESM,
                128 for SLAE). Validated against the data at forward time.
            fusion_dim: Output width of the fused scalar features (output_dims[0]).
        """
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._fusion_dim: int = fusion_dim
        self._embedding_key = embedding_key
        self._encoder_type = encoder_type
        # Project each modality to fusion_dim, LayerNorm each stream, then fuse.
        # Separate projections + per-stream norm keep the 16-dim element signal from
        # being swamped by the wide, large-norm ESM vector (raw ESM norms are ~1e3-1e4
        # vs 1 for a one-hot), so both modalities enter the fuse MLP at comparable scale.
        self.esm_proj = nn.Linear(embedding_dim, fusion_dim)
        self.elem_proj = nn.Linear(NODE_FEATURE_DIM, fusion_dim)
        self.esm_norm = nn.LayerNorm(fusion_dim)
        self.elem_norm = nn.LayerNorm(fusion_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

    @property
    def output_dims(self) -> tuple[int, int]:
        """Return (fusion_dim, 0) — scalars only."""
        return self._fusion_dim, 0

    @property
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        return self._encoder_type

    def forward(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """
        Fuse the cached embedding with the element one-hot and return (s, V, None).

        Args:
            data: HeteroData with cached embeddings and element features in
                data['protein'] ('embedding' and 'x').

        Returns:
            s: (N, fusion_dim) — fused scalar features
            V: (N, 0, 3)       — empty vector features
            pp_edge_attr: None — cached embedding encoders don't process edges
        """
        if self._embedding_key not in data["protein"]:
            raise KeyError(
                f"{self._encoder_type.upper()} encoder requires cached embeddings. "
                f"Please provide pre-computed '{self._embedding_key}' in data['protein']."
            )

        embeddings = data["protein"][self._embedding_key]

        # Validate cached width against the configured embedding_dim, otherwise a
        # mismatched cache fails inside esm_proj with an opaque shape error.
        if embeddings.size(-1) != self._embedding_dim:
            raise ValueError(
                f"{self._encoder_type.upper()} encoder configured with "
                f"embedding_dim={self._embedding_dim}, but cached "
                f"'{self._embedding_key}' embeddings have width {embeddings.size(-1)}. "
                f"Ensure the encoder config matches the cached embeddings."
            )

        x = data["protein"].x.to(embeddings.device)
        esm = self.esm_norm(self.esm_proj(embeddings))
        elem = self.elem_norm(self.elem_proj(x))
        fused = self.fuse(torch.cat([esm, elem], dim=-1))

        V = fused.new_empty(fused.size(0), 0, 3)
        return fused, V, None

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> CachedEmbeddingEncoder:
        """
        Construct CachedEmbeddingEncoder from config dict.

        Args:
            config: Configuration dictionary with:
                - encoder_type: 'esm' or 'slae' (required)
                - embedding_key: Optional key name (defaults to 'embedding')
                - embedding_dim: Cached embedding width (required)
                - hidden_s: Fused output width / scalar hidden dim (required)
            device: Device to place the encoder on

        Returns:
            Instantiated CachedEmbeddingEncoder

        Raises:
            ValueError: If 'embedding_dim' or 'hidden_s' is missing from config
        """
        encoder_type = config["encoder_type"]  # "esm" or "slae"
        embedding_key = config.get("embedding_key", "embedding")
        embedding_dim = config.get("embedding_dim")
        if embedding_dim is None:
            raise ValueError(
                f"'{encoder_type}' encoder requires 'embedding_dim' in its config."
            )
        fusion_dim = config.get("hidden_s")
        if fusion_dim is None:
            raise ValueError(
                f"'{encoder_type}' encoder requires 'hidden_s' (fused output width) "
                f"in its config."
            )
        return cls(embedding_key, encoder_type, embedding_dim, fusion_dim).to(device)
