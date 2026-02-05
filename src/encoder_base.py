# encoder_base.py
"""
Base encoder interface and registry for modular encoder architecture.

This module provides:
- BaseProteinEncoder: Abstract base class that all encoders must implement
- Registry pattern: Decorator-based registration and build_encoder() function
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class BaseProteinEncoder(ABC, nn.Module):
    """
    Abstract base class for protein encoders.

    All encoder implementations must inherit from this class and implement
    the required properties and methods.
    """

    @property
    @abstractmethod
    def output_dims(self) -> Tuple[int, int]:
        """Return (scalar_dim, vector_dim) output dimensions."""
        pass

    @property
    @abstractmethod
    def encoder_type(self) -> str:
        """Return encoder type identifier ('gvp', 'slae', etc.)."""
        pass

    @abstractmethod
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein data.

        Args:
            data: HeteroData with protein nodes

        Returns:
            Tuple of (s, V) where:
                s: (N, scalar_dim) scalar features
                V: (N, vector_dim, 3) vector features
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict, device: torch.device) -> 'BaseProteinEncoder':
        """
        Construct encoder from config dict.

        Args:
            config: Configuration dictionary with encoder parameters
            device: Device to place the encoder on

        Returns:
            Instantiated encoder
        """
        pass


# Global encoder registry
_ENCODER_REGISTRY: Dict[str, Type[BaseProteinEncoder]] = {}


def register_encoder(name: str):
    """
    Decorator to register an encoder class.

    Usage:
        @register_encoder('my_encoder')
        class MyEncoder(BaseProteinEncoder):
            ...
    """
    def decorator(cls: Type[BaseProteinEncoder]) -> Type[BaseProteinEncoder]:
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder '{name}' is already registered")
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder_class(name: str) -> Type[BaseProteinEncoder]:
    """
    Get encoder class by name.

    Args:
        name: Encoder type identifier

    Returns:
        Encoder class

    Raises:
        ValueError: If encoder name is not registered
    """
    if name not in _ENCODER_REGISTRY:
        available = list(_ENCODER_REGISTRY.keys())
        raise ValueError(f"Unknown encoder type '{name}'. Available: {available}")
    return _ENCODER_REGISTRY[name]


def build_encoder(config: Dict, device: torch.device) -> BaseProteinEncoder:
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
    encoder_type = config.get('encoder_type', 'gvp')
    encoder_cls = get_encoder_class(encoder_type)
    return encoder_cls.from_config(config, device)
