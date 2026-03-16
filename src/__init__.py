# src/__init__.py
"""
WaterFlow source package.

Exports the encoder registry and build function.
Importing this module triggers encoder registration.
"""

from src import gvp_encoder
from src.encoder_base import (
    BaseProteinEncoder as BaseProteinEncoder,
    CachedEmbeddingEncoder as CachedEmbeddingEncoder,
    build_encoder as build_encoder,
    register_encoder as register_encoder,
)
