# src/__init__.py
"""
WaterFlow source package.

Exports the encoder registry and build function.
Importing this module triggers encoder registration.
"""

from src import esm_encoder as esm_encoder, gvp_encoder as gvp_encoder, slae as slae
from src.encoder_base import (
    BaseProteinEncoder as BaseProteinEncoder,
    build_encoder as build_encoder,
    register_encoder as register_encoder,
)
