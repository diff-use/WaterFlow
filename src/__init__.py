# src/__init__.py
"""
WaterFlow source package.

Exports the encoder registry and build function.
Importing this module triggers encoder registration.
"""

from .encoder_base import build_encoder, BaseProteinEncoder, register_encoder

from . import gvp_encoder
from . import slae
