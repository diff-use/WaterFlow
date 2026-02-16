# src/__init__.py
"""
WaterFlow source package.

Exports the encoder registry and build function.
Importing this module triggers encoder registration.
"""

from src import gvp_encoder, slae
from src.encoder_base import BaseProteinEncoder, build_encoder, register_encoder
