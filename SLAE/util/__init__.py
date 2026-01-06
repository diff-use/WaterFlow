"""
Utility functions and helpers for SLAE.

This module provides various utility functions including:
- types: Type definitions and data structures
- base_model: Base model classes
- embedding_extractor: Tools for extracting embeddings
- constants: Protein structure constants
- memory_utils: Memory management utilities
"""

from SLAE.util._irrep_utils import _fix_irreps_dict, _irreps_compatible
from SLAE.util.types import ModelOutput
from SLAE.nn.base_model import BaseModel
from SLAE.util.memory_utils import clean_up_torch_gpu_memory
from SLAE.nn.mlp_decoder import MLPDecoder, PositionDecoder

__all__ = [
    # Irrep utilities
    "_fix_irreps_dict",
    "_irreps_compatible",

    # Types
    "ModelOutput",


    # Base classes
    "BaseModel",

    # Utilities
    "clean_up_torch_gpu_memory",
    "MLPDecoder",
    "PositionDecoder",
]
