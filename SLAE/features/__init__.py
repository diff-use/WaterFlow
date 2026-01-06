"""
Feature extraction and protein representation for SLAE.

This module provides:
- fa_factory: ProteinGraphFeaturizer for featurizing protein structures
- fa_representation: Functions for transforming protein representations
"""

from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.features.fa_representation import (
    transform_representation_fa,
)

__all__ = [
    "ProteinGraphFeaturizer",
    "transform_representation_fa",
]
