"""
SLAE: Structure-based Latent All-atom Encoder
==============================================

A deep learning framework for protein structure analysis using full-atom representations.

Main Components:
- datasets: Data loading and preprocessing
- features: Feature extraction and representation
- model: Neural network architectures (encoder, decoder, autoencoder)
- nn: Neural network building blocks
- util: Utility functions and helpers
- loss: Loss functions
"""

__version__ = "0.1.0"

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Core model imports for convenience
from SLAE.model.encoder import ProteinEncoder
from SLAE.model.autoencoder import AutoEncoderModel
from SLAE.model.decoder import AllAtomDecoder

# Dataset imports
from SLAE.datasets.dataset import ProteinDataset
from SLAE.datasets.dataloader import ProteinDataLoader
from SLAE.datasets.datamodule import PDBDataModule

# Feature imports
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.features.fa_representation import transform_representation_fa

# Utility imports
from SLAE.util.embedding_extractor import EmbeddingExtractor, extract_embeddings_simple

__all__ = [
    # Models
    "ProteinEncoder",
    "AutoEncoderModel",
    "AllAtomDecoder",

    # Datasets
    "ProteinDataset",
    "ProteinDataLoader",
    "PDBDataModule",

    # Features
    "ProteinGraphFeaturizer",
    "transform_representation_fa",

    # Utilities
    "EmbeddingExtractor",
    "extract_embeddings_simple",
]
