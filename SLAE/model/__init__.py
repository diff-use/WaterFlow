"""
Model architectures for SLAE.

This module contains the core neural network architectures:
- encoder: SLAE encoder architecture
- decoder: All-atom decoder
- autoencoder: Complete autoencoder model
- cs: Chemical shift prediction models
"""

from SLAE.model.encoder import ProteinEncoder
from SLAE.model.autoencoder import AutoEncoderModel
from SLAE.model.decoder import AllAtomDecoder

__all__ = [
    "ProteinEncoder",
    "AutoEncoderModel",
    "AllAtomDecoder",
]
