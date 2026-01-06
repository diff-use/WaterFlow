"""
Neural network building blocks for SLAE.

This module provides various neural network components:
- Allegro_Module: E(3)-equivariant graph network
- AtomwiseReduce: Atom-wise reduction layers
- EdgewiseEnergySum: Edge-wise energy aggregation
- ScalarMLP: Multi-layer perceptron for scalar features
- TransformerStack: Transformer architecture components
- Dim6RotStructureHead: Structure prediction head
"""

from SLAE.nn._allegro_module import Allegro_Module
from SLAE.nn._edgewise import EdgewiseEnergySum
from SLAE.nn._atomwise import AtomwiseReduce
from SLAE.nn.mlp_encoder import ScalarMLP, ScalarMLPFunction
from SLAE.nn.transformer_stack import TransformerStack
from SLAE.nn.bb_proj import Dim6RotStructureHead

__all__ = [
    "Allegro_Module",
    "EdgewiseEnergySum",
    "AtomwiseReduce",
    "ScalarMLP",
    "ScalarMLPFunction",
    "TransformerStack",
    "Dim6RotStructureHead",
]
