# constants.py
"""
Constants for edge types and other shared values.

Using constants instead of string literals helps prevent typos and enables
IDE autocompletion and refactoring support.
"""

# Node feature dimensions
NODE_FEATURE_DIM = 16  # Default node scalar feature dimension

# RBF (Radial Basis Function) parameters
NUM_RBF = 16          # Number of RBF basis functions
RBF_CUTOFF = 8.0      # Distance cutoff in Angstroms for RBF encoding

# Edge type tuples: (src_node_type, edge_name, dst_node_type)
EDGE_PP = ('protein', 'pp', 'protein')  # protein -> protein
EDGE_WW = ('water', 'ww', 'water')      # water -> water
EDGE_PW = ('protein', 'pw', 'water')    # protein -> water
EDGE_WP = ('water', 'wp', 'protein')    # water -> protein

# All edge types used in the model
ALL_EDGE_TYPES = [EDGE_PW, EDGE_WW, EDGE_PP, EDGE_WP]
