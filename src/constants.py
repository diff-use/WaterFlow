# constants.py
"""
Constants for edge types and other shared values.
"""

# feature dimensions
NUM_RBF = 16  # RBF basis functions for edge features
NODE_FEATURE_DIM = 16  # Element one-hot encoding dimension (15 elements + 1 "other")
RBF_CUTOFF = 8.0  # Distance cutoff for RBF encoding (Angstroms)

# Edge type tuples: (src_node_type, edge_name, dst_node_type)
EDGE_PP = ("protein", "pp", "protein")  # protein -> protein
EDGE_WW = ("water", "ww", "water")  # water -> water
EDGE_PW = ("protein", "pw", "water")  # protein -> water
EDGE_WP = ("water", "wp", "protein")  # water -> protein

# all edge types used in the model (future support: het atoms)
ALL_EDGE_TYPES = [EDGE_PW, EDGE_WW, EDGE_PP, EDGE_WP]
