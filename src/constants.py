# constants.py
"""
Constants for edge types and other shared values.

Using constants instead of string literals helps prevent typos and enables
IDE autocompletion and refactoring support.
"""

# Edge type tuples: (src_node_type, edge_name, dst_node_type)
EDGE_PP = ('protein', 'pp', 'protein')  # protein -> protein
EDGE_WW = ('water', 'ww', 'water')      # water -> water
EDGE_PW = ('protein', 'pw', 'water')    # protein -> water
EDGE_WP = ('water', 'wp', 'protein')    # water -> protein

# All edge types used in the model
ALL_EDGE_TYPES = [EDGE_PW, EDGE_WW, EDGE_PP, EDGE_WP]
