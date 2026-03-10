# constants.py
"""
Constants for edge types and other shared values.
"""

# Node feature dimensions
NODE_FEATURE_DIM = 16  # Default node scalar feature dimension

# RBF (Radial Basis Function) parameters
NUM_RBF = 16          # Number of RBF basis functions
RBF_CUTOFF = 8.0      # Distance cutoff in Angstroms for RBF encoding

# Edge type tuples: (src_node_type, edge_name, dst_node_type)
EDGE_PP = ("protein", "pp", "protein")  # protein -> protein
EDGE_WW = ("water", "ww", "water")  # water -> water
EDGE_PW = ("protein", "pw", "water")  # protein -> water
EDGE_WP = ("water", "wp", "protein")  # water -> protein

# all edge types used in the model (future support: het atoms)
ALL_EDGE_TYPES = [EDGE_PW, EDGE_WW, EDGE_PP, EDGE_WP]

# Standard 3-letter to 1-letter amino acid mapping
# Includes 20 canonical amino acids plus common non-standard residues
# Non-canonical residues not in this dict should be mapped to 'X'
THREE_TO_ONE = {
    # 20 canonical amino acids
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    # common non-standard amino acids
    "MSE": "M",  # Selenomethionine -> Methionine
    "SEC": "U",  # Selenocysteine
    "PYL": "O",  # Pyrrolysine
}

# Standard 1-to-3 letter mapping to feed sanitized residues back to ESM3.
# This acts as the inverse of THREE_TO_ONE, ensuring ESM3 recognizes 
# the atoms and safely maps true unknowns to 'UNK'. (probably a more efficient way to do this I know)
ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    "X": "UNK", "U": "SEC", "O": "PYL"
}
