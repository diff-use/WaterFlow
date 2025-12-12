# inference.py

"""
Script that:
- given user defined txt file of paths to PDBs 
- run rk4/euler integration on checkpointed model
- create plot+gif+pdb file with predicted water positions and with/without mates (also save native pdb for comparison)
"""