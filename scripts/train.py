# train.py

"""
Script that:
- loads in sharded dataset of train and validation graphs of protein+water+mates
- loads in ProteinGVPEnccoder from either pre-trained checkpoint or blank
- trains water matching model to predict water positions with toggles for self c and path distortion
- saves best model checkpoint based on validation loss
- runs rk4 integration on randomly sampled graphs every N epochs and saves plots+gifs
- tracked with wandb
"""