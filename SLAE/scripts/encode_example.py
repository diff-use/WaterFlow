#SLAE encoder

from pathlib import Path
from typing import Dict

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from SLAE.model.encoder import ProteinEncoder
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.datasets.datamodule import PDBDataModule  
from SLAE.util.constants import FILL


ENCODER_CFG = "SLAE/configs/encoder/protein_encoder.yaml"
CKPT="checkpoints/autoencoder.ckpt"


def load_yaml_dict(p: Path) -> dict:
    with open(p, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"YAML at {p} did not parse to a dict.")
    d.pop("_target_", None)
    return d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

enc_cfg = load_yaml_dict(ENCODER_CFG)
encoder = ProteinEncoder(**enc_cfg).to(device).eval()
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"[debug] Checkpoint keys: {ckpt.keys()}")

enc_state = {
    k.replace("encoder.", ""): v
    for k, v in ckpt.items()
    if k.startswith("encoder.")
}

missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
if missing or unexpected:
    print(f"[load_state] missing: {missing}\n[load_state] unexpected: {unexpected}")

# --- Data module in inference mode ---
dm = PDBDataModule(pdb_dir="../slae_test_pdbs",  # set path where PDB files are stored
                   processed_dir="../slae_processed", # set path where processed files are stored
                   inference_only=True, 
                   batch_size=1) # batch size 1 for inference
dm.setup(stage="lazy_init")
featuriser = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

loader = dm.inference_dataloader()

# get first batch
batch = next(iter(loader))
batch = batch.to(device)
batch = featuriser(batch)

print(batch)

with torch.no_grad():
    embeddings = encoder(batch)
    print(embeddings.keys())

residue_embeddings = embeddings["residue_embedding"].cpu() # (N_residues, D_embedding=128)
node_embeddings = embeddings["node_embedding"].cpu()     

print(residue_embeddings.shape)
print(node_embeddings.shape)    
# graph and atom embeddings can also be accessed via embeddings["graph_embeddings"] and embeddings["atom_embeddings"]



