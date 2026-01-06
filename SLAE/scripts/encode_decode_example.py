from pathlib import Path
from typing import Dict

import torch
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# SLAE stack
from SLAE.model.encoder import ProteinEncoder
from SLAE.model.decoder import AllAtomDecoder
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.util.constants import FILL


ENCODER_CFG = "SLAE/configs/encoder/protein_encoder.yaml"
DECODER_CFG = "SLAE/configs/decoder/allatom_decoder.yaml"
CKPT = "checkpoints/autoencoder.ckpt"
DECODED_PDB = "../slae_test_files/slae_outputs/decoded_structure.pdb"

def load_yaml_dict(p: Path) -> dict:
    with open(p, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"YAML at {p} did not parse to a dict.")
    d.pop("_target_", None)
    return d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

# --- Data module in inference mode ---
dm = PDBDataModule(pdb_dir="../slae_test_files/slae_test_pdbs",  # set path where PDB files are stored
                   processed_dir="../slae_test_files/slae_processed", # set path where processed files are stored
                   inference_only=True, 
                   batch_size=1) # batch size 1 for inference

dm.setup(stage="lazy_init")
featuriser = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

loader = dm.inference_dataloader()

# --- Build encoder / decoder and load cleaned weights ---
enc_cfg = load_yaml_dict(Path(ENCODER_CFG))
encoder = ProteinEncoder(**enc_cfg).to(device).eval()

dec_cfg = load_yaml_dict(Path(DECODER_CFG))["allatom_decoder"]
dec_cfg.pop("_target_", None)
decoder = AllAtomDecoder(**dec_cfg).to(device).eval()

# Load weights
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)

# encoder weights are in ckpt["encoder"]
missing, unexpected = encoder.load_state_dict(ckpt["encoder"], strict=False)
#if missing or unexpected:
#    print(f"[encoder load_state] missing: {missing}\n[encoder load_state] unexpected: {unexpected}")

# decoder weights are in ckpt["decoder"]
missing, unexpected = decoder.load_state_dict(ckpt["decoder"], strict=False)
#if missing or unexpected:
#    print(f"[decoder load_state] missing: {missing}\n[decoder load_state] unexpected: {unexpected}")

# --- Get a batch and encode ---
out: Dict[str, torch.Tensor] = {}
batch = next(iter(loader))
batch = batch.to(device)
# featurize batch
batch = featuriser(batch)

print(f"Loaded batch containing PDB ID: {batch.id[0]}")

# find which entry in batch.coords is NaN
# coords is [batch_size, N, 37, 3]
nan_mask = torch.isnan(batch.coords).any(dim=(-1, -2, -3))  # [batch_size, N]
if nan_mask.any():
    nan_indices = nan_mask.nonzero(as_tuple=False)
    print(f"NaN found in batch at indices: {nan_indices}")

enc_out = encoder(batch)

# residue embedding
emb = enc_out["residue_embedding"]  # [sum Li, D]
print(f"Embedding: {emb}")
# print(f"Average number of neighbors {g.avg_num_neighbors.item()}")

# --- Decode to all-atom coordinates ---
res_embedding = enc_out["residue_embedding"]  # [sum Li, D]

# --- Prepare inputs for decoder ---
residue_batch = batch.residue_batch  # [sum Li]
batch_size = residue_batch.max().item() + 1
lengths = torch.bincount(residue_batch, minlength=batch_size)  # [batch_size]

# Split the embeddings into a list of [seq_len_i, d_model]
seq_list = []
start = 0
for length in lengths:
    seq_len = length.item()
    seq_list.append(res_embedding[start : start + seq_len])
    start += seq_len

# Pad sequences to form [batch_size, max_seq_len, d_model]
res_embedding_padded = torch.nn.utils.rnn.pad_sequence(
    seq_list, batch_first=True, padding_value=FILL
)
max_seq_len = res_embedding_padded.size(1)

# Create a mask indicating valid positions
mask = torch.arange(max_seq_len, device=res_embedding_padded.device).unsqueeze(0) < lengths.unsqueeze(1)

decoder_output = decoder(
    res_embedding_padded,  # [batch_size, max_seq_len, d_model]
    mask,                  # [batch_size, max_seq_len]
    batch=batch.residue_batch,
    res_idx=batch.residue_id,
)

bb = decoder_output["backbone_coords"]   # [batch_size, max_seq_len, 4, 3]
sc = decoder_output["sidechain_coords"]  # [batch_size, max_seq_len, 33, 3]
atom_mask = decoder_output["atom_mask"].bool()  # [batch_size, N, 37]
allatom = torch.cat([bb, sc], dim=2)    # [batch_size, N, 37, 3]

# apply atom mask and send invalid atoms to filler
allatom0 = allatom[0].detach().cpu()                   # [N, 37, 3]
atom_mask0 = atom_mask[0].detach().cpu().unsqueeze(-1) # [N, 37, 1]
allatom0 = torch.where(atom_mask0, allatom0, torch.full_like(allatom0, FILL))

print(f"All-atom coords: {allatom0}")

# write to file
from SLAE.io.write_pdb import to_pdb
to_pdb(allatom0, DECODED_PDB)