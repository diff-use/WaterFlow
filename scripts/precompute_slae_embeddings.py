#!/usr/bin/env python3
"""
Precompute SLAE embeddings for protein structures and save to cache files.

This script:
1. Loads a trained SLAE encoder from checkpoint
2. Iterates over all cached protein structures
3. For each structure:
   - Parses protein atoms from PDB
   - Converts to atom37 representation
   - Runs SLAE encoder to get node embeddings
   - Saves embeddings and atom37 data back to cache

Usage:
    python scripts/precompute_slae_embeddings.py \
        --processed_dir /path/to/cache \
        --base_pdb_dir /path/to/pdbs \
        --slae_ckpt checkpoints/autoencoder.ckpt \
        --slae_config SLAE/configs/encoder/protein_encoder.yaml
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import Data
from tqdm import tqdm
import yaml
import numpy as np

# Biotite for PDB parsing
import biotite.structure as bts
from biotite.structure.io.pdb import PDBFile, get_structure

# SLAE imports
from SLAE.model.encoder import ProteinEncoder
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.io.atom_tensor import atomarray_to_tensors, atom37_to_atoms


def load_yaml_dict(config_path):
    """Load YAML config and return as dict."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config.pop('_target_', None)
    return config


def parse_pdb_to_atom37(pdb_path, chain_filter=None):
    """
    Parse PDB file and convert protein atoms to atom37 representation.

    Args:
        pdb_path: Path to PDB file
        chain_filter: Optional list of chain IDs to filter

    Returns:
        coords: (N_res, 37, 3) atom37 coordinates
        residue_type: (N_res,) residue type indices
        chains: (N_res,) chain IDs (object dtype)
        residue_id: (N_res,) residue IDs
        protein_atoms: biotite AtomArray for reference
    """
    pdb_file = PDBFile.read(pdb_path)
    atoms = get_structure(pdb_file, model=1, altloc="occupancy")

    if chain_filter is not None:
        mask = np.isin(atoms.chain_id, np.array(chain_filter, dtype=atoms.chain_id.dtype))
        atoms = atoms[mask]

    # Filter hydrogen and non-protein atoms
    atoms = atoms[atoms.element != "H"]
    protein_mask = bts.filter_canonical_amino_acids(atoms)
    protein_atoms = atoms[protein_mask]

    # Convert to atom37 representation
    coords, residue_type, chains, residue_id = atomarray_to_tensors(protein_atoms)

    return coords, residue_type, chains, residue_id, protein_atoms


def compute_slae_embeddings(coords, residue_type, encoder, featurizer, device):
    """
    Compute SLAE node embeddings from atom37 coordinates.

    Args:
        coords: (N_res, 37, 3) atom37 coordinates
        residue_type: (N_res,) residue type indices
        encoder: SLAE ProteinEncoder
        featurizer: ProteinGraphFeaturizer
        device: torch device

    Returns:
        embeddings: (N_atoms, 128) node embeddings
    """
    # Convert atom37 to flat atom representation
    pos, residue_index, atom_type = atom37_to_atoms(coords)

    # Create PyG Data object
    batch_idx = torch.zeros(residue_index.size(0), dtype=torch.long)
    data = Data(
        pos=pos,
        residue_index=residue_index,
        atom_type=atom_type,
        batch=batch_idx,
        residue_type=residue_type,
    )

    # Move to device and build graph
    data = data.to(device)
    data = featurizer(data)

    # Run encoder
    with torch.no_grad():
        outputs = encoder(data)
        embeddings = outputs['node_embedding']  # (N_atoms, 128)

    return embeddings.cpu()


def process_cache_file(cache_path, pdb_path, chain_id, encoder, featurizer, device):
    """
    Process a single cached file: compute SLAE embeddings and save.

    Args:
        cache_path: Path to cached .pt file
        pdb_path: Path to PDB file
        chain_id: Chain ID to process
        encoder: SLAE encoder
        featurizer: Graph featurizer
        device: torch device

    Returns:
        success: bool
    """
    try:
        # Load existing cache
        cached = torch.load(cache_path, weights_only=False)

        # Parse PDB and get atom37 representation for ASU protein
        coords, residue_type, chains, residue_id, protein_atoms = parse_pdb_to_atom37(
            pdb_path, chain_filter=[chain_id]
        )

        # Compute embeddings for ASU protein
        asu_embeddings = compute_slae_embeddings(
            coords, residue_type, encoder, featurizer, device
        )

        # Save atom37 data and embeddings for ASU
        cached['protein_atom37_coords'] = coords
        cached['protein_residue_type'] = residue_type
        cached['protein_chains'] = chains
        cached['protein_residue_id'] = residue_id
        cached['protein_slae_embedding'] = asu_embeddings

        # Handle symmetry mates if they exist
        if 'mate_pos' in cached and cached['mate_pos'].size(0) > 0:
            # For mates, we need to reconstruct atom37 from positions
            # This is a simplified approach - we'll compute embeddings from mate positions
            # by matching them back to the original structure

            # Load mate info
            mate_pos = cached['mate_pos']  # (N_mate_atoms, 3)
            mate_x = cached['mate_x']      # (N_mate_atoms, 16)

            # For now, we'll skip mate embeddings and compute them on-the-fly
            # A more sophisticated approach would parse the full crystal structure
            # and compute embeddings for all symmetry-related atoms
            cached['mate_slae_embedding'] = torch.zeros(mate_pos.size(0), 128)

            print(f"  Warning: Mate embeddings set to zero for {cache_path.name}")
            print(f"  Consider implementing full crystal structure parsing for accurate mate embeddings")

        # Save updated cache
        torch.save(cached, cache_path)
        return True

    except Exception as e:
        print(f"  Error processing {cache_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute SLAE embeddings")
    parser.add_argument("--processed_dir", type=str, required=True,
                       help="Directory containing cached .pt files")
    parser.add_argument("--base_pdb_dir", type=str,
                       default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data",
                       help="Base directory containing PDB files")
    parser.add_argument("--slae_ckpt", type=str,
                       default="checkpoints/autoencoder.ckpt",
                       help="Path to SLAE checkpoint")
    parser.add_argument("--slae_config", type=str,
                       default="SLAE/configs/encoder/protein_encoder.yaml",
                       help="Path to SLAE encoder config")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_limit", type=int, default=None,
                       help="Limit number of files to process (for testing)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing embeddings")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SLAE encoder
    print("Loading SLAE encoder...")
    enc_config = load_yaml_dict(args.slae_config)
    encoder = ProteinEncoder(**enc_config).to(device).eval()

    ckpt = torch.load(args.slae_ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"], strict=False)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Loaded encoder from {args.slae_ckpt}")

    # Create featurizer
    featurizer = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

    # Find all cache files
    processed_dir = Path(args.processed_dir)
    cache_files = sorted(processed_dir.glob("*.pt"))
    print(f"Found {len(cache_files)} cache files")

    if args.batch_limit:
        cache_files = cache_files[:args.batch_limit]
        print(f"Processing first {args.batch_limit} files")

    # Filter files that need processing
    if not args.overwrite:
        to_process = []
        for cache_path in cache_files:
            cached = torch.load(cache_path, weights_only=False)
            if 'protein_slae_embedding' not in cached:
                to_process.append(cache_path)
        cache_files = to_process
        print(f"{len(cache_files)} files need SLAE embeddings")

    # Process each cache file
    success_count = 0
    for cache_path in tqdm(cache_files, desc="Computing embeddings"):
        # Parse cache key to get PDB info
        # Expected format: {pdb_id}_final_{chain_id}.pt
        cache_key = cache_path.stem
        parts = cache_key.split('_')
        if len(parts) < 3:
            print(f"  Warning: Skipping malformed cache key: {cache_key}")
            continue

        pdb_id = parts[0]
        chain_id = parts[-1]

        # Construct PDB path
        pdb_path = Path(args.base_pdb_dir) / pdb_id / f"{pdb_id}_final.pdb"
        if not pdb_path.exists():
            print(f"  Warning: PDB file not found: {pdb_path}")
            continue

        # Process
        success = process_cache_file(
            cache_path, str(pdb_path), chain_id,
            encoder, featurizer, device
        )

        if success:
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(cache_files)} files processed successfully")


if __name__ == "__main__":
    main()
