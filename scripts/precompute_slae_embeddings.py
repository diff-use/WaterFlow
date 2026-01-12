"""
Precompute SLAE embeddings for protein structures and save to cache files.

This script:
1. Loads a trained SLAE encoder from a checkpoint
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
        --slae_config ../SLAE/configs/encoder/protein_encoder.yaml
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # For SLAE imports

import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import yaml
import numpy as np

import biotite.structure as bts
from biotite.structure.io.pdb import PDBFile, get_structure

from SLAE.model.encoder import ProteinEncoder
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.io.atom_tensor import atomarray_to_tensors, atom37_to_atoms


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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

    atoms = atoms[atoms.element != "H"]
    protein_mask = bts.filter_canonical_amino_acids(atoms)
    protein_atoms = atoms[protein_mask]

    #convert to atom37 representation
    coords, residue_type, chains, residue_id = atomarray_to_tensors(protein_atoms)

    return coords, residue_type, chains, residue_id, protein_atoms


def compute_slae_embeddings_batch(coords_list, residue_type_list, residue_id_list, encoder, featurizer, device):
    """
    Compute SLAE node embeddings from atom37 coordinates for a batch of structures.

    Args:
        coords_list: List of (N_res, 37, 3) atom37 coordinates tensors
        residue_type_list: List of (N_res,) residue type indices tensors
        residue_id_list: List of (N_res,) residue ID tensors
        encoder: SLAE ProteinEncoder
        featurizer: ProteinGraphFeaturizer
        device: torch device

    Returns:
        embeddings_list: List of (N_atoms, 128) node embeddings tensors
    """
    # create Data objects for all structures
    data_list = []
    num_atoms_list = []

    for coords, residue_type, residue_id in zip(coords_list, residue_type_list, residue_id_list):
        # create PyG Data object with atom37 coords (featurizer will convert to flat)
        data = Data(
            coords=coords,
            residue_type=residue_type,
            residue_id=residue_id,
        )
        data_list.append(data)

        # track number of atoms for later splitting (after featurizer converts to flat)
        pos, _, _ = atom37_to_atoms(coords)
        num_atoms_list.append(pos.size(0))

    batch = Batch.from_data_list(data_list)
    batch = batch.to(device)
    batch = featurizer(batch)

    with torch.no_grad():
        outputs = encoder(batch)
        embeddings = outputs['node_embedding']  # (total_atoms, 128)

    # split embeddings back into individual structures
    embeddings_list = []
    start_idx = 0
    for num_atoms in num_atoms_list:
        embeddings_list.append(embeddings[start_idx:start_idx + num_atoms].cpu())
        start_idx += num_atoms

    return embeddings_list


def prepare_cache_data(cache_path, pdb_path, chain_filter=None):
    """
    Load cache and parse PDB to prepare data for embedding computation.

    Args:
        cache_path: Path to cached .pt file
        pdb_path: Path to PDB file
        chain_filter: Optional list of chain IDs to filter

    Returns:
        Tuple of (cached_dict, coords, residue_type, chains, residue_id) or None if error
    """
    try:
        cached = torch.load(cache_path, weights_only=False)
        coords, residue_type, chains, residue_id, protein_atoms = parse_pdb_to_atom37(
            pdb_path, chain_filter=chain_filter
        )

        return cached, coords, residue_type, chains, residue_id

    except Exception as e:
        print(f"  Error preparing {cache_path.name}: {e}")
        return None


def save_cache_with_embeddings(cache_path, cached, coords, residue_type, chains, residue_id, embeddings):
    """
    Save cache file with computed embeddings.

    Args:
        cache_path: Path to cached .pt file
        cached: Loaded cache dictionary
        coords: atom37 coordinates
        residue_type: residue type indices
        chains: chain IDs
        residue_id: residue IDs
        embeddings: computed SLAE embeddings
    """
    try:
        # save atom37 data and embeddings for ASU
        cached['protein_atom37_coords'] = coords
        cached['protein_residue_type'] = residue_type
        cached['protein_chains'] = chains
        cached['protein_residue_id'] = residue_id
        cached['protein_slae_embedding'] = embeddings

        # set mate embeddings to zero (the residue index here becomes annoying wrt slae, will fix in future pr)
        if 'mate_pos' in cached and cached['mate_pos'].size(0) > 0:
            mate_pos = cached['mate_pos']
            cached['mate_slae_embedding'] = torch.zeros(mate_pos.size(0), 128)

        torch.save(cached, cache_path)
        return True

    except Exception as e:
        print(f"  Error saving {cache_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute SLAE embeddings")
    parser.add_argument("--processed_dir", type=str, required=True,
                       help="Directory containing cached .pt files")
    parser.add_argument("--base_pdb_dir", type=str,
                       default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data",
                       help="Base directory containing PDB files")
    parser.add_argument("--slae_ckpt", type=str,
                       default="/home/srivasv/SLAE_Internal/checkpoints/autoencoder.ckpt",
                       help="Path to SLAE checkpoint")
    parser.add_argument("--slae_config", type=str,
                       default="../SLAE/configs/encoder/protein_encoder.yaml",
                       help="Path to SLAE encoder config")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch_limit", type=int, default=None,
                       help="Limit number of files to process (for testing)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing embeddings")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of structures to process in each batch")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load SLAE encoder
    print("Loading SLAE encoder...")
    enc_config = load_yaml_dict(args.slae_config)
    encoder = ProteinEncoder(**enc_config).to(device).eval()

    ckpt = torch.load(args.slae_ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"], strict=False)

    for p in encoder.parameters():
        p.requires_grad = False

    print(f"Loaded encoder from {args.slae_ckpt}")

    # create featurizer
    featurizer = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

    # find all cache files
    processed_dir = Path(args.processed_dir)
    cache_files = sorted(processed_dir.glob("*.pt"))
    print(f"Found {len(cache_files)} cache files")

    if args.batch_limit:
        cache_files = cache_files[:args.batch_limit]
        print(f"Processing first {args.batch_limit} files")

    # filter files that need processing
    if not args.overwrite:
        to_process = []
        for cache_path in cache_files:
            cached = torch.load(cache_path, weights_only=False)
            if 'protein_slae_embedding' not in cached:
                to_process.append(cache_path)
        cache_files = to_process
        print(f"{len(cache_files)} files need SLAE embeddings")

    # process cache files in batches
    success_count = 0
    total_batches = (len(cache_files) + args.batch_size - 1) // args.batch_size

    for batch_idx, cache_batch in enumerate(tqdm(
        list(chunks(cache_files, args.batch_size)),
        desc="Computing embeddings",
        total=total_batches
    )):
        batch_data = []
        batch_info = []

        for cache_path in cache_batch:
            cache_key = cache_path.stem
            parts = cache_key.split('_')
            if len(parts) < 2:
                print(f"  Warning: Skipping malformed cache key: {cache_key}")
                continue

            pdb_id = parts[0]
            chain_id = parts[-1] if len(parts) >= 3 else None

            # construct PDB path
            pdb_path = Path(args.base_pdb_dir) / pdb_id / f"{pdb_id}_final.pdb"
            if not pdb_path.exists():
                print(f"  Warning: PDB file not found: {pdb_path}")
                continue

            chain_filter = [chain_id] if chain_id and chain_id != "final" else None
            result = prepare_cache_data(cache_path, str(pdb_path), chain_filter)

            if result is not None:
                cached, coords, residue_type, chains, residue_id = result
                batch_data.append((coords, residue_type, residue_id))
                batch_info.append((cache_path, cached, coords, residue_type, chains, residue_id))

        # compute embeddings for batch
        if batch_data:
            try:
                coords_list = [item[0] for item in batch_data]
                residue_type_list = [item[1] for item in batch_data]
                residue_id_list = [item[2] for item in batch_data]

                embeddings_list = compute_slae_embeddings_batch(
                    coords_list, residue_type_list, residue_id_list, encoder, featurizer, device
                )

                for (cache_path, cached, coords, residue_type, chains, residue_id), embeddings in zip(batch_info, embeddings_list):
                    success = save_cache_with_embeddings(
                        cache_path, cached, coords, residue_type, chains, residue_id, embeddings
                    )
                    if success:
                        success_count += 1

            except Exception as e:
                print(f"\n  Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nCompleted: {success_count}/{len(cache_files)} files processed successfully")


if __name__ == "__main__":
    main()
