# Data preparation

WaterFlow reads PDB or mmCIF files and preprocesses them into cached graph
representations. This document covers the input layout, split files, the cache
structure, and the quality filters.

## Input structure files

Structures live one per directory under `--base_pdb_dir`:

```
<base_pdb_dir>/
├── 1abc/
│   └── 1abc_final.cif      # .cif or .pdb
├── 2xyz/
│   └── 2xyz_final.pdb
└── ...
```

Each structure carries the `_final` suffix and contains protein atoms
(conditioning context) and water molecules (`HOH` residues, the ground truth).

**Format resolution:** split-file entries are bare IDs (`6eey_final`, no
extension). For each entry WaterFlow looks in `<base_pdb_dir>/<pdb_id>/` and
prefers `<pdb_id>_final.cif`, falling back to `<pdb_id>_final.pdb`. Both parse to
identical atom counts. If neither exists, reading raises an error naming the
missing path.

## Split files

Plain text, one entry per line:

```
# splits/train_list_0.95.txt
110m_final
1a2p_final
1a3h_final
```

## Structure parsing

- Biotite extracts protein atoms, waters (`HOH`), and ligands, dispatching on
  extension (`.cif` via `CIFFile`, otherwise `PDBFile`).
- "Ligand" means every non-protein, non-water heavy atom: small molecules, ions,
  cofactors, nucleic acids. Included by default; disable with `--no-include_ligands`.
- Modified residues are kept through parsing and geometry preprocessing. When ESM
  embeddings are generated, they are mapped to encoder-compatible identities
  (MSE→MET, SEC→SEC, etc.).
- Hydrogens are dropped. Only the first model is used. For alternate
  conformations, the highest-occupancy conformer is kept.

## Crystal symmetry mates

Included only with `--include_mates`; a no-mates cache never invokes PyMOL.

- PyMOL's `symexp` generates symmetry mates, keeping whole residues and whole
  ligand entities with any atom within the cutoff of the ASU. Protein and ligand
  mates are classified separately, so `is_ligand` stays exact for mate nodes.
- **Mate waters are never kept.** A mate water is a symmetry image of an ASU water
  — the thing the model predicts — so keeping it as context would leak the label.
- Symmetry can map an atom onto itself (special position) or reach one residue
  through two operators. Mate atoms within 0.3 Å of an ASU atom, a target water,
  or an already-kept mate atom are dropped; mate ligands are judged whole, so a
  ligand is never fragmented.
- A mate inherits its source residue's `(chain, res_id, ins_code)`, so it picks up
  that residue's ESM row through `emb_res_idx` instead of a zero vector, and it
  joins the distance-filter reference — a water in a crystal contact is not dropped
  as solvent-far.

## Cache structure

Preprocessed data is cached under `--processed_dir` in three layers:

```
<processed_dir>/
├── geometry/                       # graph structures (see naming below)
│   └── <pdb_id>_final.pt
│       - protein_pos      centered node coordinates (N, 3)
│       - protein_x        element one-hot (N, 16)
│       - protein_res_idx  residue indices for grouping
│       - is_ligand        mask marking ligand atoms (N,)
│       - is_mate          mask marking symmetry-mate atoms (N,)
│       - emb_res_idx      embedding row per atom; -1 = no row (N,)
│       - water_pos, water_x   water coordinates and features
│       - num_asu_protein  ASU protein atom count (mate boundary)
├── geometry/_filter_meta.json      # settings this directory was built with
├── esm/                            # ESM embeddings (per residue)
│   └── <pdb_id>_final.pt
│       - residue_embeddings (N_res, embed_dim)
│       - sequence, num_residues
└── slae/                           # SLAE embeddings (per atom, legacy)
    └── <pdb_id>_final.pt
```

The `protein_*` names predate mates and ligands: `N` is the total node count and
these arrays hold every node. Node order is
`[ASU protein | mate protein | ASU ligand | mate ligand]`, recovered with the two
masks:

```
ASU protein  = ~is_mate & ~is_ligand    # == the first num_asu_protein
mate protein =  is_mate & ~is_ligand
ASU ligand   = ~is_mate &  is_ligand
mate ligand  =  is_mate &  is_ligand
```

`emb_res_idx` indexes the ESM table: a mate atom carries the row of the ASU
residue it images, and every ligand carries `-1` (a zero row).

### Cache directory naming

The geometry directory name encodes the flags that change which nodes are cached,
so configs that produce different graphs never share a directory:

| `--include_mates` | `--include_ligands` | Directory |
|---|---|---|
| true | true (default) | `geometry_mates/` |
| true | false | `geometry_mates_noligands/` |
| false | true | `geometry/` |
| false | false | `geometry_noligands/` |

The base name comes from `--geometry_cache_name` (default `geometry`).

### Filter metadata

Filtering happens *before* the cache is written, so the thresholds are a property
of the directory, not of the run reading it — the `.pt` files record none of them.
Each geometry directory carries a `_filter_meta.json` recording the per-water
filters and their toggles, the structure-level checks
(`min_water_residue_ratio`, `max_com_dist`, `max_clash_fraction`, `clash_dist`,
`interface_dist_threshold`), and the graph parameters behind the cached PP edges
(`cutoff`, `max_neighbors`).

The first run with `preprocess=True` writes this file; every later run compares
against it and **refuses to start** on a mismatch rather than mixing differently
filtered entries into one directory. A disabled filter records `null`. Directories
built before this file existed load with a warning until a preprocessing run
stamps them — check your thresholds match before that first run.

### Cache generation notes

- The geometry cache is generated automatically when `preprocess=True` (default).
- ESM/SLAE caches require running the respective `generate_*_embeddings.py` script
  first.
- Preprocessing failures are logged to `<geometry_dir>/preprocessing_failures.log`.
- A cache file missing a field the loader reads (`is_ligand`, `is_mate`,
  `emb_res_idx`, …) raises `KeyError`. Delete the geometry directory and let it
  regenerate.

## Quality filtering

### Structure-level checks

These decide whether a structure is included at all:

| Parameter | Default | Description |
|---|---|---|
| `--max_com_dist` | `25.0` | Max protein–water center-of-mass distance (Å) |
| `--max_clash_fraction` | `0.05` | Max fraction of waters clashing with protein |
| `--clash_dist` | `2.0` | Distance threshold for a clash (Å) |
| `--min_water_residue_ratio` | `0.1` | Minimum waters-per-residue ratio |

### Per-water filters

These remove individual low-quality waters and can each be toggled off:

| Parameter | Default | Toggle | Description |
|---|---|---|---|
| `--max_protein_dist` | `5.0` | `--no_filter_by_distance` | Remove waters far from protein |
| `--min_edia` | `0.4` | `--no_filter_by_edia` | Remove waters with low EDIA scores |
| `--max_bfactor_zscore` | `2.0` | `--no_filter_by_bfactor` | Remove waters with high B-factor |

**EDIA** measures how well an atom's position is supported by the experimental
electron density map; higher is more reliable. EDIA data is read from
`<pdb_id>_final.json` in the same directory as the structure (from PDB-REDO).
Filtering is on by default; disable with `--no_filter_by_edia`.

## Feature encoding

- Element vocabulary (15 elements + "other" = 16 dims):
  `C N O S P SE MG ZN CA FE NA K CL F BR`.
- Edge features: RBF distance encoding (16 Bessel basis functions).
