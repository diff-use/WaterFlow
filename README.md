# WaterFlow

Predicting water molecule placements on protein surfaces using flow matching conditioned on learned protein structure embeddings.

## Project Structure

```
WaterFlow/
├── src/                    # Core library code
│   ├── dataset.py          # ProteinWaterDataset and data loading
│   ├── flow.py             # FlowMatcher and FlowWaterGVP model
│   ├── gvp.py              # Geometric Vector Perceptron layers
│   ├── gvp_encoder.py      # GVP-based protein encoder
│   ├── encoder_base.py     # Encoder registry and factory (includes ESM/SLAE)
│   ├── constants.py        # Shared constants (RBF bins, etc.)
│   └── utils.py            # Metrics, plotting, logging utilities
├── scripts/                # Executable scripts
│   ├── train.py            # Training pipeline
│   ├── inference.py        # Run inference on trained models
│   ├── generate_esm_embeddings.py   # Precompute ESM embeddings
│   └── generate_slae_embeddings.py  # Precompute SLAE embeddings
├── tests/                  # Test suite
│   ├── test_dataset.py     # Dataset and preprocessing tests
│   ├── test_flow.py        # Flow matching tests
│   ├── test_encoder.py     # Encoder tests
│   ├── test_forward.py     # End-to-end forward pass tests
│   ├── test_gvp.py         # GVP layer tests
│   ├── test_train_config.py # Training configuration tests
│   └── test_utils.py       # Utility function tests
└── splits/                 # Train/val/test split files
    ├── train_list_0.95.txt # Training set (95% of data)
    ├── valid_list_0.05.txt # Validation set (5% of data)
    └── water_pdbs.txt      # Full list of PDBs with waters
```

## Data Preparation

### Input Structure Files

WaterFlow reads PDB or mmCIF files, and expects them in a specific directory structure:

```
<base_pdb_dir>/
├── 1abc/
│   └── 1abc_final.cif      # .cif or .pdb
├── 2xyz/
│   └── 2xyz_final.pdb
└── ...
```

Each structure should have the `_final` suffix and contain:
- Protein atoms (used as conditioning context)
- Water molecules (HOH residues, used as ground truth)

**Format resolution:** entries in a split file are bare IDs (`6eey_final`) with no extension.
For each entry WaterFlow looks in `<base_pdb_dir>/<pdb_id>/` and **prefers
`<pdb_id>_final.cif` when it exists**, otherwise falls back to `<pdb_id>_final.pdb`. Both
formats parse to identical atom counts, so the choice does not change the resulting graph.
If neither file exists, reading the structure raises an error naming the missing path.

### Data Processing Pipeline

WaterFlow processes structure files through several stages to create training-ready graph representations:

**Structure Parsing**
- Uses Biotite to extract protein atoms, water molecules (HOH residues), and ligands, dispatching on file extension (`.cif` via `CIFFile`, otherwise `PDBFile`)
- "Ligand" means every non-protein, non-water heavy atom: small molecules, ions, cofactors, and nucleic acids. Included by default; disable with `--no-include_ligands`
- Modified residues are retained during structure parsing and geometry preprocessing
- When generating ESM embeddings, modified residues are mapped to encoder-compatible amino acid identities (e.g., MSE→M/MET, SEC→U/SEC)
- Hydrogen atoms are excluded
- Only the first model is used
- For atoms with alternate conformations, the highest-occupancy conformer is selected

**Crystal Contact Detection**
- Uses PyMOL's `symexp` to generate symmetry mates, keeping whole residues and whole ligand entities with any atom within the cutoff of the ASU. Runs only when `include_mates=True`; a no-mates cache never invokes PyMOL
- Protein mates and ligand mates are selected separately by PyMOL's own classifiers, so `is_ligand` stays exact for mate nodes too
- **Mate waters are never selected.** A mate water is a symmetry image of an ASU water, which is what the model predicts, so keeping it as context leaks the label
- Symmetry also maps atoms onto themselves (special positions) and reaches one residue through two operators. Mate atoms within 0.3Å of an ASU atom, a target water, or an already-kept mate atom are dropped (`dedup_mate_atoms`); mate ligands are judged whole, so a ligand is never fragmented (`dedup_mate_ligands_by_residue`)
- A mate keeps its source residue's `(chain, res_id, ins_code)`, so it inherits that residue's ESM row through `emb_res_idx` instead of a zero vector, and it joins the distance-filter reference so a water in a crystal contact — near a neighbour surface but far from the ASU — is not dropped as solvent-far

**Graph Representation**
- Node types: `protein` (ASU + symmetry mates + ligands), `water` (ground truth)
- Ligand atoms are appended after ASU and mate atoms and carry the boolean `is_ligand` mask plus `residue_index = -1` (they have no residue embedding, so residue pooling masks them out)
- `is_mate` marks every non-ASU node, protein or ligand. The flow prior anchors on `~is_mate` so sampled waters start where the targets live
- Edge types (defined in `src/constants.py`):
  - `('protein', 'pp', 'protein')`: protein-protein edges
  - `('protein', 'pw', 'water')`: protein to water
  - `('water', 'wp', 'protein')`: water to protein
  - `('water', 'ww', 'water')`: water-water edges
- Default edge cutoff: 8.0Å (`RBF_CUTOFF` in constants.py)

**Feature Encoding**
- Element vocabulary (15 elements + "other" bucket = 16 dims):
  `C, N, O, S, P, SE, MG, ZN, CA, FE, NA, K, CL, F, BR`
- Edge features: RBF distance encoding (16 Bessel basis functions)

### Split File Format

Split files are plain text with one PDB entry per line:

```
# Example: splits/train_list_0.95.txt
110m_final
1a2p_final
1a3h_final
```

### Cache Directory Structure

Preprocessed data is cached under `--processed_dir` in a three-layer architecture:

```
<processed_dir>/
├── geometry/              # Graph structures; see cache directory naming below
│   └── <pdb_id>_final.pt
│       - protein_pos: centered node coordinates (N, 3)
│       - protein_x: element one-hot encoding (N, 16)
│       - protein_res_idx: residue indices for grouping
│       - is_ligand: bool mask marking the ligand atoms (N,)
│       - is_mate: bool mask marking the symmetry-mate atoms (N,)
│       - emb_res_idx: embedding row per atom; -1 means no row (N,)
│       - water_pos, water_x: water coordinates and features
│       - num_asu_protein: ASU protein atom count (mate boundary metadata)
│       # The protein_* names predate mates and ligands: N is the total node
│       # count and these arrays hold every node, not just protein atoms (same
│       # for the data["protein"] node type). Select blocks with the masks.
│       #
│       # Node order is [ASU protein | mate protein | ASU ligand | mate ligand],
│       # so the two masks recover every block:
│       #   ASU protein  = ~is_mate & ~is_ligand    (== the first num_asu_protein)
│       #   mate protein =  is_mate & ~is_ligand
│       #   ASU ligand   = ~is_mate &  is_ligand
│       #   mate ligand  =  is_mate &  is_ligand
│       #
│       # emb_res_idx indexes the ESM table: mate atoms carry the row of the ASU
│       # residue they are a symmetry image of, and every ligand carries -1,
│       # which reads as a zero row.
├── <geometry_dir>/_filter_meta.json   # settings this directory was built with
├── esm/                   # ESM embeddings (per-residue)
│   └── <pdb_id>_final.pt
│       - residue_embeddings: ESM3 embeddings (N_res, embed_dim)
│       - sequence: extracted sequence string
│       - num_residues: residue count
└── slae/                  # SLAE embeddings (per-atom, 128-dim)
    └── <pdb_id>_final.pt
        - node_embeddings: atom-level embeddings aligned to geometry order
        - atom37_coords: standard atom37 coordinates (N_res, 37, 3)
```

**Cache Directory Naming:**

The geometry cache directory name encodes the flags that change which nodes get cached, so
configs that produce different graphs never share a directory:

| `--include_mates` | `--include_ligands` | Directory |
|---|---|---|
| true | true (default) | `geometry_mates/` |
| true | false | `geometry_mates_noligands/` |
| false | true | `geometry/` |
| false | false | `geometry_noligands/` |

The base name comes from `--geometry_cache_name` (default `geometry`).

**Filter Provenance:**

Filtering happens *before* the cache is written, so the thresholds are a property of the
directory, not of the run reading it — and the `.pt` files record none of them. Each geometry
directory therefore carries a `_filter_meta.json` sidecar holding the per-water filters and
their toggles, the structure-level checks that decide which entries exist at all
(`min_water_residue_ratio`, `max_com_dist`, `max_clash_fraction`, `clash_dist`,
`interface_dist_threshold`), and the graph parameters behind the cached PP edges (`cutoff`,
`max_neighbors`).

The first run with `preprocess=True` writes it; every later run compares against it and
**refuses to start** on a mismatch rather than appending differently filtered entries to the
same directory. A disabled filter records `null` for its threshold, which cannot have changed
the cached waters. Directories built before this existed have no sidecar: they load, and warn
that their provenance is unverifiable, until a preprocessing run stamps them — so check your
thresholds match the cache before that first run.

**Cache Generation Notes:**
- Geometry cache is generated automatically when `preprocess=True` (default)
- ESM/SLAE caches require running the respective `generate_*_embeddings.py` scripts first
- Preprocessing failures are logged to `<geometry_dir>/preprocessing_failures.log`
- A cache file missing any field the loader reads (`is_ligand`, `is_mate`, `emb_res_idx`, …)
  raises `KeyError`. Delete the geometry cache directory and let it regenerate

## Environment Setup

We use `uv` for our environment and package management, with Python 3.12.

You can install the environment by running `uv sync` and running the scripts with `uv run python <script>` (Recommended). 

Or if you want to install a fresh virtual environment from scratch, follow the steps below.

Installing the environment:

```bash
uv venv water --python 3.12
source water/bin/activate

uv pip install torch==2.8.0
uv pip install torch_geometric
uv pip install torch_cluster torch_scatter pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
uv pip install esm biotite pymol-open-source scipy pandas numpy matplotlib pillow loguru tqdm wandb e3nn
uv pip install pytest pytest-cov  # dev dependencies
```

If you have trouble installing torch_cluster or scatter, I would suggest changing the cuda version in the wheel.

## Model Architecture

WaterFlow uses a two-stage architecture:

1. **Protein Encoder**: Encodes protein structure into per-residue embeddings
2. **Flow Network**: Predicts velocity field for water molecule trajectories

### Encoder Types

| Encoder | Description | Precomputation Required |
|---------|-------------|------------------------|
| `gvp` | Geometric Vector Perceptron encoder that learns from 3D coordinates | No |
| `esm` | Uses ESM3 language model embeddings | Yes (`generate_esm_embeddings.py`) |
| `slae` | Uses SLAE ([Strictly Local All-Atom Environment](https://www.biorxiv.org/content/10.1101/2025.10.03.680398v1)) embeddings | Yes (`generate_slae_embeddings.py`) |

## Embedding Generation

For `esm` and `slae` encoder types, you must precompute embeddings before training or inference.

### ESM Embeddings (for `--encoder_type esm`)

```bash
uv run python -m scripts.generate_esm_embeddings \
    --split_file splits/water_pdbs.txt \
    --cache_dir ~/flow_cache/ \
    --device cuda:0
```

### SLAE Embeddings (for `--encoder_type slae`)

```bash
uv run python -m scripts.generate_slae_embeddings \
    --split_file splits/water_pdbs.txt \
    --cache_dir ~/flow_cache/ \
    --slae_ckpt /path/to/SLAE/checkpoints/autoencoder.ckpt
```

## Training

### GVP Encoder (no precomputed embeddings required)

```bash
uv run python -m scripts.train \
    --train_list splits/train_list_0.95.txt \
    --val_list splits/valid_list_0.05.txt \
    --encoder_type gvp \
    --batch_size 4
```

### ESM Encoder (requires precomputed ESM embeddings)

```bash
uv run python -m scripts.train \
    --train_list splits/train_list_0.95.txt \
    --val_list splits/valid_list_0.05.txt \
    --encoder_type esm \
    --batch_size 1 \
    --grad_accum_steps 4 \
    --processed_dir ~/flow_cache/
```

### Resuming from Checkpoints

To resume training from a checkpoint, you can load the model weights and optimizer state:

```bash
# Checkpoints are saved in <save_dir>/<run_name>/checkpoints/
# - best.pt: Best validation loss
# - epoch_N.pt: Periodic checkpoints every --save_every epochs
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_list` | required | Path to training split file |
| `--val_list` | required | Path to validation split file |
| `--encoder_type` | `gvp` | Encoder type: `gvp`, `esm`, or `slae` |
| `--batch_size` | `4` | Batch size (use smaller for ESM due to memory) |
| `--grad_accum_steps` | `1` | Gradient accumulation steps (effective batch = batch_size * grad_accum_steps) |
| `--flow_layers` | `3` | Number of flow GVP layers |
| `--hidden_s` | `256` | Scalar hidden dimension |
| `--hidden_v` | `64` | Vector hidden dimension |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--scheduler` | `cosine` | LR scheduler: `cosine`, `step`, or `none` |
| `--warmup_steps` | `0` | Linear warmup steps |
| `--processed_dir` | `~/flow_cache/` | Cache directory for preprocessed data |
| `--include_mates` | `false` | Include symmetry mate atoms as protein nodes |
| `--include_ligands` | `true` | Include ligand/ion/cofactor/nucleic acid heavy atoms as protein nodes. Negate with `--no-include_ligands` |
| `--save_dir` | `../flow_checkpoints` | Directory to save checkpoints |
| `--save_every` | `10` | Save checkpoint every N epochs |
| `--eval_every` | `5` | Run evaluation every N epochs |
| `--min_edia` | `0.4` | Minimum EDIA score threshold for waters |
| `--no_filter_by_edia` | - | Disable EDIA-based water filtering |

### Weights & Biases Logging

Training automatically logs to W&B. Configure with:

| Argument | Default | Description |
|----------|---------|-------------|
| `--wandb_project` | `water-flow` | W&B project name |
| `--wandb_dir` | `../wandb_logs` | Local W&B log directory |
| `--run_name` | auto-generated | Custom run name (format: `YYYYMMDD_HHMMSS_encoder_layers_hidden`) |

## Quality Filtering

WaterFlow applies multiple quality filters to ensure high-quality training data.

### Structure-Level Quality Checks

These checks determine whether a structure is included in training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_com_dist` | `25.0` | Max protein-water center-of-mass distance (A) |
| `--max_clash_fraction` | `0.05` | Max fraction of waters clashing with protein |
| `--clash_dist` | `2.0` | Distance threshold for clash detection (A) |
| `--min_water_residue_ratio` | `0.1` | Minimum waters per residue ratio |

### Per-Water Quality Filters

These filters remove individual low-quality waters (can be toggled):

| Parameter | Default | Toggle Flag | Description |
|-----------|---------|-------------|-------------|
| `--max_protein_dist` | `5.0` | `--no_filter_by_distance` | Remove waters far from protein |
| `--min_edia` | `0.4` | `--no_filter_by_edia` | Remove waters with low EDIA scores |
| `--max_bfactor_zscore` | `2.0` | `--no_filter_by_bfactor` | Remove waters with high B-factor |

<details>
<summary><strong>About EDIA Scores</strong></summary>

EDIA measures how well an atom's position is supported by the experimental electron density map. Higher EDIA scores indicate more reliable atomic positions.

**Configuration:**
- EDIA filtering is enabled by default 
- The EDIA data lives in the `json` file of the format `<pdb_id>_final.json` in the same directory as the structure file, and is obtained from PDB-REDO.
- Use `--no_filter_by_edia` to explicitly disable EDIA filtering

</details>

## Inference

Run inference on a trained model:

```bash
uv run python -m scripts.inference \
    --run_dir /path/to/training_run \
    --pdb_list splits/test_list.txt \
    --output_dir ./outputs \
    --method rk4 \
    --num_steps 100
```

### Key Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_dir` | required | Path to training run directory (contains config.json) |
| `--pdb_list` | required | Text file with PDB entries (one per line) |
| `--output_dir` | required | Directory for output plots, GIFs, and metrics |
| `--method` | `rk4` | Integration method: `euler` (fast) or `rk4` (accurate) |
| `--num_steps` | `100` | Number of integration steps |
| `--checkpoint` | `best.pt` | Checkpoint filename to load |
| `--batch_size` | `8` | Number of proteins to process in parallel |
| `--save_gifs` | `false` | Save trajectory GIFs (slower) |
| `--threshold` | `1.0` | Distance threshold for precision/recall (A) |
| `--water_ratio` | `None` | Sample `num_residues * ratio` waters (if not set, uses ground truth count) |
| `--use_sc` | `false` | Use self-conditioning during integration |

> **`--water_ratio` counts mate residues too.** `num_residues` covers ASU *and* mate
> residues, so `--include_mates` emits ~1.7x more waters at the same ratio (~440 vs
> ~263 particles at ratio 1, against ~238 true waters). Two runs share a sampling
> budget only if their mate settings match; compare density-sensitive metrics at
> parity, not at equal ratio. `--include_mates` is inherited from the training config
> when the flag is absent.

### Output Structure

```
<output_dir>/<run_name>/
├── plots/              # 3D visualization PNGs for each PDB
│   ├── 1abc_final.png
│   └── ...
├── gifs/               # Trajectory GIFs (if --save_gifs)
│   ├── 1abc_final.gif
│   └── ...
└── metrics.json        # Per-sample and summary statistics
```
