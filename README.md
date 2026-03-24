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
│   ├── train.py            # Training pipeline (Hydra-based)
│   ├── inference.py        # Run inference on trained models (Hydra-based)
│   ├── generate_esm_embeddings.py   # Precompute ESM embeddings
│   └── generate_slae_embeddings.py  # Precompute SLAE embeddings
├── configs/                # Hydra configuration files
│   ├── train.yaml          # Main training config
│   ├── inference.yaml      # Main inference config
│   ├── data/default.yaml   # Data paths and quality filters
│   ├── model/              # Encoder configs (gvp, esm, slae)
│   ├── flow/default.yaml   # Flow matching parameters
│   └── logging/default.yaml # Checkpointing and W&B settings
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

### Input PDB Files

WaterFlow expects PDB files in a specific directory structure:

```
<base_pdb_dir>/
├── 1abc/
│   └── 1abc_final.pdb
├── 2xyz/
│   └── 2xyz_final.pdb
└── ...
```

Each PDB should have `_final` suffix and contain:
- Protein atoms (used as conditioning context)
- Water molecules (HOH residues, used as ground truth)

### Data Processing Pipeline

WaterFlow processes PDB files through several stages to create training-ready graph representations:

**PDB Parsing**
- Uses Biotite to extract protein atoms and water molecules (HOH residues)
- Modified residues are retained during structure parsing and geometry preprocessing
- When generating ESM embeddings, modified residues are mapped to encoder-compatible amino acid identities (e.g., MSE→M/MET, SEC→U/SEC)
- Hydrogen atoms are excluded
- Only the first model is used
- For atoms with alternate conformations, the highest-occupancy conformer is selected

**Crystal Contact Detection**
- Uses PyMOL's `symexp` to generate symmetry mates within 5.0Å cutoff
- Symmetry mate atoms are included as additional protein context when `include_mates=True`
- Mate atoms are stored separately for proper handling during training

**Graph Representation**
- Node types: `protein` (ASU + symmetry mates), `water` (ground truth)
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
├── geometry/              # Graph structures (or geometry_mates/ when include_mates=True)
│   └── <pdb_id>_final.pt
│       - protein_pos: centered protein coordinates (N, 3)
│       - protein_feat: element one-hot encoding (N, 16)
│       - protein_res_idx: residue indices for grouping
│       - water_pos, water_feat: water coordinates and features
│       - num_asu_protein: ASU atom count (mate boundary metadata)
│       # Note: When include_mates=True, mate atoms are concatenated into
│       # protein_pos/protein_feat. Recover boundaries via:
│       #   ASU atoms = protein_pos[:num_asu_protein]
│       #   Mate atoms = protein_pos[num_asu_protein:]
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

**Cache Generation Notes:**
- Geometry cache is generated automatically when `preprocess=True` (default)
- ESM/SLAE caches require running the respective `generate_*_embeddings.py` scripts first
- Preprocessing failures are logged to `<geometry_dir>/preprocessing_failures.log`

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

WaterFlow uses [Hydra](https://hydra.cc/) for configuration. Configs are structured under `configs/` with modular defaults.

### GVP Encoder (no precomputed embeddings required)

```bash
uv run python -m scripts.train \
    train_list=splits/train_list_0.95.txt \
    val_list=splits/valid_list_0.05.txt
```

### ESM Encoder (requires precomputed ESM embeddings)

```bash
uv run python -m scripts.train \
    train_list=splits/train_list_0.95.txt \
    val_list=splits/valid_list_0.05.txt \
    model=esm \
    training.batch_size=1 \
    training.optimizer.grad_accum_steps=4 \
    data.processed_dir=~/flow_cache/
```

### Resuming from Checkpoints

To resume training from a checkpoint, you can load the model weights and optimizer state:

```bash
# Checkpoints are saved in <save_dir>/<run_name>/checkpoints/
# - best.pt: Best validation loss
# - epoch_N.pt: Periodic checkpoints every logging.save_every epochs
```

### Configuration Structure

Training configuration is organized into modular YAML files:

| Config | Path | Description |
|--------|------|-------------|
| Main | `configs/train.yaml` | Training entry point with defaults |
| Data | `configs/data/default.yaml` | Data paths and quality filters |
| Model | `configs/model/{gvp,esm,slae}.yaml` | Encoder and flow architecture |
| Flow | `configs/flow/default.yaml` | Flow matching parameters |
| Logging | `configs/logging/default.yaml` | Checkpointing and W&B settings |

### Key Training Options

| Config Key | Default | Description |
|------------|---------|-------------|
| `train_list` | required | Path to training split file |
| `val_list` | required | Path to validation split file |
| `model` | `gvp` | Model config: `gvp`, `esm`, or `slae` |
| `training.batch_size` | `4` | Batch size (use smaller for ESM) |
| `training.optimizer.grad_accum_steps` | `1` | Gradient accumulation steps |
| `training.epochs` | `200` | Number of training epochs |
| `training.optimizer.lr` | `1e-3` | Learning rate |
| `training.scheduler.type` | `cosine` | LR scheduler: `cosine`, `step`, `none` |
| `data.processed_dir` | null | Cache directory for preprocessed data |
| `logging.save_dir` | null | Directory to save checkpoints |
| `logging.save_every` | `10` | Save checkpoint every N epochs |
| `logging.eval_every` | `5` | Run evaluation every N epochs |

### Weights & Biases Logging

Training automatically logs to W&B. Configure with:

| Config Key | Default | Description |
|------------|---------|-------------|
| `logging.wandb.project` | `water-flow` | W&B project name |
| `logging.wandb.dir` | null | Local W&B log directory |
| `logging.run_name` | auto-generated | Custom run name |

## Quality Filtering

WaterFlow applies multiple quality filters to ensure high-quality training data. These are configured in `configs/data/default.yaml`.

### Structure-Level Quality Checks

These checks determine whether a structure is included in training:

| Config Key | Default | Description |
|------------|---------|-------------|
| `data.quality.max_com_dist` | `25.0` | Max protein-water center-of-mass distance (A) |
| `data.quality.max_clash_fraction` | `0.05` | Max fraction of waters clashing with protein |
| `data.quality.clash_dist` | `2.0` | Distance threshold for clash detection (A) |
| `data.quality.min_water_residue_ratio` | `0.6` | Minimum waters per residue ratio |

### Per-Water Quality Filters

These filters remove individual low-quality waters (toggleable via config):

| Config Key | Default | Description |
|------------|---------|-------------|
| `data.water_filter.max_protein_dist` | `5.0` | Remove waters far from protein |
| `data.water_filter.filter_by_distance` | `true` | Toggle distance filtering |
| `data.water_filter.min_edia` | `0.4` | Remove waters with low EDIA scores |
| `data.water_filter.filter_by_edia` | `true` | Toggle EDIA filtering |
| `data.water_filter.max_bfactor_zscore` | `1.5` | Remove waters with high B-factor |
| `data.water_filter.filter_by_bfactor` | `true` | Toggle B-factor filtering |

<details>
<summary><strong>About EDIA Scores</strong></summary>

EDIA measures how well an atom's position is supported by the experimental electron density map. Higher EDIA scores indicate more reliable atomic positions.

**Configuration:**
- EDIA filtering is enabled by default but only activates if `data.edia_dir` is provided
- If `data.edia_dir` is not set, EDIA filtering is skipped (with a warning logged)
- Set `data.water_filter.filter_by_edia=false` to explicitly disable EDIA filtering

**Directory structure:** `{edia_dir}/{pdb_id}/{pdb_id}_residue_stats.csv`

</details>

## Inference

Run inference on a trained model:

```bash
uv run python -m scripts.inference \
    run_dir=/path/to/training_run \
    pdb_list=splits/test_list.txt \
    output_dir=./outputs
```

With custom integration settings:

```bash
uv run python -m scripts.inference \
    run_dir=/path/to/training_run \
    pdb_list=splits/test_list.txt \
    output_dir=./outputs \
    inference.integration.method=euler \
    inference.integration.num_steps=50
```

### Key Inference Options

| Config Key | Default | Description |
|------------|---------|-------------|
| `run_dir` | required | Path to training run directory (contains config.json) |
| `pdb_list` | required | Text file with PDB entries (one per line) |
| `output_dir` | required | Directory for output plots, GIFs, and metrics |
| `inference.integration.method` | `rk4` | Integration method: `euler` (fast) or `rk4` (accurate) |
| `inference.integration.num_steps` | `100` | Number of integration steps |
| `inference.checkpoint` | `best.pt` | Checkpoint filename to load |
| `inference.hardware.batch_size` | `8` | Number of proteins to process in parallel |
| `inference.visualization.save_gifs` | `false` | Save trajectory GIFs (slower) |
| `inference.evaluation.threshold` | `1.0` | Distance threshold for precision/recall (A) |
| `inference.water.water_ratio` | null | Sample `num_residues * ratio` waters (if not set, uses ground truth count) |
| `inference.integration.use_sc` | `false` | Use self-conditioning during integration |

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
