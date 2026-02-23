# WaterFlow
Predicting water molecule placements on protein surfaces using flow matching conditioned on learned protein structure embeddings.

This repo contains all the code required to process a dataset, load in features from a pre-trained model for protein structure embeddings, and run inference (and training) of the flow matching classes.

# Environment Setup

We use `uv` for our environment and package management, with python 3.12.

You can install the environment by running `uv sync` and running the scripts with `uv run python <script>`. Or if you want to install a fresh virtual environment from scratch, follow the steps below.

Installing the environment:

```
uv venv water --python 3.12
source water/bin/activate

uv pip install torch==2.8.0
uv pip install torch_geometric
uv pip install torch_cluster torch_scatter pyg_lib -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
uv pip install biotite wandb Bio networkx e3nn pytest pytest-cov
```

If you have trouble installing torch_cluster or scatter, I would suggest changing the cuda version in the wheel.

# Embedding Generation

For `esm` and `slae` encoder types, you must precompute embeddings before training or inference.

### ESM Embeddings (for `--encoder_type esm`)

```bash
uv run python -m scripts.generate_esm_embeddings \
    --split_file splits/train_list_0.95.txt \
    --cache_dir ~/flow_cache/ \
    --device cuda:0
```

### SLAE Embeddings (for `--encoder_type slae`)

```bash
uv run python -m scripts.generate_slae_embeddings \
    --split_file splits/train_list_0.95.txt \
    --cache_dir ~/flow_cache/ \
    --slae_ckpt /path/to/SLAE/checkpoints/autoencoder.ckpt
```

# Training

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

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder_type` | `gvp` | Encoder type: `gvp`, `esm`, or `slae` |
| `--batch_size` | `4` | Batch size (use smaller for ESM) |
| `--grad_accum_steps` | `1` | Gradient accumulation steps |
| `--flow_layers` | `3` | Number of flow GVP layers |
| `--hidden_s` | `256` | Scalar hidden dimension |
| `--hidden_v` | `64` | Vector hidden dimension |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--processed_dir` | `~/flow_cache/` | Cache directory for preprocessed data |

# Inference

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
| `--run_dir` | required | Path to training run directory |
| `--pdb_list` | required | Text file with PDB entries |
| `--output_dir` | required | Directory for outputs |
| `--method` | `rk4` | Integration method: `euler` or `rk4` |
| `--num_steps` | `100` | Number of integration steps |
| `--checkpoint` | `best.pt` | Checkpoint filename |
| `--save_gifs` | `false` | Save trajectory GIFs |
| `--threshold` | `1.0` | Distance threshold for metrics (Angstroms) |
| `--water_ratio` | `None` | Sample `num_residues * ratio` waters (if not set, uses ground truth count) |
