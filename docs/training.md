# Training reference

Argument reference for the flow trainer (`scripts/train.py`) and the confidence
trainer (`scripts/train_confidence.py`). For the end-to-end walkthrough, see the
[README](../README.md); for data preparation and quality filters, see
[data.md](data.md).

In the **flow** trainer, `--processed_dir` and `--base_pdb_dir` are required; `--save_dir`
defaults to `flow_checkpoints` in the working directory. The confidence trainer requires
all three.

## Flow trainer

```bash
uv run python -m scripts.train \
    --train_list splits/train_list_0.95.txt \
    --val_list splits/valid_list_0.05.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --encoder_type esm \
    --batch_size 1 --grad_accum_steps 4
```

### Data and model

| Argument | Default | Description |
|---|---|---|
| `--train_list` | required | Training split file |
| `--val_list` | required | Validation split file |
| `--base_pdb_dir` | required | Base PDB directory (used to build the geometry cache) |
| `--processed_dir` | required | Cache root (geometry + embeddings) |
| `--geometry_cache_name` | `geometry` | Base name for the geometry cache directory |
| `--encoder_type` | `esm` | `esm` and `slae` need embeddings under `--processed_dir`; `gvp` learns from coordinates alone |
| `--include_mates` | off | Include symmetry-mate atoms as protein nodes |
| `--include_ligands` | on | Include ligand/ion/cofactor/nucleic-acid heavy atoms; negate with `--no-include_ligands` |
| `--hidden_s` | `256` | Scalar hidden dimension |
| `--hidden_v` | `64` | Vector hidden dimension |
| `--flow_layers` | `3` | Number of flow GVP layers |
| `--drop_rate` | `0.1` | Dropout |

### Sampling and edges

| Argument | Default | Description |
|---|---|---|
| `--sampling_strategy` | `uniform_ball` | Flow prior: `uniform_ball` or `scaled_gaussian`; also resolves `--dynamic_edge_policy auto` |
| `--dynamic_edge_policy` | `auto` | Water-edge construction: `auto`, `radius`, `knn`, `knn_if_isolated` (see [model.md](model.md)) |
| `--cutoff` | `8.0` | Radius-edge distance cutoff (Å) for the **model's** dynamic water edges. Does not reach the dataset: cached PP edges and the crystal-contact radius always use the dataset default of 8.0 |
| `--max_neighbors` | `256` | Cap per source for radius edges, **model-side only** (same caveat as `--cutoff`) |
| `--knn_fallback_k` | `8` | Neighbors attached to waters stranded under `knn_if_isolated`; `0` disables |
| `--k_pw` / `--k_ww` / `--k_wp` | `12` / `8` / `8` | Neighbour counts for protein→water, water→water, water→protein under `knn` |
| `--disable_ww` / `--disable_wp` | off | Ablate water→water / water→protein edges |

### Optimization

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `200` | Training epochs |
| `--batch_size` | `4` | Per-rank batch size (use `1` for ESM) |
| `--grad_accum_steps` | `1` | Effective batch = n_gpus × batch_size × grad_accum_steps |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-4` | Weight decay |
| `--grad_clip` | `1.0` | Gradient-norm clip |
| `--scheduler` | `cosine` | `cosine`, `step`, or `none` |
| `--warmup_steps` | `0` | Linear warmup steps |
| `--eta_min_factor` | `0.001` | Cosine floor = lr × this |
| `--use_amp` | on | bfloat16 autocast (CUDA only); `--no-use_amp` to disable |
| `--fused_adamw` | off | Fused AdamW (CUDA only) |
| `--seed` / `--val_seed` | `42` / `1234` | Train and validation RNG seeds |

### Evaluation and checkpoints

| Argument | Default | Description |
|---|---|---|
| `--eval_every` | `5` | Run evaluation every N epochs |
| `--eval_method` | `euler` | Sampling method during eval: `euler` or `rk4` |
| `--eval_steps` | `50` | Integration steps during eval |
| `--n_eval_samples` | `3` | Number of validation structures evaluated (drawn once at start, fixed thereafter) |
| `--threshold` | `1.0` | Distance (Å) for precision/recall matching |
| `--selection_metric` | `blend` | Checkpoint-selection metric. `val_loss` is checked every epoch; `f1`, `auc_pr` and `blend` (0.85×F1 + 0.15×AUC-PR) come from the sampling eval, so they are checked on eval epochs only and averaged over the last 3. Falls back to `val_loss` when no eval epoch will run |
| `--save_dir` | `flow_checkpoints` | Parent directory for runs |
| `--run_name` | auto | Run identifier; default is `YYYYMMDD_HHMMSS_<encoder>_L<flow_layers>_h<hidden_s>` |
| `--save_every` | `10` | Save a periodic checkpoint every N epochs |
| `--resume` | off | Resume from the highest-numbered `epoch_*.pt` in the run (not `best.pt`); requires `--run_name` |

Checkpoints land in `<save_dir>/<run_name>/checkpoints/`: `best.pt` (best
selection metric) and `epoch_N.pt` (periodic). `config.json` one level up in the run directory records
the run configuration, and later stages read it via `--flow_run_dir` / `--run_dir`.

### Weights & Biases

Opt-in, as in the confidence trainer. Without `--wandb_project` the run logs
nothing and needs no account or `wandb login`; setting it logs online, with
`--wandb_dir` and `--run_name` controlling where the run is written and what it is
called.

## Multi-GPU (DDP)

Launch the same script with `torchrun`; there is no separate code flag. DDP switches
on when `WORLD_SIZE > 1`, so plain `python -m scripts.train` — and
`torchrun --nproc_per_node=1` — stay single-GPU.

```bash
uv run torchrun --nproc_per_node=4 -m scripts.train \
    --train_list splits/train_list_0.95.txt \
    --val_list splits/valid_list_0.05.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --encoder_type esm \
    --batch_size 4        # per rank -> effective 16
```

Each rank trains a disjoint shard, the loss is all-reduced, and rank 0 alone
writes checkpoints. The geometry cache is prebuilt before workers spawn, so ranks
never race to write it.

## Confidence trainer

Trains `ConfidenceGVP` on candidates sampled from a flow checkpoint (see
`scripts/cache_candidates.py` in the [README](../README.md#3-cache-candidate-waters)).
It reuses the flow run's cache layout and config.

```bash
uv run python -m scripts.train_confidence \
    --flow_run_dir <flow_run> \
    --train_list splits/conf_train.txt \
    --val_list splits/conf_valid.txt \
    --candidate_dir <candidate_dir> \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --save_dir <out> --run_name <run_name> \
    --init_from <flow_run>/checkpoints/best.pt --freeze_backbone
```

| Argument | Default | Description |
|---|---|---|
| `--flow_run_dir` | required | Flow run directory (provides encoder/filter config) |
| `--candidate_dir` | required | Candidate cache directory from `cache_candidates.py` |
| `--processed_dir` | required | Cache root shared with flow training (geometry + ESM) |
| `--base_pdb_dir` | required | Base PDB directory, as in flow training |
| `--save_dir` / `--run_name` | required | Outputs go to `<save_dir>/<run_name>/` |
| `--init_from` | none | Warm-start the shared backbone from a checkpoint |
| `--freeze_backbone` | off | Train only the score head |
| `--max_candidates` | all | Cap candidates per structure (fresh random subsample each epoch) |
| `--strict_cache` | off | Raise on a structure with no candidate file (default: skip and log) |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `8` | Per-rank batch size |
| `--lr` | `1e-4` | Learning rate |
| `--scheduler` | `cosine` | `cosine` or `plateau` (both run after linear warmup) |
| `--warmup_steps` | `500` | Linear warmup steps |
| `--grad_accum_steps` | `1` | Effective batch = n_gpus × batch_size × grad_accum_steps |
| `--weight_decay` | `1e-5` | Weight decay |
| `--eta_min_factor` | `0.01` | Cosine floor = lr × this |
| `--use_amp` | on | bfloat16 autocast (CUDA only); `--no-use_amp` to disable |
| `--fused_adamw` | off | Fused AdamW (CUDA only) |
| `--num_workers` | `4` | DataLoader workers |
| `--geometry_cache_name` / `--include_mates` | inherit | Override the flow run's cache layout; default is to reuse it |
| `--r_in` / `--r_out` | `0.5` / `1.5` | Smootherstep plateau/floor radii (Å) |
| `--accept_radius` | `1.0` | Acceptance radius (Å) for the AUC-PR label and `--hard_label` |
| `--hard_label` | off | Train on `1[d <= accept_radius]` instead of the soft target |
| `--wandb_project` | off | Set to enable W&B (opt-in, as in the flow trainer) |

Validation reports AUC-PR (used for checkpoint selection) and best F1. Multi-GPU
works exactly like flow training — prefix with `torchrun --nproc_per_node=N`. Each
rank trains a disjoint shard, the loss is all-reduced, and (score, label) pairs are
pooled across ranks so AUC-PR/F1 rank the full candidate set. Rank 0 alone writes
checkpoints.
