# WaterFlow: Prediction of Ordered Water Molecule Positions on Protein Structures

<p align="center">
  <img src="figures/graphical_abstract.png" alt="WaterFlow pipeline: prior → flow → candidates → clustering → kept waters" width="900">
</p>

WaterFlow is a two-stage Deep Learning model that predicts the positions of ordered water molecules conditioned on a protein structure. It has two trained components: a **candidate generator** trained with flow-matching, and a **confidence model** that ranks and scores each candidate to obtain final water coordinates.

- **[Predicting waters](#predicting-waters)** — running inference on the shipped models on your structure/s.
- **[Training your own models](#training-your-own-models)** — reproduce both models from your own training sets or our own splits. 

## Table of Contents

- [Installation](#installation)
  - [System libraries](#system-libraries)
  - [Building an environment from scratch](#building-an-environment-from-scratch)
- [Predicting waters](#predicting-waters)
  - [Step 1 — Fetch the model weights](#step-1--fetch-the-model-weights)
  - [Step 2 — Pick a checkpoint model set](#step-2--pick-a-checkpoint-model-set)
  - [Step 3 — Generate ESM embeddings](#step-3--generate-esm-embeddings)
  - [Step 4 — Predict](#step-4--predict)
  - [Predicting on many structures](#predicting-on-many-structures)
  - [Reusing work between runs](#reusing-work-between-runs)
  - [Selecting the final waters](#selecting-the-final-waters)
  - [Outputs](#outputs)
  - [All prediction options](#all-prediction-options)
  - [Troubleshooting](#troubleshooting)
- [Training your own models](#training-your-own-models)
  - [Before you train](#before-you-train)
  - [1. Precompute embeddings](#1-precompute-embeddings)
  - [2. Train the flow generator](#2-train-the-flow-generator)
  - [3. Cache candidate waters](#3-cache-candidate-waters)
  - [4. Train the confidence scorer](#4-train-the-confidence-scorer)
  - [Reproducing the released checkpoints](#reproducing-the-released-checkpoints)
  - [Evaluating the flow generator alone](#evaluating-the-flow-generator-alone)
- [Documentation](#documentation)

## Installation

WaterFlow uses [`uv`](https://docs.astral.sh/uv/) with Python 3.12. From the repository root:

```bash
uv sync
```

Every command below runs through `uv run`.

### System libraries

`uv sync` installs the Python dependencies but **not** the OpenGL system library that PyMOL
loads at import time. On a bare machine or container, pymol commands can fail with:

```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

The traceback points at `import pymol2` in `src/dataset.py`, which every script imports. Fix it
by installing the library (Debian/Ubuntu):

```bash
sudo apt-get install -y libgl1
```

If PyMOL still fails to import, add `libxrender1 libxext6`. 

### Building an environment from scratch

<details>
<summary>Only if you cannot use <code>uv sync</code></summary>

`pyproject.toml` is the single source of truth for dependencies, including the pinned CUDA
12.6 wheel indexes for `torch`, `torch-scatter`, `torch-cluster` and `pyg-lib`.

If you must build an environment manually:

```bash
uv venv water --python 3.12
source water/bin/activate
uv pip install -r pyproject.toml          # resolves the same pins as uv sync
```
To target a different CUDA build, change the index URLs under `[[tool.uv.index]]` in
`pyproject.toml` rather than installing wheels one by one.

</details>

## Predicting waters

`scripts/predict_waters.py` is the end-to-end prediction tool script. Given a raw PDB or mmCIF
structure it strips any existing waters, builds the graph from protein + het-atoms, samples
candidates with the flow model, scores them with the confidence model, selects the final set,
and writes the input structure back out with the predicted waters added.

The four steps below cover the pipeline end to end. Fetch the weights, pick a checkpoint model set, generate
embeddings, and predict waters.

### Step 1 — Fetch the model weights

The pretrained weights live in `checkpoints/` and are stored with
[Git LFS](https://git-lfs.com). A plain `git clone` succeeds without LFS, but each `.pt`
arrives as a few-hundred-byte text pointer instead of the ~42 MB model, and loading it fails.

Install Git LFS if you do not have it:

```bash
# Option A — conda / mamba:
conda install -c conda-forge git-lfs

# Option B — user-local binary, from https://github.com/git-lfs/git-lfs/releases:
VERSION=3.5.1   # set to the latest release
curl -L https://github.com/git-lfs/git-lfs/releases/download/v${VERSION}/git-lfs-linux-amd64-v${VERSION}.tar.gz \
  | tar -xz -C /tmp
mkdir -p ~/.local/bin && cp /tmp/git-lfs-${VERSION}/git-lfs ~/.local/bin/   # ensure ~/.local/bin is on PATH

# Option C — system package manager (needs root):
sudo apt-get install git-lfs   # Debian/Ubuntu (macOS: brew install git-lfs)
```

Then fetch the weights:

```bash
git lfs install    # once per machine
git lfs pull       # replaces the pointers with the real files
```

### Step 2 — Pick a checkpoint model set

Two model sets ship with the repo, differing in whether symmetry mates were used as additional
context nodes during training:

| `--ckpt_dir` | Symmetry mates | Use when |
|---|---|---|
| `checkpoints/mates` (default) | yes | The input `.cif`/`.pdb` carries crystal header information |
| `checkpoints/mates_off` | no | No crystal symmetry available — predicted structures, models, stripped files |

Each directory holds the same four files: `flow.pt`, `confidence.pt`, `flow_config.json`,
`confidence_config.json`. To use your own models, point `--ckpt_dir` at a directory with those
four names — see [Training your own models](#training-your-own-models).

> **Path resolution.** `--ckpt_dir` is an ordinary path resolved against your **current working
> directory**, not the repository root. The default `checkpoints/mates` therefore only works
> when you run from the top of the repo. From anywhere else, pass an absolute path such as
> `--ckpt_dir /path/to/WaterFlow/checkpoints/mates`.

### Step 3 — Generate ESM embeddings

The shipped checkpoints encode the protein with **ESM3** ([EvolutionaryScale
ESM](https://github.com/evolutionaryscale/esm)), so this step is required. It is also the *only*
embedding step in the repo — prediction and training both **load** cached embeddings and never
generate them on the fly, so you run this same script once for whatever structures you need,
whether you are predicting or training.

```bash
uv run python -m scripts.generate_esm_embeddings \
    --struc <protein>.cif \
    --cache_dir <cache_root>
```

This writes `<cache_root>/esm/<protein>.pt`, keyed by the input's **file stem**.

> **`<protein>`** stands for your input file's stem, for e.g. `1abc` for `1abc.cif`.
> The stem of the file you pass is what
> names the cached embedding and, later, every output file, and prediction looks the embedding up
> by that same stem. So `<protein>.cif` and `<protein>.pdb` both pair with `esm/<protein>.pt`.

Embed several structures in one pass by listing them after `--struc` (a shell glob works too),
then predict them together with `--pdb_list` (see
[Predicting on many structures](#predicting-on-many-structures)):

```bash
uv run python -m scripts.generate_esm_embeddings \
    --struc a.cif b.cif c.pdb \
    --cache_dir <cache_root>
```

Pass the cache root as `--processed_dir <cache_root>` when predicting. The first run downloads the
`esm3-open` weights from HuggingFace (network access, plus a HuggingFace login if the model is
gated for your account); later runs reuse the local copy.

> For training the invocation is the same script with a split file instead of raw paths
> (`--split_file <split> --base_pdb_dir <dir>`); see [Precompute embeddings](#1-precompute-embeddings).
> The `gvp` encoder learns from coordinates and chemical identity alone — no embeddings, no
> `--processed_dir` — but none of the shipped checkpoints use it.

### Step 4 — Predict

```bash
uv run python -m scripts.predict_waters \
    --struc <protein>.cif \
    --processed_dir <cache_root> \
    --out_dir out/
```

To run without symmetry mates, add `--ckpt_dir checkpoints/mates_off`.

If an embedding is missing, prediction stops immediately and names the files it could not find,
rather than failing later inside graph construction.

What happens to your structure along the way:

- **Existing waters are removed.** They are what the model predicts, so the graph starts with
  no water nodes.
- **Protein and hets are kept** (ligands, ions, cofactors, nucleic acids). Coordinates are
  centered on the ASU protein centroid, and symmetry mates are added when the selected
  checkpoint used them.
- Predicted waters are returned to the **input coordinate frame** before writing.

### Predicting on many structures

Point `--pdb_list` at a text file of structure names, one per line, resolved under
`--base_pdb_dir`. Each entry may include or omit a `.pdb`/`.cif` extension and may name a
subdirectory:

```bash
uv run python -m scripts.predict_waters \
    --pdb_list structures.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --out_dir out/
```

Every listed structure still needs its own embedding under `<cache_root>/esm/<stem>.pt` —
generate them all first (Step 3). Structures are processed in batches of `--batch_size`
(default 4).

### Reusing work between runs

`--geometry_cache <dir>` caches the flow inputs (inference graphs) at `<dir>/<name>.pt` and the
flow outputs (sampled candidates) at `<dir>/candidates/<name>.pt`. Both are reused when present,
so a re-run skips graph construction and flow sampling for structures already cached. For a
single small protein this barely matters; it pays off across repeated runs over many structures.

Cache entries written by a mates checkpoint carry a `_mates` suffix (`<name>_mates.pt`), so mates
and `mates_off` runs can safely share one cache directory without reusing each other's graphs.

### Selecting the final waters

This is the main knob that determines how many waters your predict. Both modes sample `--water_ratio × num_residues` candidates and score
each one, then cluster them in two rounds:

1. **Absorb** — seed a cluster with the highest-confidence unassigned candidate, absorb every
   unassigned candidate within the van der Waals radius of oxygen (1.52 Å), and emit a
   confidence-weighted centroid carrying the cluster's highest confidence.
2. **Merge** — run non-maximum suppression over those centroids, dropping the lower-confidence
   member of any pair still within the same radius.

The modes differ in how the surviving centroids are culled:

| `--selection` | Rule | Use when |
|---|---|---|
| `confidence` (default) | Drop candidates below `--confidence_threshold` (default `0.5`) before clustering, then keep every centroid. | You want a calibrated score cutoff, or only high-confidence waters. |
| `density` | Cluster with no cutoff, then keep the top `floor(--density_ratio × ASU_residues)` centroids by confidence (default ratio `0.6`). | You want a hydration level tied to protein size rather than an absolute score. |

Each mode accepts only its own knob: `--confidence_threshold` is rejected under `density`, and
`--density_ratio` is rejected under `confidence`.

> **`--water_ratio` sets how many candidates are *drawn*** (× the graph's residue count, which
> mates roughly double), **while `--density_ratio` sets how many are *kept* after scoring candidates** (× ASU residues
> only, so mates don't change it).

### Outputs

Per structure, in the input coordinate frame:

- `<protein>_pred.pdb` (or `.cif` with `--out_format .cif`) — the input protein and hets with
  predicted waters added as `HOH` oxygens.
- `<protein>_waters.txt` — one `x y z confidence` row per predicted water.

### All prediction options

| Argument | Default | Description |
|---|---|---|
| `--ckpt_dir` | `checkpoints/mates` | Directory with `flow.pt`, `confidence.pt`, `flow_config.json`, `confidence_config.json`; resolved against your working directory |
| `--struc` / `--pdb_list` | one required | A single structure file, or a list of names under `--base_pdb_dir` |
| `--base_pdb_dir` | none | Directory that `--pdb_list` names resolve against |
| `--out_dir` | required | Output directory |
| `--out_format` | `.pdb` | Written structure format: `.pdb` or `.cif` |
| `--selection` | `confidence` | `confidence` or `density` (see above) |
| `--confidence_threshold` | `0.5` | `confidence` mode only: drop candidates scoring below this |
| `--density_ratio` | `0.6` | `density` mode only: keep `floor(ratio × ASU residues)` waters |
| `--water_ratio` | `8.0` | Candidates sampled = ratio × residues in the graph (mates included when on) |
| `--num_steps` | `20` | Flow integration steps |
| `--method` | `euler` | Integration method: `euler` or `rk4` |
| `--include_mates` | model's setting | Force symmetry mates on or off (`--no-include_mates` to disable) |
| `--processed_dir` | none | Embedding cache root for `esm`/`slae` (unused for `gvp`) |
| `--geometry_cache` | none | Cache inference graphs and candidates for reuse |
| `--batch_size` | `4` | Structures per batch |
| `--device` | `cuda` | Compute device |
| `--log_level` | `INFO` | Logging verbosity |

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: libGL.so.1: cannot open shared object file` | PyMOL's OpenGL library is missing. `sudo apt-get install -y libgl1` (see [System libraries](#system-libraries)). |
| Checkpoint fails to load, `.pt` is only a few hundred bytes | Git LFS pointers were never resolved. Run `git lfs install && git lfs pull`. |
| `Missing esm embeddings under ... for [...]` | Generate them for those files first with `generate_esm_embeddings --struc <files> --cache_dir <cache_root>`, and make sure `--processed_dir` points at that same cache root. |
| `flow_config.json` not found | `--ckpt_dir` was resolved against the wrong directory. Run from the repository root or pass an absolute path. |
| `<name>: no waters selected` | Every candidate scored below `--confidence_threshold`. Lower it, or switch to `--selection density`. |

## Training your own models

Producing the flow and confidence checkpoints is a four-step pipeline. Unlike prediction,
training reads structures from a `--base_pdb_dir` laid out as
`<base_pdb_dir>/<pdb_id>/<pdb_id>_final.{cif,pdb}`, with split files listing one bare ID per
line (`6eey_final`). See [docs/data.md](docs/data.md) for the directory layout, cache
structure, and quality filters.

### Before you train

Two defaults will stop a first run if you are not ready for them:

**Quality filtering needs EDIA scores.** Water filtering by EDIA is on by default and expects a
per-structure `<pdb_id>_final.json` next to each structure file. If those files are absent,
every structure is rejected and training aborts before the first epoch:

```
EDIA filtering enabled but JSON file missing for <pdb_id>. Expected file: <pdb_id>_final.json
...
Dataset contains 0 valid entries
```

**What the file is.** EDIAm measures how well an
atom's modelled position is supported by the experimental electron density — higher is better
supported. WaterFlow reads one number per water: the `EDIAm` field of each `HOH`/`WAT` record,
matched to the structure by chain, residue number and insertion code. Waters scoring below
`--min_edia` (default `0.4`) are dropped as unreliably placed. The file is a flat JSON array of
per-residue records:

```json
[{"EDIAm": 0.609, "RSCCS": 0.907, "RSR": 0.116, "compID": "HIS",
  "pdb": {"strandID": "A", "seqNum": -1, "insCode": ""}, "seqID": 1}, ...]
```

**Where to get it.** [PDB-REDO](https://pdb-redo.eu/) entries ship with this data, so if your
structures came from PDB-REDO you may already have it. Otherwise compute it with
[**density-fitness**](https://github.com/PDB-REDO/density-fitness), the PDB-REDO tool that
produces exactly these fields. It needs the model *and* its structure factors (an MTZ or
reflection file) — EDIA cannot be derived from coordinates alone:

```bash
density-fitness <structure>.cif <structure>.mtz -o <pdb_id>_final.json
```

Check `density-fitness --help` for your build's exact flag names, and write the output beside
the structure as `<pdb_id>_final.json`.

**If you have no reflection data**, disable that one filter — the distance and B-factor filters
need nothing extra and stay on:

```bash
--no_filter_by_edia
```

Training then runs on all waters that survive the geometric filters. This is the right choice
for predicted or non-crystallographic inputs, where electron density does not exist.
(`--no_filter_by_distance` and `--no_filter_by_bfactor` disable the other two.)

**Logging is opt-in.** By default Weights & Biases runs in *disabled* mode: nothing is logged or
uploaded and no W&B account or login is needed. Pass `--wandb_project <name>` to log the run
online to that project instead, which does require a login. See
[docs/training.md](docs/training.md#weights--biases) for the disabled-vs-online distinction and
how to authenticate.

### 1. Precompute embeddings

Training loads ESM3 embeddings exactly as prediction does ([Step 3 — Generate ESM
embeddings](#step-3--generate-esm-embeddings)). The only difference is that a training run keys
them by split entry, so point the script at the split file rather than raw paths:

```bash
uv run python -m scripts.generate_esm_embeddings \
    --split_file splits/water_pdbs.txt \
    --base_pdb_dir <pdb_dir> \
    --cache_dir <cache_root>
```

This writes `<cache_root>/esm/<pdb_id>_final.pt` (e.g. `6eey_final` → `esm/6eey_final.pt`).

> **SLAE is legacy.** The `slae` encoder is kept for older runs. It depends on the external
> `SLAE` package (not a WaterFlow dependency) and an autoencoder checkpoint — see
> `scripts/generate_slae_embeddings.py`. New runs use ESM.

### 2. Train the flow generator

The geometry cache is built automatically from `--base_pdb_dir` on the first run and reused
afterward.

```bash
uv run python -m scripts.train \
    --train_list splits/train_list_0.95.txt \
    --val_list splits/valid_list_0.05.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --encoder_type esm \
    --batch_size 1 \
    --grad_accum_steps 4
```

> `--encoder_type` defaults to `esm`, matching the shipped models, and requires the embeddings
> from step 1 under `--processed_dir`. Pass `--encoder_type gvp` to train from coordinates
> alone with no embeddings (and no `--processed_dir`).

Checkpoints, `config.json`, and logs land in `<save_dir>/<run_name>/`. That run directory is
the `--flow_run_dir` for later stages.

**Multi-GPU:** launch the same command with `torchrun` — no code flag needed. Plain
`python -m scripts.train` stays single-GPU.

```bash
uv run torchrun --nproc_per_node=4 -m scripts.train \
    --train_list splits/train_list_0.95.txt \
    --val_list splits/valid_list_0.05.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --encoder_type esm \
    --batch_size 4        # per rank -> effective 16
```

Full argument reference, DDP mechanics, and W&B logging: [docs/training.md](docs/training.md).

### 3. Cache candidate waters

The confidence model trains on waters sampled from a trained flow checkpoint. Generate that
candidate cache once:

```bash
uv run python -m scripts.cache_candidates \
    --flow_run_dir <flow_run> \
    --pdb_list splits/conf_pdbs.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root>
```

Candidates are written to
`<cache_root>/candidate_cache/<run>_<ckpt>_<method><steps>_r<ratio>_s<seed>/`, one
`<pdb_id>.pt` per structure. That directory is the `--candidate_dir` for the next step.

### 4. Train the confidence scorer

```bash
uv run python -m scripts.train_confidence \
    --flow_run_dir <flow_run> \
    --train_list splits/conf_train.txt \
    --val_list splits/conf_valid.txt \
    --candidate_dir <candidate_dir> \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --save_dir <out> \
    --run_name <run_name> \
    --init_from <flow_run>/checkpoints/best.pt \
    --freeze_backbone
```

`--init_from` warm-starts the shared backbone from the flow checkpoint; `--freeze_backbone`
then trains only the score head. Validation reports AUC-PR (used for checkpoint selection) and
best F1. Multi-GPU works the same way as flow training — prefix with `torchrun --nproc_per_node=N`.

To use your own models for prediction, collect them into one directory under the four names
`predict_waters.py` expects:

```bash
mkdir -p my_ckpts
cp <flow_run>/checkpoints/best.pt        my_ckpts/flow.pt
cp <flow_run>/config.json                my_ckpts/flow_config.json
cp <conf_run>/checkpoints/best.pt        my_ckpts/confidence.pt
cp <conf_run>/config.json                my_ckpts/confidence_config.json
```

Then predict with `--ckpt_dir my_ckpts`.

### Reproducing the released checkpoints

The commands below are the recipe recorded in the shipped
`checkpoints/*/flow_config.json` and `confidence_config.json`, reduced to the flags that differ
from current defaults. They reproduce every recorded setting.

**Flow generator — `checkpoints/mates`:**

```bash
uv run python -m scripts.train \
    --train_list <train.txt> --val_list <val.txt> \
    --base_pdb_dir <pdb_dir> --processed_dir <cache_root> \
    --save_dir <out> --run_name <name> \
    --include_mates \
    --disable_ww --disable_wp \
    --epochs 110 \
    --batch_size 1 --grad_accum_steps 2 \
    --lr 0.002 --weight_decay 0.03 \
    --warmup_steps 300 --lr_decay_epochs 100 \
    --fused_adamw \
    --eval_every 2 --save_every 2 --n_eval_samples 115 \
    --seed -1
```

**Flow generator — `checkpoints/mates_off`:** the same command with three changes — drop
`--include_mates`, use `--lr 0.004`, and drop `--batch_size 1 --grad_accum_steps 2` (that run
used the defaults, batch 4 with no accumulation).

**Candidate cache** (feeds the confidence model; the released run used ratio 8, seed 0):

```bash
uv run python -m scripts.cache_candidates \
    --flow_run_dir <flow_run> \
    --pdb_list <conf_pdbs.txt> \
    --base_pdb_dir <pdb_dir> --processed_dir <cache_root> \
    --water_ratio 8 --seed 0
```

**Confidence scorer** (identical for both checkpoint sets — note **no** `--freeze_backbone`):

```bash
uv run python -m scripts.train_confidence \
    --flow_run_dir <flow_run> \
    --train_list <conf_train.txt> --val_list <conf_valid.txt> \
    --candidate_dir <candidate_dir> \
    --base_pdb_dir <pdb_dir> --processed_dir <cache_root> \
    --save_dir <out> --run_name <name> \
    --init_from <flow_run>/checkpoints/best.pt \
    --epochs 25 \
    --batch_size 1 --grad_accum_steps 2 \
    --lr 1e-4 --weight_decay 0.01 \
    --warmup_steps 300 \
    --max_candidates 1500 \
    --strict_cache \
    --num_workers 8
```

Everything else — `--encoder_type esm`, `--scheduler cosine`, `--r_in 0.5`, `--r_out 1.5`,
`--accept_radius 1.0`, `--grad_clip 1.0`, AMP on in bfloat16, all three water filters on — is
already the default.

The remaining flags in those configs are dataloader performance settings that do not change the
model: `--num_workers 12 --pin_memory --persistent_workers --cache_load_mmap` for flow.

<details>
<summary><strong>Keys the configs record that do not exist in current train scripts</strong></summary>

The shipped `*_config.json` files were written by an earlier version of the training code that
had features this repository does not, so they record some keys with no matching CLI flag.

**These leftover keys do not affect model loading or inference.** Loading reads only the
architecture and graph-construction keys — encoder type, hidden dimensions, layer counts,
cutoff, neighbour limits, edge ablations — and ignores everything else. Each unmatched key was
either switched off in the recorded run or set to the behaviour the current code already
implements:

| Keys | Recorded value | Effect |
|---|---|---|
| `loss_weighting`, `loss_eps` | `uniform`, `null` | Matches: the trainer uses a plain unweighted MSE over the velocity field. No effect on inference. |
| `min_snr_gamma`, `t_logit_mean/std` | `5.0`, `0.0`/`1.0` | Inert under `loss_weighting: uniform`; neither exists in this repo. |
| `t_dist` | `uniform` | Matches: `training_step` draws `t` from `torch.rand` (`src/flow.py:1147`). |
| `plateau_*` | — | Only read when `scheduler` is `plateau`; both runs used `cosine`. |
| `early_stopping_*`, `max_train_steps`, `max_val_steps`, `benchmark_*`, `profile_*` | `None` / `false` | All disabled. |
| `use_bce`, `use_mse` (confidence) | `true`, `false` | This repo's confidence trainer already uses BCE-with-logits on the smootherstep target. |
| `coverage_weight`, `target_recall` (confidence) | `0.0`, `0.45` | The coverage term is weighted zero, making `target_recall` inert. |
| `amp_dtype` (confidence) | `bfloat16` | The trainer hardcodes bfloat16 autocast. |
| `active_water_filters`, `ignored_water_filter_thresholds` | all three filters on, `[]` | A record of the filters used, matching this repo's defaults. |
| `resume_extend_lr`, `skip_wandb`, `wandb_log_interval` | `false`, `false`, `1` | Resume and logging bookkeeping; no effect on the model. |

The model **architecture** — encoder type, hidden dimensions, layer counts — is what must match
for a checkpoint to load, and a run following the commands above matches it.

</details>

### Evaluating the flow generator alone

`scripts/inference.py` scores the flow generator against ground-truth waters, separately from
the confidence model. It reads cached training-format graphs (not raw files) and reports
precision, recall, and RMSD.

```bash
uv run python -m scripts.inference \
    --run_dir <flow_run> \
    --pdb_list splits/test_list.txt \
    --base_pdb_dir <pdb_dir> \
    --processed_dir <cache_root> \
    --output_dir ./eval \
    --method rk4 \
    --num_steps 100
```

`--threshold` (default `1.0` Å) sets the distance for precision/recall matching. Use
`--water_ratio` to sample a fixed count instead of the ground-truth number; metrics that need
ground truth are skipped automatically when it is set.

## Documentation

- [docs/data.md](docs/data.md) — input layout, split files, EDIA, the geometry / ESM / SLAE
  cache structure, and quality filters.
- [docs/model.md](docs/model.md) — the two-stage architecture, encoder types, and edge
  construction.
- [docs/training.md](docs/training.md) — full argument reference for the flow and confidence
  trainers, DDP, checkpoints, and W&B.

<p align="center">
  <img src="figures/inference_sweep.gif" alt="Flow ODE integration sweeping candidate waters from the prior to final kept waters" width="900">
</p>
