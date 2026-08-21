# Model

WaterFlow is a two-stage model: a flow generator that proposes candidate waters
and a confidence scorer that ranks them.

## Graph representation

Each structure becomes a heterogeneous graph with two node types:

- `protein` — ASU atoms, plus symmetry mates and ligands when enabled. Ligand
  atoms carry `is_ligand` and `residue_index = -1` (no residue embedding, so
  residue pooling masks them out). `is_mate` marks every non-ASU node; the flow
  prior anchors on `~is_mate` so sampled waters start where the targets live.
- `water` — the molecules being predicted.

Edge types (defined in `src/constants.py`):

| Edge | Direction | Built |
|---|---|---|
| `('protein', 'pp', 'protein')` | protein–protein | cached at preprocessing |
| `('protein', 'pw', 'water')` | protein → water | runtime |
| `('water', 'wp', 'protein')` | water → protein | runtime, ablatable |
| `('water', 'ww', 'water')` | water–water | runtime, ablatable |

Only PP edges are cached; every water-touching edge is rebuilt each forward pass,
because water positions move during integration. Default edge cutoff is 8.0 Å
(`RBF_CUTOFF`).

## Flow generator

A GVP network predicts the velocity field for water trajectories, conditioned on
per-residue protein embeddings. Sampling integrates that field from a prior
(`--sampling_strategy`, either `uniform_ball` or `scaled_gaussian`) to final water
positions, using Euler or RK4 steps.

### Encoder types

| Encoder | Description | Precomputation |
|---|---|---|
| `esm` (default) | ESM3 language-model embeddings, per residue | Yes — `generate_esm_embeddings.py` |
| `gvp` | Geometric Vector Perceptron over 3D coordinates | No |
| `slae` | SLAE per-atom embeddings ([preprint](https://www.biorxiv.org/content/10.1101/2025.10.03.680398v1)) — **legacy** | Yes — external `SLAE` package + checkpoint |

ESM is the encoder used for current runs. SLAE is retained for older checkpoints;
it depends on the external `SLAE` package, which is not a WaterFlow dependency.

### Edge construction

How water-touching edges (PW, WW, WP) are built is fixed at model construction, so
training and inference always agree:

| `--dynamic_edge_policy` | Behavior |
|---|---|
| `auto` (default) | Resolves off the prior: `radius` under `uniform_ball`, `knn_if_isolated` under `scaled_gaussian` |
| `radius` | Connect every pair within `--cutoff`, capped at `--max_neighbors` per source |
| `knn` | Connect a fixed number of nearest neighbors (`--k_pw`, `--k_ww`, `--k_wp`) |
| `knn_if_isolated` | A `radius` graph plus a KNN rescue for any node the cutoff stranded |

`radius` and `knn` differ in which side the neighbor budget applies to. KNN queries
*per destination*, so every destination gets edges but a source may have none —
coverage checks must read the destination row. `radius` guarantees nothing: a water
with no protein atom inside `--cutoff` gets no PW edges.

`knn_if_isolated` repairs that — any water the radius query stranded is reconnected
to its `--knn_fallback_k` nearest protein atoms regardless of distance (`0` disables
the rescue). `auto` selects it under `scaled_gaussian` precisely because Gaussian
samples can land outside every cutoff, whereas uniform-ball samples cannot.

Set `--disable_ww` / `--disable_wp` to ablate those edge types; PW and PP are
always active.

## Confidence scorer

`ConfidenceGVP` scores each candidate water in `[0, 1]`. It reuses the flow
generator's backbone with the time and self-conditioning inputs removed, and a
single scalar head per candidate. Because the backbone stays structurally
identical, the scorer warm-starts from a flow checkpoint (`--init_from`). It uses
PW and PP edges only.

- **Target** — `smootherstep_target`: a soft cutoff on the nearest ground-truth
  distance, 1 inside `r_in` (default 0.5 Å) and 0 outside `r_out` (default 1.5 Å).
  `--hard_label` switches to a binary `1[d <= accept_radius]` target, giving a
  calibrated P(within radius).
- **Clustering** — `cluster_waters_vdw`: candidates are absorbed into
  confidence-weighted centroids at 1.52 Å (oxygen van der Waals radius), followed
  by non-maximum suppression. Each centroid carries its cluster's maximum
  confidence.

The scorer is trained on candidates sampled from a flow checkpoint; see
[training.md](training.md) and the pipeline in the [README](../README.md).
