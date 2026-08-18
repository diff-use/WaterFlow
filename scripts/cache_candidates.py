"""
Generate the candidate cache for confidence-model training.

Samples candidate waters from a trained flow checkpoint over the flow dataset
cache layout and writes one `<pdb_id>.pt = {"candidate_pos": (Nc, 3)}` per
structure, plus a `generation.json` record of how it was made. Train the confidence
model on the result with `scripts/train_confidence.py --candidate_dir <out_dir>`.

Everything the confidence model needs (protein graph, embeddings, GT waters, PP edges) 
is loaded from the flow caches at train time. Model loading and integration reuse the flow inference
machinery verbatim, so candidates are sampled exactly as `scripts/inference.py`
would sample them.

Example:
    python -m scripts.cache_candidates \\
        --flow_run_dir <run_dir> \\
        --pdb_list splits/conf_train.txt \\
        --processed_dir <cache_root> \\
        --base_pdb_dir <pdb_dir> \\
        --water_ratio 3.0 --seed 0 --method euler --num_steps 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from scripts.inference import (
    _extract_dataset_filter_config,
    build_model_from_config,
    load_checkpoint,
    load_config,
    run_inference_batch,
)
from src.dataset import ProteinWaterDataset
from src.flow import FlowMatcher
from src.utils import setup_logging_for_tqdm


def default_candidate_dir(
    processed_dir: str | Path,
    flow_run_dir: str | Path,
    water_ratio: float,
    seed: int,
) -> Path:
    """Namespaced candidate dir so caches from different checkpoints, ratios, or
    seeds never collide: `{processed_dir}/candidate_cache/<run_name>_r{ratio}_s{seed}`.
    """
    run_name = Path(flow_run_dir).name
    return (
        Path(processed_dir) / "candidate_cache" / f"{run_name}_r{water_ratio:g}_s{seed}"
    )


def _write_candidate_cache(
    dataset,
    sample_batch,
    out: Path,
    *,
    run_info: dict,
    batch_size: int = 8,
    overwrite: bool = False,
) -> dict:
    """Sample and write one thin candidate file per uncached structure.

    Split from `generate_candidate_cache` so the loop and skip logic runs against a
    plain dataset and sampler instead of a live flow checkpoint.

    Args:
        dataset: Indexable structures. `dataset.entries[i]["cache_key"]` names the
            output file, which equals the sampled graph's `pdb_id` and the key
            `ConfidenceDataset` reads back.
        sample_batch: Callable `(graphs) -> [{"pdb_id", "water_pred"}, ...]`.
        out: Directory for the `<pdb_id>.pt` files and `generation.json`.
        run_info: Generation parameters recorded, with the counts, in
            `generation.json`.
        batch_size: Structures sampled per `sample_batch` call.
        overwrite: Re-generate even if a `<pdb_id>.pt` already exists.

    Returns:
        dict stats: {"out_dir", "n_written", "n_skipped", "n_total"}.
    """
    out.mkdir(parents=True, exist_ok=True)
    entries = dataset.entries
    n_total = len(dataset)
    n_written = 0
    n_skipped = 0
    for start in tqdm(range(0, n_total, batch_size), desc="candidate-gen"):
        chunk = range(start, min(start + batch_size, n_total))
        # Decide what to (re)generate from the cache key alone, so an already
        # cached structure never pays for a graph build.
        todo = [
            i
            for i in chunk
            if overwrite or not (out / f"{entries[i]['cache_key']}.pt").exists()
        ]
        n_skipped += len(chunk) - len(todo)
        if not todo:
            continue
        for result in sample_batch([dataset[i] for i in todo]):
            candidate_pos = torch.as_tensor(result["water_pred"], dtype=torch.float32)
            torch.save({"candidate_pos": candidate_pos}, out / f"{result['pdb_id']}.pt")
            n_written += 1

    (out / "generation.json").write_text(
        json.dumps(
            {**run_info, "n_written": n_written, "n_skipped": n_skipped, "n_total": n_total},
            indent=2,
        )
    )

    logger.info(
        f"Candidate cache written to {out}: {n_written} written, "
        f"{n_skipped} skipped (already cached)."
    )
    return {
        "out_dir": str(out),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_total": n_total,
    }


def generate_candidate_cache(
    flow_run_dir: str | Path,
    pdb_list: str | Path,
    processed_dir: str | Path,
    base_pdb_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    checkpoint: str = "best.pt",
    water_ratio: float = 3.0,
    seed: int = 0,
    num_steps: int = 100,
    method: str = "euler",
    batch_size: int = 8,
    geometry_cache_name: str | None = None,
    include_mates: bool | None = None,
    device: str = "cuda",
    overwrite: bool = False,
) -> dict:
    """
    Sample candidate waters from a trained flow checkpoint into candidate files.

    Args:
        flow_run_dir: Flow training run dir (contains config.json + checkpoints/).
        pdb_list: Text file of `<pdb_id>_final` keys, one per line.
        processed_dir: Cache root shared with flow training (geometry + esm).
        base_pdb_dir: Base PDB dir, as used by flow training.
        out_dir: Output dir. Defaults to `default_candidate_dir(...)`.
        checkpoint: Checkpoint filename under `{flow_run_dir}/checkpoints`.
        water_ratio: Oversampling ratio — sample `num_residues * water_ratio` waters.
        seed: RNG seed for the sampling prior (reproducible candidates).
        num_steps, method: Integration settings, as in inference.
        batch_size: Graphs per integration batch.
        geometry_cache_name / include_mates: Optional overrides; default to the
            flow config's values so the graph matches what the flow model saw.
        device: 'cuda' or 'cpu'.
        overwrite: Re-generate even if a `<pdb_id>.pt` already exists.

    Returns:
        dict stats: {"out_dir", "n_written", "n_skipped", "n_total"}.
    """
    run_dir = Path(flow_run_dir)
    config = load_config(run_dir)
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    # Frozen flow model, loaded exactly as scripts/inference.py loads it.
    model = build_model_from_config(config, device_t)
    checkpoint_path = run_dir / "checkpoints" / checkpoint
    epoch = load_checkpoint(model, checkpoint_path, device_t)
    logger.info(f"Loaded flow checkpoint {checkpoint_path} (epoch {epoch})")

    flow_matcher = FlowMatcher(
        model=model,
        sampling_strategy=config.get("sampling_strategy", "uniform_ball"),
    )

    if include_mates is None:
        include_mates = config.get("include_mates", False)
    if geometry_cache_name is None:
        geometry_cache_name = config.get("geometry_cache_name", "geometry")
    encoder_type = config.get("encoder_type", "gvp")

    dataset = ProteinWaterDataset(
        pdb_list_file=str(pdb_list),
        processed_dir=str(processed_dir),
        base_pdb_dir=str(base_pdb_dir),
        encoder_type=encoder_type,
        include_mates=include_mates,
        # Also picks the cache directory, so it has to track the flow run.
        include_ligands=config.get("include_ligands", True),
        geometry_cache_name=geometry_cache_name,
        preprocess=True,
        **_extract_dataset_filter_config(config),
    )
    logger.info(
        f"Generating candidates for {len(dataset)} structures "
        f"(encoder={encoder_type}, geometry={geometry_cache_name}, "
        f"mates={include_mates}, ratio={water_ratio}, seed={seed})"
    )

    out = (
        Path(out_dir)
        if out_dir is not None
        else default_candidate_dir(processed_dir, run_dir, water_ratio, seed)
    )
    torch.manual_seed(seed)  # reproducible sampling prior

    def sample_batch(graphs):
        return run_inference_batch(
            flow_matcher,
            graphs,
            method=method,
            num_steps=num_steps,
            device=str(device_t),
            water_ratio=water_ratio,
        )

    run_info = {
        "flow_run_dir": str(run_dir),
        "pdb_list": str(pdb_list),
        "checkpoint": checkpoint,
        "epoch": epoch,
        "water_ratio": water_ratio,
        "seed": seed,
        "num_steps": num_steps,
        "method": method,
        "encoder_type": encoder_type,
        "geometry_cache_name": geometry_cache_name,
        "include_mates": include_mates,
    }
    return _write_candidate_cache(
        dataset,
        sample_batch,
        out,
        run_info=run_info,
        batch_size=batch_size,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the candidate cache for confidence-model training."
    )
    p.add_argument(
        "--flow_run_dir",
        required=True,
        help="Flow training run dir (config.json + checkpoints/).",
    )
    p.add_argument(
        "--pdb_list",
        required=True,
        help="Text file of <pdb_id>_final keys, one per line.",
    )
    p.add_argument(
        "--processed_dir",
        required=True,
        help="Cache root shared with flow training (geometry + esm).",
    )
    p.add_argument(
        "--base_pdb_dir",
        required=True,
        help="Base PDB dir, as used by flow training.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output dir. Default: "
        "{processed_dir}/candidate_cache/<run>_r{ratio}_s{seed}.",
    )
    p.add_argument(
        "--checkpoint",
        default="best.pt",
        help="Checkpoint filename under {flow_run_dir}/checkpoints.",
    )
    p.add_argument(
        "--water_ratio",
        type=float,
        default=3.0,
        help="Oversampling: num_residues * water_ratio waters per structure.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the sampling prior (reproducible candidates).",
    )
    p.add_argument("--num_steps", type=int, default=100, help="Integration steps.")
    p.add_argument(
        "--method",
        choices=["euler", "rk4"],
        default="euler",
        help="Integration method.",
    )
    p.add_argument(
        "--batch_size", type=int, default=8, help="Graphs per integration batch."
    )
    p.add_argument(
        "--geometry_cache_name",
        default=None,
        help="Override geometry cache base name (default: flow config).",
    )
    p.add_argument(
        "--include_mates",
        action="store_true",
        default=None,
        help="Force-include symmetry mates (default: flow config).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate even if a <pdb_id>.pt already exists.",
    )
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging_for_tqdm(level=args.log_level)
    stats = generate_candidate_cache(
        flow_run_dir=args.flow_run_dir,
        pdb_list=args.pdb_list,
        processed_dir=args.processed_dir,
        base_pdb_dir=args.base_pdb_dir,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        water_ratio=args.water_ratio,
        seed=args.seed,
        num_steps=args.num_steps,
        method=args.method,
        batch_size=args.batch_size,
        geometry_cache_name=args.geometry_cache_name,
        include_mates=args.include_mates,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(
        f"Done: {stats['n_written']} written, {stats['n_skipped']} skipped "
        f"-> {stats['out_dir']}"
    )


if __name__ == "__main__":
    main()
