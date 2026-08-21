# predict_waters.py

"""Predict waters for raw PDB/CIF files.

Per structure: drop existing waters, build the flow graph from protein + hets,
sample candidate waters with a flow checkpoint, score them with a confidence
checkpoint, cluster and threshold, then write the input structure with the
predicted waters added (`<name>_pred.pdb|cif`) and a `<name>_waters.txt` of
`x y z confidence` rows. No ground truth is involved.

NOTE: scripts/inference.py on the other hand, evaluates the flow model alone, 
on cached training-format graphs, against the ground-truth waters.

For esm/slae encoders the protein embeddings must already be in
--processed_dir (see generate_esm_embeddings.py / generate_slae_embeddings.py).

Usage:
    python -m scripts.predict_waters \\
        --flow_run_dir <flow_run> --confidence_run_dir <conf_run> \\
        --struc protein.cif --out_dir out/ --confidence_threshold 0.5

    Density mode keeps a fixed count per residue instead of a cutoff:
        ... --selection density --density_ratio 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import biotite.structure as bts
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from scripts.inference import build_model_from_config, load_config, run_inference_batch
from src.confidence import build_confidence_model, cluster_waters_vdw, ConfidenceGVP
from src.confidence_dataset import _oxygen_features
from src.dataset import parse_asu_with_biotite
from src.flow import FlowMatcher
from src.inference_graph import build_inference_graph
from src.structure_io import merge_waters, read_space_group, write_structure
from src.utils import setup_logging_for_tqdm


DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # confidence mode
DEFAULT_DENSITY_RATIO = 0.6  # density mode, waters per ASU residue


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_state_dict_lenient(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """Load weights with strict=False, warning on any missing/unexpected keys.

    A checkpoint and the current model can differ by a few layers across
    versions; a strict load would raise. The matched weights still transfer,
    unmatched checkpoint keys are dropped, and any unmatched model layer stays
    at init. Missing or unexpected keys are warned, not fatal.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(
            f"{checkpoint_path.name}: {len(missing)} params left at init "
            f"(e.g. {missing[:3]})"
        )
    if unexpected:
        logger.warning(
            f"{checkpoint_path.name}: {len(unexpected)} checkpoint keys ignored "
            f"(e.g. {unexpected[:3]})"
        )
    model.eval()


# ---------------------------------------------------------------------------
# Scoring and selection
# ---------------------------------------------------------------------------


def score_candidates(
    conf_model: ConfidenceGVP,
    graph,
    candidate_pos: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Confidence score in [0, 1] for each candidate, given the protein graph."""
    n = candidate_pos.size(0)
    if n == 0:
        return candidate_pos.new_zeros(0)
    scored = graph.clone()
    scored["water"].pos = candidate_pos.to(device)
    scored["water"].x = _oxygen_features(n, device=device)
    scored["water"].num_nodes = n
    scored = scored.to(device)
    with torch.inference_mode():
        return conf_model(scored).detach().cpu()


def select_waters(
    candidate_pos: torch.Tensor,
    confidences: torch.Tensor,
    *,
    mode: str,
    threshold: float | None = None,
    density_ratio: float | None = None,
    num_asu_residues: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster candidates and cull to the final set.

    confidence: threshold is applied before clustering (sub-threshold
    candidates cannot pull a centroid), then all resulting centroids are kept.
    density: cluster with no cutoff, then keep the top
    floor(density_ratio * num_asu_residues) centroids by confidence.
    """
    if mode == "confidence":
        return cluster_waters_vdw(candidate_pos, confidences, threshold=threshold)
    if mode == "density":
        if density_ratio is None or num_asu_residues is None:
            raise ValueError(
                "density selection needs density_ratio and num_asu_residues"
            )
        pos, conf = cluster_waters_vdw(candidate_pos, confidences)  # sorted desc
        n_keep = int(density_ratio * num_asu_residues)
        return pos[:n_keep], conf[:n_keep]
    raise ValueError(f"Unknown selection mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-structure prediction
# ---------------------------------------------------------------------------


def _input_frame(struc_path: str) -> tuple[bts.AtomArray, np.ndarray, str | None]:
    """Atoms to write out, the ASU protein centroid, and the input space group.

    The centroid is the one build_inference_graph centred on, so adding it back
    returns predicted waters to the input frame. Hets are always written, whether
    or not the flow model saw them: the output is the input structure plus waters.
    """
    protein_atoms, _waters, ligand_atoms = parse_asu_with_biotite(struc_path)
    center = protein_atoms.coord.mean(axis=0)
    kept = protein_atoms + ligand_atoms if len(ligand_atoms) else protein_atoms
    return kept, center, read_space_group(struc_path)


def predict_structures(
    struc_paths: list[str],
    flow_matcher: FlowMatcher,
    conf_model: ConfidenceGVP,
    flow_config: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Predict + write final waters for a batch of structures."""
    graphs, frames, out_names = [], [], []
    encoder_type = flow_config.get("encoder_type", "gvp")
    for path in struc_paths:
        graph = build_inference_graph(
            path,
            encoder_type=encoder_type,
            processed_dir=args.processed_dir,
            include_mates=args.include_mates,
            include_ligands=flow_config.get("include_ligands", True),
            cutoff=flow_config.get("cutoff", 8.0),
            max_neighbors=flow_config.get("max_neighbors", 256),
        )
        graphs.append(graph)
        frames.append(_input_frame(path))
        out_names.append(Path(path).stem)

    # Batched flow sampling -> candidate waters (centred frame).
    results = run_inference_batch(
        flow_matcher,
        graphs,
        method=args.method,
        num_steps=args.num_steps,
        device=str(device),
        water_ratio=args.water_ratio,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for graph, result, (kept, center, space_group), name in zip(
        graphs, results, frames, out_names
    ):
        candidate_pos = torch.as_tensor(result["water_pred"], dtype=torch.float32)
        conf = score_candidates(conf_model, graph, candidate_pos, device)
        sel_pos, sel_conf = select_waters(
            candidate_pos,
            conf,
            mode=args.selection,
            threshold=args.confidence_threshold,
            density_ratio=args.density_ratio,
            num_asu_residues=int(graph["protein"].num_protein_residues),
        )
        # Back to the input frame, then write structure + scored coordinates.
        water_xyz = sel_pos.numpy() + center
        structure = merge_waters(kept, water_xyz)
        write_structure(
            structure,
            str(out_dir / f"{name}_pred{args.out_format}"),
            space_group=space_group,
        )
        xyz_conf = np.column_stack([water_xyz, sel_conf.numpy()])
        np.savetxt(
            out_dir / f"{name}_waters.txt",
            xyz_conf,
            fmt=["%.3f", "%.3f", "%.3f", "%.4f"],
            header="x y z confidence",
        )
        logger.info(f"{name}: {len(water_xyz)} waters -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _collect_struc_paths(args: argparse.Namespace) -> list[str]:
    """Resolve the structures to run: a single --struc, or names from --pdb_list.

    Each list entry is a path under --base_pdb_dir. It may carry a .pdb/.cif
    extension or omit it (both are tried), and may include a subdirectory.
    """
    if args.struc:
        return [args.struc]
    base = Path(args.base_pdb_dir)
    names = [
        ln.strip() for ln in Path(args.pdb_list).read_text().splitlines() if ln.strip()
    ]
    paths = []
    for name in names:
        if (base / name).suffix.lower() in (".cif", ".pdb"):
            candidates = [base / name]
        else:
            candidates = [base / f"{name}{ext}" for ext in (".cif", ".pdb")]
        match = next((c for c in candidates if c.exists()), None)
        if match is not None:
            paths.append(str(match))
        else:
            logger.warning(f"No structure file found for {name!r} under {base}")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flow_run_dir",
        required=True,
        help="Flow run dir (config.json + checkpoints/).",
    )
    p.add_argument("--confidence_run_dir", required=True, help="Confidence run dir.")
    p.add_argument("--flow_checkpoint", default="best.pt")
    p.add_argument("--confidence_checkpoint", default="best.pt")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--struc", help="A single PDB/CIF file.")
    src.add_argument(
        "--pdb_list",
        help="Text file of structure names under --base_pdb_dir, one per line; "
        "each may include or omit a .pdb/.cif extension.",
    )
    p.add_argument("--base_pdb_dir", help="Directory --pdb_list names resolve against.")
    p.add_argument(
        "--processed_dir",
        default=None,
        help="Embedding cache root for esm/slae encoders (unused for gvp). "
        "Embeddings are loaded, not generated: run generate_esm_embeddings.py or "
        "generate_slae_embeddings.py first. Looked up by file stem under "
        "processed_dir/<encoder_type>.",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_format", default=".pdb", choices=[".pdb", ".cif"])

    p.add_argument(
        "--selection",
        default="confidence",
        choices=["confidence", "density"],
        help="Final-water selection rule.",
    )
    p.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="confidence mode: drop candidates scoring below this, in [0, 1] "
        f"(default {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    p.add_argument(
        "--density_ratio",
        type=float,
        default=None,
        help="density mode: keep floor(ratio * ASU residues) waters by confidence "
        f"(default {DEFAULT_DENSITY_RATIO}).",
    )

    p.add_argument(
        "--include_mates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add symmetry mates. Default: the flow run's setting.",
    )
    p.add_argument(
        "--water_ratio",
        type=float,
        default=8.0,
        help="Candidates = ratio * num_residues.",
    )
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--method", default="euler", choices=["euler", "rk4"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_level", default="INFO")

    args = p.parse_args(argv)
    if args.pdb_list and not args.base_pdb_dir:
        p.error("--pdb_list requires --base_pdb_dir")
    # Each mode has one knob. Reject the other mode's and fill the default.
    if args.selection == "confidence":
        if args.density_ratio is not None:
            p.error("--density_ratio only applies to --selection density")
        if args.confidence_threshold is None:
            args.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        if not 0.0 <= args.confidence_threshold <= 1.0:
            p.error("--confidence_threshold must be in [0, 1]")
    else:
        if args.confidence_threshold is not None:
            p.error("--confidence_threshold only applies to --selection confidence")
        if args.density_ratio is None:
            args.density_ratio = DEFAULT_DENSITY_RATIO
        if args.density_ratio <= 0:
            p.error("--density_ratio must be > 0")
    return args


def main() -> None:
    args = parse_args()
    setup_logging_for_tqdm(level=args.log_level)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    flow_dir = Path(args.flow_run_dir)
    flow_config = load_config(flow_dir)
    if args.include_mates is None:
        args.include_mates = flow_config.get("include_mates", False)

    flow_model = build_model_from_config(flow_config, device)
    load_state_dict_lenient(
        flow_model, flow_dir / "checkpoints" / args.flow_checkpoint, device
    )
    flow_matcher = FlowMatcher(
        model=flow_model,
        sampling_strategy=flow_config.get("sampling_strategy", "uniform_ball"),
    )

    conf_dir = Path(args.confidence_run_dir)
    conf_config = load_config(conf_dir)
    conf_config = conf_config.get("flow_config", conf_config)  # confidence runs nest it
    conf_model = build_confidence_model(conf_config, device)
    load_state_dict_lenient(
        conf_model, conf_dir / "checkpoints" / args.confidence_checkpoint, device
    )

    paths = _collect_struc_paths(args)
    logger.info(f"Predicting waters for {len(paths)} structure(s) on {device}")
    for start in tqdm(range(0, len(paths), args.batch_size), desc="predict"):
        predict_structures(
            paths[start : start + args.batch_size],
            flow_matcher,
            conf_model,
            flow_config,
            args,
            device,
        )


if __name__ == "__main__":
    main()
