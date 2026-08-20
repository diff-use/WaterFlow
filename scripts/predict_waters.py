# predict_waters.py

"""End-to-end water prediction from a raw PDB/CIF file.

Takes structures (with or without waters), strips them to protein + hets, samples
candidate waters from a trained flow checkpoint, scores them with a confidence
checkpoint, clusters and selects the final set, and writes out predicted-water
structures (PDB/CIF) plus their coordinates.

This is the *prediction* entry point; inference.py remains the cache-based
evaluation/benchmark tool. Shared flow machinery is imported, not duplicated.

Usage:
    python -m scripts.predict_waters \\
        --flow_run_dir <flow_run> --confidence_run_dir <conf_run> \\
        --struc protein.cif --out_dir out/ \\
        --selection confidence --threshold 0.5

    Density mode keeps a fixed count instead of a cutoff:
        ... --selection density --density_ratio 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

from scripts.inference import build_model_from_config, load_config, run_inference_batch
from src.confidence import cluster_waters_vdw, ConfidenceGVP
from src.constants import NUM_RBF
from src.dataset import element_onehot, parse_asu_with_biotite
from src.encoder_base import build_encoder
from src.flow import FlowMatcher
from src.inference_graph import build_inference_graph
from src.structure_io import merge_waters, write_structure
from src.utils import setup_logging_for_tqdm


VDW_RADIUS_A = 1.52  # oxygen vdW radius; clustering/NMS radius
DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # confidence-mode cutoff when unset
DEFAULT_DENSITY_RATIO = 0.6  # density-mode waters per ASU residue when unset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def build_confidence_model(config: dict, device: torch.device) -> ConfidenceGVP:
    """Instantiate ConfidenceGVP from a flow run's config (mirrors train_confidence)."""
    resolved = config.get("resolved_encoder_config")
    if resolved:
        encoder_config = resolved.copy()
    else:
        encoder_type = config.get("encoder_type", "gvp")
        encoder_config = {
            "encoder_type": encoder_type,
            "hidden_s": config.get("hidden_s") or 256,
            "hidden_v": config.get("hidden_v") or 64,
            "node_scalar_in": config.get("node_scalar_in") or 16,
            "freeze_encoder": config.get("freeze_encoder", False),
            "encoder_ckpt": config.get("encoder_ckpt"),
        }
        if encoder_type in {"slae", "esm"}:
            encoder_config["embedding_key"] = "embedding"
            encoder_config["embedding_dim"] = config.get("embedding_dim")

    encoder = build_encoder(encoder_config, device)
    return ConfidenceGVP(
        encoder=encoder,
        hidden_dims=(config.get("hidden_s") or 256, config.get("hidden_v") or 64),
        edge_scalar_dim=config.get("edge_scalar_dim") or NUM_RBF,
        layers=config.get("flow_layers") or 3,
        drop_rate=config.get("drop_rate", 0.1),
        n_message_gvps=config.get("n_message_gvps", 2),
        n_update_gvps=config.get("n_update_gvps", 2),
        cutoff=config.get("cutoff", 8.0),
        max_neighbors=config.get("max_neighbors", 256),
        dynamic_edge_policy="knn_if_isolated",
    ).to(device)


def load_state_dict_lenient(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """Load weights with strict=False, warning on any missing/unexpected keys.

    Older flow checkpoints predate the self-conditioning layers, so a strict load
    would raise; the trained weights still transfer, and the extra layers stay at
    init (harmless when self-conditioning is off at inference).
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
    scored["water"].x = element_onehot(["O"] * n).to(device)
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
    radius: float = VDW_RADIUS_A,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster the scored candidates and cull to the final set of waters.

    confidence: the threshold is applied *before* clustering, so a
    sub-threshold candidate cannot pull a centroid off-site; every resulting
    centroid is kept.
    density: cluster with no cutoff, then keep the highest-confidence
    N = floor(density_ratio * num_asu_residues) centroids. cluster_waters_vdw
    returns centroids in descending confidence, so this is a slice.
    """
    if mode == "confidence":
        return cluster_waters_vdw(
            candidate_pos, confidences, radius=radius, threshold=threshold
        )
    if mode == "density":
        if density_ratio is None or num_asu_residues is None:
            raise ValueError(
                "density selection needs density_ratio and num_asu_residues"
            )
        centroids, centroid_conf = cluster_waters_vdw(
            candidate_pos, confidences, radius=radius, threshold=None
        )
        n_keep = int(density_ratio * num_asu_residues)  # floor for a positive ratio
        return centroids[:n_keep], centroid_conf[:n_keep]
    raise ValueError(f"Unknown selection mode: {mode!r}")


# ---------------------------------------------------------------------------
# Per-structure prediction
# ---------------------------------------------------------------------------


def _kept_atoms_and_center(struc_path: str):
    """Protein + hets to write out, and the ASU protein centroid used to centre.

    Recomputes the centroid build_inference_graph used, so predicted waters
    (sampled in the centred frame) return to the input reference frame.
    """
    protein_atoms, _waters, ligand_atoms = parse_asu_with_biotite(struc_path)
    center = protein_atoms.coord.mean(axis=0)
    kept = protein_atoms + ligand_atoms if len(ligand_atoms) else protein_atoms
    return kept, center


def predict_structures(
    struc_paths: list[str],
    flow_matcher: FlowMatcher,
    conf_model: ConfidenceGVP,
    flow_config: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Predict + write final waters for a batch of structures."""
    graphs, centers, kept_atoms, out_names = [], [], [], []
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
        kept, center = _kept_atoms_and_center(path)
        graphs.append(graph)
        centers.append(center)
        kept_atoms.append(kept)
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

    for graph, result, center, kept, name in zip(
        graphs, results, centers, kept_atoms, out_names
    ):
        candidate_pos = torch.as_tensor(result["water_pred"], dtype=torch.float32)
        conf = score_candidates(conf_model, graph, candidate_pos, device)
        sel_pos, sel_conf = select_waters(
            candidate_pos,
            conf,
            mode=args.selection,
            threshold=args.threshold,
            density_ratio=args.density_ratio,
            num_asu_residues=int(graph["protein"].num_protein_residues),
        )
        # Back to the input frame, then write structure + coordinates.
        water_xyz = sel_pos.numpy() + center
        structure = merge_waters(kept, water_xyz)
        write_structure(structure, str(out_dir / f"{name}_pred{args.out_format}"))
        np.savetxt(out_dir / f"{name}_waters.txt", water_xyz, fmt="%.3f")
        logger.info(f"{name}: {len(water_xyz)} waters -> {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _collect_struc_paths(args: argparse.Namespace) -> list[str]:
    if args.struc:
        return [args.struc]
    keys = [
        ln.strip() for ln in Path(args.pdb_list).read_text().splitlines() if ln.strip()
    ]
    base = Path(args.base_pdb_dir)
    paths = []
    for key in keys:
        pdb_id = key.removesuffix("_final")
        for ext in (".cif", ".pdb"):
            cand = base / pdb_id / f"{key}{ext}"
            if cand.exists():
                paths.append(str(cand))
                break
        else:
            logger.warning(f"No structure file found for {key} under {base}")
    return paths


def parse_args() -> argparse.Namespace:
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
    src.add_argument("--pdb_list", help="Text file of <pdb_id>_final keys.")
    p.add_argument("--base_pdb_dir", help="Base dir for --pdb_list entries.")
    p.add_argument(
        "--processed_dir", default=None, help="Embedding cache root (esm/slae only)."
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
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Confidence cutoff for --selection confidence "
            f"(default {DEFAULT_CONFIDENCE_THRESHOLD}). Not allowed with density."
        ),
    )
    p.add_argument(
        "--density_ratio",
        type=float,
        default=None,
        help=(
            f"Waters per ASU residue for --selection density "
            f"(default {DEFAULT_DENSITY_RATIO}); keeps "
            f"floor(density_ratio * num_ASU_residues) by confidence. "
            f"Not allowed with confidence."
        ),
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

    args = p.parse_args()
    if args.pdb_list and not args.base_pdb_dir:
        p.error("--pdb_list requires --base_pdb_dir")

    # --threshold belongs to confidence mode and --density_ratio to density mode.
    # Reject the other one rather than silently ignoring it, and fill the mode's
    # own default when the user left it unset.
    if args.selection == "confidence":
        if args.density_ratio is not None:
            p.error("--density_ratio only applies to --selection density")
        if args.threshold is None:
            args.threshold = DEFAULT_CONFIDENCE_THRESHOLD
    else:  # density
        if args.threshold is not None:
            p.error("--threshold only applies to --selection confidence")
        if args.density_ratio is None:
            args.density_ratio = DEFAULT_DENSITY_RATIO
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
