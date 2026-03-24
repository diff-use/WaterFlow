# inference.py

"""
Inference script for WaterFlow model.

Takes a text file of PDB paths, runs trajectory integration (euler/rk4),
and outputs plots, gifs, and metrics for each PDB.

Usage:
    python scripts/inference.py run_dir=... pdb_list=... output_dir=...
    python scripts/inference.py run_dir=... pdb_list=... output_dir=... integration.method=euler
"""

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.constants import NUM_RBF
from src.dataset import ProteinWaterDataset
from src.encoder_base import build_encoder
from src.flow import FlowMatcher, FlowWaterGVP
from src.utils import (
    compute_placement_metrics,
    compute_rmsd,
    create_trajectory_gif,
    plot_3d_frame,
    setup_logging_for_tqdm,
)


# Configure logging to work with tqdm progress bars
setup_logging_for_tqdm()


def load_config(run_dir: Path) -> dict:
    """
    Load training configuration from run directory.

    Args:
        run_dir: Path to training run directory containing config.yaml

    Returns:
        Dict with training configuration parameters

    Raises:
        FileNotFoundError: If config.yaml doesn't exist in run_dir
    """
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from typing import cast

    from omegaconf import OmegaConf

    config = cast(dict, OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))

    return config


def _extract_dataset_filter_config(config: dict) -> dict:
    """Extract dataset filter params from training config with fallback to defaults."""
    return {
        "max_com_dist": config.get("max_com_dist", 25.0),
        "max_clash_fraction": config.get("max_clash_fraction", 0.05),
        "clash_dist": config.get("clash_dist", 2.0),
        "interface_dist_threshold": config.get("interface_dist_threshold", 4.0),
        "min_water_residue_ratio": config.get("min_water_residue_ratio", 0.6),
        "edia_dir": config.get("edia_dir"),
        "max_protein_dist": config.get("max_protein_dist", 5.0),
        "min_edia": config.get("min_edia", 0.4),
        "max_bfactor_zscore": config.get("max_bfactor_zscore", 1.5),
        "filter_by_distance": config.get("filter_by_distance", True),
        "filter_by_edia": config.get("filter_by_edia", True),
        "filter_by_bfactor": config.get("filter_by_bfactor", True),
    }


def build_model_from_config(config: dict, device: torch.device) -> nn.Module:
    """
    Build model architecture from training configuration.

    Uses registry-based encoder construction to instantiate the correct
    encoder type (GVP, SLAE, or ESM) based on config.

    Args:
        config: Training configuration dict with model hyperparameters.
            Expected keys include:
            - encoder_type: "gvp", "slae", or "esm"
            - hidden_s, hidden_v: Hidden dimensions for scalars/vectors
            - flow_layers: Number of flow layers
            - For cached encoders: embedding_dim and embedding_key="embedding"
        device: Device to place model on

    Returns:
        FlowWaterGVP model instance
    """
    # Use resolved_encoder_config if available (from training), otherwise build from config
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

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(config.get("hidden_s") or 256, config.get("hidden_v") or 64),
        edge_scalar_dim=config.get("edge_scalar_dim") or NUM_RBF,
        layers=config.get("flow_layers") or 3,
        drop_rate=config.get("drop_rate", 0.1),
        n_message_gvps=config.get("n_message_gvps", 2),
        n_update_gvps=config.get("n_update_gvps", 2),
        k_pw=config.get("k_pw") or 16,
        k_ww=config.get("k_ww") or 16,
    ).to(device)

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    """
    Load model weights from checkpoint file.

    Args:
        model: FlowWaterGVP model instance to load weights into
        checkpoint_path: Path to checkpoint .pt file
        device: Device to map checkpoint tensors to

    Returns:
        Epoch number from checkpoint, or None if not stored

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint.get("epoch", None)


def run_inference_batch(
    flow_matcher: FlowMatcher,
    graphs: list,
    method: str,
    num_steps: int,
    use_sc: bool,
    device: str,
    water_ratio: float | None = None,
) -> list:
    """
    Run inference on a batch of graphs.

    Args:
        flow_matcher: FlowMatcher instance
        graphs: List of HeteroData graphs
        method: Integration method ('euler' or 'rk4')
        num_steps: Number of integration steps
        use_sc: Whether to use self-conditioning
        device: Device to run on
        water_ratio: If provided, sample num_residues * water_ratio waters

    Returns:
        List of result dicts, each with:
            - protein_pos, water_true, water_pred
            - trajectory (if rk4)
            - pdb_id
    """
    if method == "rk4":
        results = flow_matcher.rk4_integrate(
            graphs,
            num_steps=num_steps,
            use_sc=use_sc,
            device=device,
            return_trajectory=True,
            water_ratio=water_ratio,
        )
    else:  # euler
        results = flow_matcher.euler_integrate(
            graphs,
            num_steps=num_steps,
            use_sc=use_sc,
            device=device,
            water_ratio=water_ratio,
        )

    return results


def save_plot(
    result: dict,
    pdb_id: str,
    output_path: Path,
    metrics: dict | None,
):
    """
    Save 3D visualization plot of water prediction results.

    Args:
        result: Dict with 'protein_pos', 'water_pred', 'water_true' arrays
        pdb_id: PDB identifier for title
        output_path: Path to save PNG image
        metrics: Dict with 'rmsd', 'precision', 'recall', 'f1' for title, or None if no ground truth
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if metrics is not None:
        title = (
            f"{pdb_id} | RMSD={metrics['rmsd']:.2f}Å | "
            f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} F1={metrics['f1']:.3f}"
        )
    else:
        n_pred = result["water_pred"].shape[0]
        title = f"{pdb_id} | {n_pred} waters predicted (no ground truth)"

    # water_true may be None or empty when no ground truth is available
    water_true = result.get("water_true")
    if water_true is not None and water_true.shape[0] == 0:
        water_true = None

    plot_3d_frame(
        ax,
        result["protein_pos"],
        None,  # no separate mate positions
        result["water_pred"],
        water_true,
        title=title,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    """Run inference pipeline on a list of PDB structures."""
    setup_logging_for_tqdm()

    # setup paths
    run_dir = Path(cfg.run_dir)
    output_root = Path(cfg.output_dir)
    output_dir = output_root / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    if cfg.inference.visualization.save_gifs:
        (output_dir / "gifs").mkdir(exist_ok=True)

    # Device
    device = torch.device(
        cfg.inference.hardware.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # load config and build model
    logger.info(f"Loading model from: {run_dir}")
    config = load_config(run_dir)
    model = build_model_from_config(config, device)

    # load checkpoint
    checkpoint_path = run_dir / "checkpoints" / cfg.inference.checkpoint
    epoch = load_checkpoint(model, checkpoint_path, device)
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")

    # Create FlowMatcher
    flow_matcher = FlowMatcher(
        model=model,
        p_self_cond=config.get("p_self_cond", 0.5),
    )

    # Load dataset
    logger.info(f"Loading PDBs from: {cfg.pdb_list}")

    # Determine include_mates from cfg or training config
    include_mates = cfg.inference.include_mates or config.get("include_mates", False)
    encoder_type = config.get("encoder_type", "gvp")

    # Use geometry_cache if provided, otherwise use config's geometry_cache_name
    geometry_cache_name = cfg.inference.geometry_cache or config.get(
        "geometry_cache_name", "geometry"
    )

    # Extract dataset filter config from training config for consistency
    filter_config = _extract_dataset_filter_config(config)

    dataset = ProteinWaterDataset(
        pdb_list_file=cfg.pdb_list,
        processed_dir=cfg.processed_dir,
        base_pdb_dir=cfg.base_pdb_dir,
        encoder_type=encoder_type,
        include_mates=include_mates,
        geometry_cache_name=geometry_cache_name,
        preprocess=True,
        **filter_config,
    )

    logger.info(f"Found {len(dataset)} PDB entries")
    logger.info(f"Using geometry cache: {geometry_cache_name}")

    # run inference
    method = cfg.inference.integration.method
    num_steps = cfg.inference.integration.num_steps
    use_sc = cfg.inference.integration.use_sc
    threshold = cfg.inference.evaluation.threshold
    batch_size = cfg.inference.hardware.batch_size
    water_ratio = cfg.inference.water.water_ratio

    logger.info(f"Running inference with method={method}, steps={num_steps}")
    logger.info(f"Self-conditioning: {use_sc}")
    logger.info(f"Threshold for metrics: {threshold}Å")
    logger.info(f"Batch size: {batch_size}")

    # Determine if metrics should be skipped
    # Skip metrics when explicitly requested or when using water_ratio (no ground truth count)
    skip_metrics = cfg.inference.evaluation.skip_metrics or water_ratio is not None

    if water_ratio is not None:
        logger.info(
            f"Water ratio: {water_ratio} (sampling num_residues × {water_ratio} waters)"
        )
    else:
        logger.info("Water ratio: None (using ground truth water count)")
    if skip_metrics:
        logger.info("Metrics computation: DISABLED (no ground truth comparison)")
    else:
        logger.info("Metrics computation: ENABLED")
    logger.info("-" * 60)

    all_metrics = []

    # collect graphs for inference
    valid_graphs = []
    skipped_pdbs = []
    for idx in range(len(dataset)):
        graph = dataset[idx]
        # Only skip zero-water PDBs when we need ground truth for metrics
        if graph["water"].num_nodes == 0 and not skip_metrics:
            skipped_pdbs.append(graph.pdb_id)
        else:
            valid_graphs.append(graph)

    if skipped_pdbs:
        logger.info(f"Skipping {len(skipped_pdbs)} PDBs with no water molecules")

    # process in batches
    num_batches = (len(valid_graphs) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(valid_graphs))
        batch_graphs = valid_graphs[start_idx:end_idx]

        # run batched inference
        batch_results = run_inference_batch(
            flow_matcher,
            batch_graphs,
            method=method,
            num_steps=num_steps,
            use_sc=use_sc,
            device=cfg.inference.hardware.device,
            water_ratio=water_ratio,
        )

        # process each result in the batch
        for result in batch_results:
            pdb_id = result.get("pdb_id", f"unknown_{len(all_metrics)}")
            water_pred = result["water_pred"]
            water_true = result["water_true"]
            has_ground_truth = water_true is not None and water_true.shape[0] > 0

            # compute metrics only when ground truth is available and metrics are enabled
            if not skip_metrics and has_ground_truth:
                metrics = compute_placement_metrics(
                    pred=water_pred,
                    true=water_true,
                    threshold=threshold,
                )
                metrics["rmsd"] = compute_rmsd(water_pred, water_true)
                metrics["pdb_id"] = pdb_id
                metrics["n_waters_true"] = water_true.shape[0]
                metrics["n_waters_pred"] = water_pred.shape[0]
                all_metrics.append(metrics)
            else:
                # no ground truth comparison - just store prediction info
                metrics = None

            plot_path = output_dir / "plots" / f"{pdb_id}.png"
            save_plot(result, pdb_id, plot_path, metrics)

            # save GIF if requested and trajectory available
            if cfg.inference.visualization.save_gifs and result.get("trajectory") is not None:
                gif_path = output_dir / "gifs" / f"{pdb_id}.gif"
                # Use water_true only if available
                gif_water_true = water_true if has_ground_truth else None
                create_trajectory_gif(
                    trajectory=result["trajectory"],
                    protein_pos=result["protein_pos"],
                    water_true=gif_water_true,
                    save_path=str(gif_path),
                    title="",
                    fps=10,
                    pdb_id=pdb_id,
                )

            # print per-sample info
            if metrics is not None:
                tqdm.write(
                    f"  {pdb_id}: RMSD={metrics['rmsd']:.2f}Å | "
                    f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} "
                    f"F1={metrics['f1']:.3f} AUC-PR={metrics['auc_pr']:.3f}"
                )
            else:
                tqdm.write(f"  {pdb_id}: {water_pred.shape[0]} waters predicted")

    # compute and save summary metrics
    if all_metrics:
        summary = {
            "n_samples": len(all_metrics),
            "avg_rmsd": float(np.mean([m["rmsd"] for m in all_metrics])),
            "std_rmsd": float(np.std([m["rmsd"] for m in all_metrics])),
            "avg_precision": float(np.mean([m["precision"] for m in all_metrics])),
            "avg_recall": float(np.mean([m["recall"] for m in all_metrics])),
            "avg_f1": float(np.mean([m["f1"] for m in all_metrics])),
            "avg_auc_pr": float(np.mean([m["auc_pr"] for m in all_metrics])),
            "avg_n_waters_true": float(
                np.mean([m["n_waters_true"] for m in all_metrics])
            ),
            "avg_n_waters_pred": float(
                np.mean([m["n_waters_pred"] for m in all_metrics])
            ),
        }

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY METRICS")
        logger.info("=" * 60)
        logger.info(f"  Samples processed: {summary['n_samples']}")
        logger.info(f"  Avg waters (true):  {summary['avg_n_waters_true']:.1f}")
        logger.info(f"  Avg waters (pred):  {summary['avg_n_waters_pred']:.1f}")
        logger.info(
            f"  Avg RMSD:      {summary['avg_rmsd']:.3f} ± {summary['std_rmsd']:.3f} Å"
        )
        logger.info(f"  Avg Precision: {summary['avg_precision']:.3%}")
        logger.info(f"  Avg Recall:    {summary['avg_recall']:.3%}")
        logger.info(f"  Avg F1:        {summary['avg_f1']:.4f}")
        logger.info(f"  Avg AUC-PR:    {summary['avg_auc_pr']:.4f}")
        logger.info("=" * 60)

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "per_sample": all_metrics,
                    "config": {
                        "run_dir": str(run_dir),
                        "checkpoint": cfg.inference.checkpoint,
                        "method": method,
                        "num_steps": num_steps,
                        "use_sc": use_sc,
                        "threshold": threshold,
                        "include_mates": include_mates,
                        "water_ratio": water_ratio,
                        "geometry_cache": geometry_cache_name,
                    },
                },
                f,
                indent=2,
            )
        logger.info(f"Metrics saved to: {metrics_path}")

    else:
        if skip_metrics:
            logger.info(f"Processed {len(valid_graphs)} samples (metrics disabled)")
        else:
            logger.warning("No valid samples processed.")

    logger.info(f"Plots saved to: {output_dir / 'plots'}")
    if cfg.inference.visualization.save_gifs:
        logger.info(f"GIFs saved to: {output_dir / 'gifs'}")

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
