"""
Training pipeline for WaterFlow model.

This module provides the main training script for the WaterFlow water placement
model. It handles:
- Dataset loading and preprocessing with configurable quality filters
- Model construction with pluggable encoders (GVP, SLAE, ESM)
- Training loop with gradient accumulation and warmup scheduling
- Validation and evaluation with RK4 trajectory integration
- Checkpointing and W&B logging

Usage:
    python scripts/train.py \\
        --train_list /path/to/train.txt \\
        --val_list /path/to/val.txt \\
        --encoder_type gvp \\
        --epochs 200 \\
        --batch_size 4
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.dataset import get_dataloader
from src.encoder_base import build_encoder
from src.flow import FlowMatcher, FlowWaterGVP
from src.utils import (
    compute_placement_metrics,
    compute_rmsd,
    create_trajectory_gif,
    plot_3d_frame,
    setup_logging_for_tqdm,
)


def generate_run_name(args: argparse.Namespace) -> str:
    """Generate a run name from timestamp and key parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers = f"L{args.flow_layers}"
    hidden = f"h{args.hidden_s}"
    name = f"{timestamp}_{args.encoder_type}_{layers}_{hidden}"
    return name


def parse_args():
    """
    Parse command-line arguments for training configuration.

    Returns:
        argparse.Namespace with all training hyperparameters and paths
    """
    # TODO: Add support for loading configuration from YAML/JSON config files.
    # This would allow users to save and share training configurations easily.
    # Example: --config config.yaml would load all arguments from the file,
    # with CLI args taking precedence for overrides.

    # TODO: Remove hardcoded default paths. These should be required arguments
    # or loaded from environment variables / config files for portability.
    # Current hardcoded paths:
    #   - processed_dir: /home/srivasv/flow_cache/
    #   - base_pdb_dir: /sb/wankowicz_lab/data/srivasv/pdb_redo_data
    #   - edia_dir: /sb/wankowicz_lab/data/srivasv/edia_results
    #   - save_dir: /home/srivasv/flow_checkpoints
    #   - wandb_dir: /home/srivasv/wandb_logs
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument(
        "--processed_dir",
        type=str,
        default="/home/srivasv/flow_cache/",
        help=(
            "Cache root. Geometry caches are expected in <processed_dir>/geometry, "
            "embeddings in <processed_dir>/<encoder_name>."
        ),
    )
    p.add_argument(
        "--base_pdb_dir",
        type=str,
        default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data",
    )
    p.add_argument(
        "--geometry_cache_name",
        type=str,
        default="geometry",
        help="Base name for geometry cache directory (e.g., 'geometry' -> geometry/ or geometry_unfiltered/)",
    )
    p.add_argument(
        "--include_mates",
        action="store_true",
        help="Include symmetry mate atoms as protein nodes",
    )
    p.add_argument(
        "--duplicate_single_sample",
        type=int,
        default=1,
        help="If training on single sample, duplicate it N times for more gradient updates per epoch",
    )
    p.add_argument(
        "--edia_dir",
        type=str,
        default="/sb/wankowicz_lab/data/srivasv/edia_results",
        help=(
            "Water filter: EDIA root directory "
            "({edia_dir}/{pdb_id}/{pdb_id}_residue_stats.csv)."
        ),
    )

    # dataset quality checks (always on)
    p.add_argument(
        "--max_com_dist",
        type=float,
        default=25.0,
        help="Quality: max allowed protein-water center-of-mass distance (Angstroms).",
    )
    p.add_argument(
        "--max_clash_fraction",
        type=float,
        default=0.05,
        help="Quality: max allowed fraction of waters clashing with protein.",
    )
    p.add_argument(
        "--clash_dist",
        type=float,
        default=2.0,
        help="Quality: distance threshold for defining a water-protein clash (Angstroms).",
    )
    p.add_argument(
        "--interface_dist_threshold",
        type=float,
        default=4.0,
        help="Quality: max inter-chain interface distance to treat chains as interacting (Angstroms).",
    )
    p.add_argument(
        "--min_water_residue_ratio",
        type=float,
        default=0.6,
        help="Quality: minimum waters/residue ratio required per structure.",
    )

    # per-water filtering (toggleable)
    p.add_argument(
        "--max_protein_dist",
        type=float,
        default=5.0,
        help="Water filter: remove waters farther than this from nearest protein atom (Angstroms).",
    )
    p.add_argument(
        "--min_edia",
        type=float,
        default=0.4,
        help="Water filter: remove waters with EDIA below this threshold.",
    )
    p.add_argument(
        "--max_bfactor_zscore",
        type=float,
        default=1.5,
        help="Water filter: remove waters with normalized B-factor above this threshold.",
    )
    p.add_argument(
        "--no_filter_by_distance",
        dest="filter_by_distance",
        action="store_false",
        help="Disable distance-from-protein water filtering (ignores --max_protein_dist).",
    )
    p.add_argument(
        "--no_filter_by_edia",
        dest="filter_by_edia",
        action="store_false",
        help="Disable EDIA-based water filtering (ignores --min_edia).",
    )
    p.add_argument(
        "--no_filter_by_bfactor",
        dest="filter_by_bfactor",
        action="store_false",
        help="Disable B-factor-based water filtering (ignores --max_bfactor_zscore).",
    )
    p.set_defaults(filter_by_distance=True, filter_by_edia=True, filter_by_bfactor=True)

    # model
    p.add_argument(
        "--encoder_type", type=str, default="gvp", choices=["gvp", "slae", "esm"]
    )
    p.add_argument("--encoder_ckpt", type=str, default=None)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--hidden_s", type=int, default=256)
    p.add_argument("--hidden_v", type=int, default=64)
    p.add_argument("--flow_layers", type=int, default=3)
    p.add_argument(
        "--n_message_gvps",
        type=int,
        default=2,
        help="Number of GVPs in message function per edge type (default: 2)",
    )
    p.add_argument(
        "--n_update_gvps",
        type=int,
        default=2,
        help="Number of GVPs in node update function (default: 2)",
    )
    p.add_argument(
        "--drop_rate",
        type=float,
        default=0.1,
        help="Dropout rate for GVP layers (default: 0.1)",
    )
    p.add_argument("--k_pw", type=int, default=16)
    p.add_argument("--k_ww", type=int, default=16)

    # optional cached-embedding override
    p.add_argument(
        "--embedding_dim",
        type=int,
        default=None,
        help="Optional cached embedding dimension override for SLAE/ESM encoders",
    )

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Number of batches to prefetch per worker",
    )
    p.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for faster CPU-GPU transfer",
    )
    p.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep workers alive between epochs",
    )

    # scheduler
    p.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"]
    )
    p.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps")
    p.add_argument(
        "--eta_min_factor",
        type=float,
        default=0.001,
        help="eta_min = lr * eta_min_factor",
    )
    p.add_argument(
        "--step_size", type=int, default=50, help="StepLR step size (epochs)"
    )
    p.add_argument("--step_gamma", type=float, default=0.5, help="StepLR gamma")

    # flow matching
    p.add_argument("--use_self_cond", action="store_true")
    p.add_argument("--p_self_cond", type=float, default=0.5)
    p.add_argument("--use_distortion", action="store_true")
    p.add_argument("--p_distort", type=float, default=0.2)
    p.add_argument("--t_distort", type=float, default=0.5)
    p.add_argument("--sigma_distort", type=float, default=0.5)

    # checkpointing
    p.add_argument("--save_dir", type=str, default="/home/srivasv/flow_checkpoints")
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (auto-generated if not provided)",
    )
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_eval_samples", type=int, default=3)
    p.add_argument("--rk4_steps", type=int, default=100)
    p.add_argument(
        "--save_gifs", action="store_true", help="Save trajectory GIFs during eval"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold in Angstroms for precision/recall (default: 1.0)",
    )

    # logging / wandb
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="water-flow")
    p.add_argument("--wandb_dir", type=str, default="/home/srivasv/wandb_logs")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    if args.encoder_type == "gvp" and args.embedding_dim is not None:
        p.error("--embedding_dim is only valid for cached encoders: slae or esm")
    return args


def _extract_quality_config(args: argparse.Namespace) -> dict:
    """Extract dataset quality check parameters (always active in preprocessing)."""
    return {
        "max_com_dist": args.max_com_dist,
        "max_clash_fraction": args.max_clash_fraction,
        "clash_dist": args.clash_dist,
        "interface_dist_threshold": args.interface_dist_threshold,
        "min_water_residue_ratio": args.min_water_residue_ratio,
    }


def _extract_water_filter_config(args: argparse.Namespace) -> dict:
    """Extract per-water filtering parameters (toggleable)."""
    return {
        "edia_dir": args.edia_dir,
        "max_protein_dist": args.max_protein_dist,
        "min_edia": args.min_edia,
        "max_bfactor_zscore": args.max_bfactor_zscore,
        "filter_by_distance": args.filter_by_distance,
        "filter_by_edia": args.filter_by_edia,
        "filter_by_bfactor": args.filter_by_bfactor,
    }


def _build_dataset_config(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    """
    Build grouped dataset configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (dataset_kwargs, quality_kwargs, water_filter_kwargs):
            - dataset_kwargs: Merged dict for DataLoader creation
            - quality_kwargs: Structure-level quality check parameters
            - water_filter_kwargs: Per-water filtering parameters
    """
    quality_kwargs = _extract_quality_config(args)
    water_filter_kwargs = _extract_water_filter_config(args)
    dataset_kwargs = {
        "encoder_type": args.encoder_type,
        "base_pdb_dir": args.base_pdb_dir,
        "geometry_cache_name": args.geometry_cache_name,
        "include_mates": args.include_mates,
        **quality_kwargs,
        **water_filter_kwargs,
    }
    return dataset_kwargs, quality_kwargs, water_filter_kwargs


def _ignored_water_filter_thresholds(args) -> list[str]:
    """
    Identify water filter thresholds that are disabled.

    Args:
        args: Parsed command-line arguments with filter_by_* flags

    Returns:
        List of threshold parameter names that are disabled (e.g., ['min_edia'])
    """
    ignored = []
    if not args.filter_by_distance:
        ignored.append("max_protein_dist")
    if not args.filter_by_edia:
        ignored.append("min_edia")
    if not args.filter_by_bfactor:
        ignored.append("max_bfactor_zscore")
    return ignored


def _log_dataset_filter_config(args, quality_kwargs: dict):
    """
    Log dataset quality check and water filter configuration.

    Args:
        args: Parsed command-line arguments with filter settings
        quality_kwargs: Structure-level quality check parameters to log
    """
    active_filters = {
        "distance": args.filter_by_distance,
        "edia": args.filter_by_edia,
        "bfactor": args.filter_by_bfactor,
    }
    logger.info(f"Dataset quality checks (always on): {quality_kwargs}")
    logger.info(f"Water filters (toggleable): {active_filters}")

    ignored = _ignored_water_filter_thresholds(args)
    if ignored:
        logger.info(f"Ignored water-filter thresholds (disabled): {ignored}")

    if args.filter_by_edia and args.edia_dir is None:
        logger.info(
            "EDIA filter enabled but --edia_dir is not set; EDIA filtering will be skipped."
        )


def _required_embedding_field(encoder_type: str) -> str | None:
    """
    Get the required embedding field name for a given encoder type.

    Args:
        encoder_type: Encoder identifier ('gvp', 'slae', or 'esm')

    Returns:
        Field name string (e.g., 'embedding') or None if encoder doesn't need embeddings
    """
    if encoder_type in {"slae", "esm"}:
        return "embedding"
    return None


def _uses_cached_embeddings(encoder_type: str) -> bool:
    """Return whether the selected encoder consumes cached protein embeddings."""
    return _required_embedding_field(encoder_type) is not None


def _resolve_embedding_dim(
    sample_data,
    encoder_type: str,
    override_dim: int | None,
) -> int | None:
    """
    Infer or validate embedding dimension from sample data.

    Args:
        sample_data: HeteroData sample from the dataset
        encoder_type: Encoder identifier ('gvp', 'slae', or 'esm')
        override_dim: User-specified dimension override, or None to infer

    Returns:
        Embedding dimension, or None if encoder doesn't use embeddings

    Raises:
        ValueError: If required embedding field is missing or dimension mismatch
    """
    field = _required_embedding_field(encoder_type)
    if field is None:
        return None
    if field not in sample_data["protein"]:
        raise ValueError(
            f"Selected encoder '{encoder_type}' requires protein.{field}, "
            f"but it is missing from dataset samples. "
            f"Expected cached embeddings in data['protein'].embedding from "
            f"--processed_dir/{encoder_type}/<cache_key>.pt."
        )

    embedding_type = sample_data["protein"].get("embedding_type")
    if embedding_type is not None and embedding_type != encoder_type:
        raise ValueError(
            f"Selected encoder '{encoder_type}' requires protein.embedding_type="
            f"'{encoder_type}', but sample data has '{embedding_type}'."
        )

    inferred_dim = int(sample_data["protein"][field].shape[-1])
    if override_dim is not None and int(override_dim) != inferred_dim:
        raise ValueError(
            f"{encoder_type} dim override mismatch: override={override_dim}, "
            f"inferred={inferred_dim} from sample data"
        )
    return inferred_dim if override_dim is None else int(override_dim)


def resolve_encoder_config(args, sample_data, node_scalar_in: int):
    """
    Build a registry-friendly encoder config with inferred dimensions.

    Args:
        args: Parsed command-line arguments containing encoder settings
        sample_data: HeteroData sample used to infer embedding dimensions
        node_scalar_in: Number of input scalar features per node

    Returns:
        dict: Encoder configuration ready for build_encoder(), e.g.:
            - GVP: {"encoder_type": "gvp", "hidden_s": 256, "hidden_v": 64, ...}
            - SLAE: {"encoder_type": "slae", "embedding_key": "embedding", "embedding_dim": 128, ...}
            - ESM: {"encoder_type": "esm", "embedding_key": "embedding", "embedding_dim": 1536, ...}
    """
    encoder_config = {
        "encoder_type": args.encoder_type,
        "hidden_s": args.hidden_s,
        "hidden_v": args.hidden_v,
        "node_scalar_in": node_scalar_in,
        "freeze_encoder": args.freeze_encoder,
        "encoder_ckpt": args.encoder_ckpt,
    }

    if _uses_cached_embeddings(args.encoder_type):
        encoder_config["embedding_key"] = "embedding"
        encoder_config["embedding_dim"] = _resolve_embedding_dim(
            sample_data, args.encoder_type, args.embedding_dim
        )

    return encoder_config


def log_encoder_sample_stats(sample_data: HeteroData, encoder_type: str) -> None:
    """Log summary statistics for the selected encoder input features."""
    field = _required_embedding_field(encoder_type)
    if field is None:
        return
    emb = sample_data["protein"][field]
    embedding_type = sample_data["protein"].get("embedding_type", "unknown")
    logger.info(
        f"{field} type={embedding_type} shape={tuple(emb.shape)} "
        f"mean={emb.mean():.4f} std={emb.std():.4f} min={emb.min():.4f} max={emb.max():.4f}"
    )


def build_model(
    args: argparse.Namespace, device: torch.device, encoder_config: dict
) -> FlowWaterGVP:
    """
    Build encoder and flow model using registry-based encoder construction.

    Args:
        args: Parsed command-line arguments with model hyperparameters
        device: Torch device to place the model on
        encoder_config: Registry-friendly config from resolve_encoder_config()

    Returns:
        FlowWaterGVP: Initialized model with the specified encoder
    """
    logger.info(f"Building model with {args.encoder_type.upper()} encoder")
    logger.info(f"Resolved encoder config: {encoder_config}")

    encoder = build_encoder(encoder_config, device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(args.hidden_s, args.hidden_v),
        layers=args.flow_layers,
        n_message_gvps=args.n_message_gvps,
        n_update_gvps=args.n_update_gvps,
        drop_rate=args.drop_rate,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
    ).to(device)

    return model


def run_eval_sampling(
    flow_matcher, val_loader, args, epoch, device, global_step, eval_indices, run_dir
):
    """Run RK4 integration on fixed eval samples and log results.

    Args:
        eval_indices: Fixed list of dataset indices to evaluate (sampled once at start)
        run_dir: Path to run directory for saving outputs
    """
    flow_matcher.model.eval()
    results = []

    for i, idx in enumerate(eval_indices):
        graph = val_loader.dataset[idx]
        if graph["water"].num_nodes == 0:
            continue

        out = flow_matcher.rk4_integrate(
            graph,
            num_steps=args.rk4_steps,
            use_sc=args.use_self_cond,
            device=device,
            return_trajectory=True,
        )[0]  # rk4_integrate returns a list, get the single result

        # compute metrics
        final_metrics = compute_placement_metrics(
            pred=out["water_pred"], true=out["water_true"], threshold=args.threshold
        )

        final_rmsd = compute_rmsd(out["water_pred"], out["water_true"])

        results.append(
            {
                "rmsd": final_rmsd,
                "precision": final_metrics["precision"],
                "recall": final_metrics["recall"],
                "f1": final_metrics["f1"],
                "auc_pr": final_metrics["auc_pr"],
            }
        )

        # plot final frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_frame(
            ax,
            out["protein_pos"],
            None,
            out["water_pred"],
            out["water_true"],
            title=f"Epoch {epoch} Sample {i} | RMSD={final_rmsd:.2f}A | F1={final_metrics['f1']:.3f}",
        )

        plot_path = run_dir / "plots" / f"epoch{epoch}_sample{i}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # save GIF if requested
        if args.save_gifs and "trajectory" in out:
            gif_path = run_dir / "gifs" / f"epoch{epoch}_sample{i}.gif"
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            create_trajectory_gif(
                trajectory=out["trajectory"],
                protein_pos=out["protein_pos"],
                water_true=out["water_true"],
                save_path=str(gif_path),
                title=f"Epoch {epoch} Sample {i}",
                fps=10,
                pdb_id=graph.pdb_id,
            )

    if results:
        avg_metrics = {
            "eval/avg_rmsd": np.mean([r["rmsd"] for r in results]),
            "eval/avg_precision": np.mean([r["precision"] for r in results]),
            "eval/avg_recall": np.mean([r["recall"] for r in results]),
            "eval/avg_f1": np.mean([r["f1"] for r in results]),
            "eval/avg_auc_pr": np.mean([r["auc_pr"] for r in results]),
        }
        wandb.log(avg_metrics, step=global_step)
        return avg_metrics
    return {}


def train_epoch(
    flow_matcher: FlowMatcher,
    train_loader: DataLoader,
    optimizer: AdamW,
    warmup_scheduler,
    args: argparse.Namespace,
    epoch: int,
    optimizer_step_count: int,
) -> tuple[dict[str, float], int, int]:
    """Single training epoch with gradient accumulation and warmup support."""
    flow_matcher.model.train()
    total_loss, total_rmsd = 0.0, 0.0
    skipped_batches = 0
    processed_batches = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        batch = batch.to(args.device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue

        metrics = flow_matcher.training_step(
            batch,
            use_self_conditioning=args.use_self_cond,
            accumulation_steps=args.grad_accum_steps,
        )

        if metrics["per_sample_info"] is not None:
            per_sample_losses = metrics["per_sample_info"]["losses"].cpu()
            num_graphs = metrics["per_sample_info"]["num_graphs"]

            if hasattr(batch, "pdb_id"):
                pdb_ids = (
                    batch.pdb_id if isinstance(batch.pdb_id, list) else [batch.pdb_id]
                )
                logger.warning("=" * 60)
                logger.warning(f"Batch loss {metrics['loss']:.2f} exceeded 100.0!")
                logger.warning(f"Per-sample losses ({num_graphs} samples):")
                for i in range(num_graphs):
                    pdb_id = pdb_ids[i] if i < len(pdb_ids) else "unknown"
                    sample_loss = per_sample_losses[i].item()
                    logger.warning(f"[{i}] {pdb_id}: {sample_loss:.2f}")
                logger.warning("=" * 60)

        processed_batches += 1
        total_loss += metrics["loss"]
        total_rmsd += metrics["rmsd"]

        # Step optimizer every grad_accum_steps
        if (step + 1) % args.grad_accum_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in flow_matcher.model.parameters() if p.requires_grad],
                    max_norm=args.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1

            # Step warmup scheduler per optimizer step
            if (
                warmup_scheduler is not None
                and optimizer_step_count <= args.warmup_steps
            ):
                warmup_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            rmsd=f"{metrics['rmsd']:.2f}",
            lr=f"{current_lr:.2e}",
        )

        global_step = (epoch - 1) * len(train_loader) + step
        wandb.log(
            {
                "train/iter_loss": metrics["loss"],
                "train/iter_rmsd": metrics["rmsd"],
                "lr": current_lr,
            },
            step=global_step,
        )

    # Handle remaining gradients at end of epoch
    if (step + 1) % args.grad_accum_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in flow_matcher.model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_count += 1
        if warmup_scheduler is not None and optimizer_step_count <= args.warmup_steps:
            warmup_scheduler.step()

    final_global_step = (epoch - 1) * len(train_loader) + len(train_loader) - 1

    if processed_batches == 0:
        logger.warning(
            f"Epoch {epoch}: skipped all {skipped_batches} train batches (no waters)."
        )
        return (
            {"train/epoch_loss": float("inf"), "train/epoch_rmsd": float("inf")},
            final_global_step,
            optimizer_step_count,
        )

    logger.info(
        f"Epoch {epoch} [Train] processed_batches={processed_batches}, skipped_batches={skipped_batches}"
    )
    return (
        {
            "train/epoch_loss": total_loss / processed_batches,
            "train/epoch_rmsd": total_rmsd / processed_batches,
        },
        final_global_step,
        optimizer_step_count,
    )


@torch.no_grad()
def val_epoch(
    flow_matcher: FlowMatcher,
    val_loader: DataLoader,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    """Single validation epoch."""
    flow_matcher.model.eval()
    total_loss, total_rmsd = 0.0, 0.0
    skipped_batches = 0
    processed_batches = 0

    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
        batch = batch.to(args.device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue
        metrics = flow_matcher.validation_step(batch)
        processed_batches += 1
        total_loss += metrics["loss"]
        total_rmsd += metrics["rmsd"]

    if processed_batches == 0:
        logger.warning(
            f"Epoch {epoch}: skipped all {skipped_batches} val batches (no waters)."
        )
        return {"val/loss": float("inf"), "val/rmsd": float("inf")}

    logger.info(
        f"Epoch {epoch} [Val] processed_batches={processed_batches}, skipped_batches={skipped_batches}"
    )
    return {
        "val/loss": total_loss / processed_batches,
        "val/rmsd": total_rmsd / processed_batches,
    }


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_checkpoint(
    model,
    optimizer,
    warmup_scheduler,
    main_scheduler,
    epoch,
    optimizer_step_count,
    path,
    best=False,
):
    """
    Save model checkpoint with optimizer and scheduler states.

    Args:
        model: FlowWaterGVP model instance
        optimizer: AdamW optimizer instance
        warmup_scheduler: LinearLR warmup scheduler, or None
        main_scheduler: Main LR scheduler (CosineAnnealingLR or StepLR), or None
        epoch: Current epoch number
        optimizer_step_count: Total number of optimizer steps taken
        path: Path object for checkpoint file destination
        best: If True, log as best checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_step_count": optimizer_step_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "warmup_scheduler_state_dict": warmup_scheduler.state_dict()
            if warmup_scheduler
            else None,
            "main_scheduler_state_dict": main_scheduler.state_dict()
            if main_scheduler
            else None,
        },
        path,
    )
    logger.info(f"{'Best ' if best else ''}Checkpoint saved: {path}")


def build_scheduler(optimizer, args):
    """
    Build warmup and main learning rate schedulers.

    Supports hybrid stepping: warmup scheduler steps per optimizer step,
    main scheduler steps per epoch after warmup completes.

    Args:
        optimizer: AdamW optimizer instance
        args: Parsed arguments with scheduler configuration

    Returns:
        Tuple of (warmup_scheduler, main_scheduler), either may be None
    """
    # Warmup scheduler (stepped per optimizer step)
    warmup_scheduler = None
    if args.warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup_steps
        )

    # Main scheduler (stepped per epoch, after warmup)
    main_scheduler = None
    if args.scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * args.eta_min_factor
        )
    elif args.scheduler == "step":
        main_scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )

    return warmup_scheduler, main_scheduler


def main():
    """Run the full training pipeline."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = generate_run_name(args)

    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "gifs").mkdir(exist_ok=True)

    log_file = Path(args.log_file) if args.log_file else run_dir / "train.log"
    setup_logging_for_tqdm(level=args.log_level, log_file=str(log_file))

    logger.info("=" * 60)
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    # data loaders
    dataset_kwargs, quality_kwargs, _ = _build_dataset_config(args)
    _log_dataset_filter_config(args, quality_kwargs)

    train_loader = get_dataloader(
        pdb_list_file=args.train_list,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        duplicate_single_sample=args.duplicate_single_sample,
        **dataset_kwargs,
    )

    val_loader = get_dataloader(
        pdb_list_file=args.val_list,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        duplicate_single_sample=args.duplicate_single_sample,
        **dataset_kwargs,
    )

    # sample fixed eval indices
    np.random.seed(42)
    eval_indices = np.random.choice(
        len(val_loader.dataset),
        min(args.n_eval_samples, len(val_loader.dataset)),
        replace=False,
    ).tolist()

    eval_indices_file = run_dir / "eval_indices.txt"
    with open(eval_indices_file, "w") as f:
        f.write("# Fixed evaluation sample indices\n")
        for idx in eval_indices:
            graph = val_loader.dataset[idx]
            pdb_id = getattr(graph, "pdb_id", "unknown")
            f.write(f"{idx}\t{pdb_id}\n")
    logger.info(f"Fixed eval indices saved to: {eval_indices_file}")
    logger.info(f"Evaluating on {len(eval_indices)} proteins at each eval epoch")

    # detect input dimension and resolve encoder configuration from sample data
    sample_data = train_loader.dataset[0]
    node_scalar_in = int(sample_data["protein"].x.shape[-1])
    logger.info(f"Detected protein input dimension: {node_scalar_in}")

    log_encoder_sample_stats(sample_data, args.encoder_type)
    encoder_config = resolve_encoder_config(
        args, sample_data, node_scalar_in=node_scalar_in
    )

    config_dict = vars(args).copy()
    config_dict["active_water_filters"] = {
        "distance": args.filter_by_distance,
        "edia": args.filter_by_edia,
        "bfactor": args.filter_by_bfactor,
    }
    config_dict["ignored_water_filter_thresholds"] = _ignored_water_filter_thresholds(
        args
    )
    config_dict["node_scalar_in"] = node_scalar_in
    config_dict["resolved_encoder_config"] = encoder_config
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Configuration saved to: {config_file}")

    wandb.init(
        project=args.wandb_project,
        dir=args.wandb_dir,
        name=args.run_name,
        config=config_dict,
    )

    model = build_model(args, device, encoder_config=encoder_config)
    trainable_params, total_params = count_parameters(model)
    logger.info("Model statistics:")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # quick forward pass sanity check for cached embedding encoders
    if _uses_cached_embeddings(args.encoder_type):
        logger.info(f"Testing forward pass with {args.encoder_type.upper()}...")
        model.eval()
        batch = next(iter(train_loader)).to(device)
        with torch.no_grad():
            num_graphs = int(batch["protein"].batch.max().item()) + 1
            t = torch.zeros(num_graphs, device=device)
            v_out = model(batch, t)
            logger.info(f"Forward pass successful! Output shape: {v_out.shape}")
            logger.info(f"Output stats: mean={v_out.mean():.4f}, std={v_out.std():.4f}")
            if v_out.std() < 1e-6:
                logger.warning("Model output is constant! This indicates a problem.")
        model.train()

    flow_matcher = FlowMatcher(
        model=model,
        p_self_cond=args.p_self_cond,
        use_distortion=args.use_distortion,
        p_distort=args.p_distort,
        t_distort=args.t_distort,
        sigma_distort=args.sigma_distort,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_scheduler, main_scheduler = build_scheduler(optimizer, args)

    best_val_loss = float("inf")
    optimizer_step_count = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step, optimizer_step_count = train_epoch(
            flow_matcher,
            train_loader,
            optimizer,
            warmup_scheduler,
            args,
            epoch,
            optimizer_step_count,
        )
        # Log epoch-level metrics with epoch number for per-epoch tracking
        train_metrics["epoch"] = epoch
        wandb.log(train_metrics, step=global_step)

        val_metrics = val_epoch(flow_matcher, val_loader, args, epoch)
        val_metrics["epoch"] = epoch
        wandb.log(val_metrics, step=global_step)

        # Step main scheduler per epoch (after warmup completes)
        if main_scheduler is not None and optimizer_step_count >= args.warmup_steps:
            main_scheduler.step()

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['train/epoch_loss']:.4f}, "
            f"val_loss={val_metrics['val/loss']:.4f}, val_rmsd={val_metrics['val/rmsd']:.2f}"
        )

        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            save_checkpoint(
                model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / "best.pt",
                best=True,
            )

        if epoch % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / f"epoch_{epoch}.pt",
            )

        if epoch % args.eval_every == 0:
            eval_metrics = run_eval_sampling(
                flow_matcher,
                val_loader,
                args,
                epoch,
                device,
                global_step,
                eval_indices,
                run_dir,
            )
            if eval_metrics:
                logger.info(
                    f"Eval: RMSD={eval_metrics['eval/avg_rmsd']:.2f}A, "
                    f"Precision={eval_metrics['eval/avg_precision']:.2%}, "
                    f"Recall={eval_metrics['eval/avg_recall']:.2%}, "
                    f"F1={eval_metrics['eval/avg_f1']:.3f}, "
                    f"AUC-PR={eval_metrics['eval/avg_auc_pr']:.3f}"
                )

    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed with an unhandled exception.")
        raise
