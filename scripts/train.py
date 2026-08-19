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
import contextlib
import json
import multiprocessing as mp
import os
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.dataset import get_dataloader, ProteinWaterDataset
from src.distributed import (
    all_reduce_means,
    ddp_barrier,
    ddp_is_active,
    ddp_rank_and_world,
    is_main_process,
    run_once_on_main,
    setup_distributed,
    teardown_distributed,
)
from src.encoder_base import build_encoder
from src.flow import DYNAMIC_EDGE_POLICIES, FlowMatcher, FlowWaterGVP
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
        "--include_ligands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include ligand, ion, cofactor and nucleic acid heavy atoms as protein "
            "nodes. Disabling appends '_noligands' to the geometry cache directory, "
            "so the two configs cache separately."
        ),
    )
    p.add_argument(
        "--duplicate_single_sample",
        type=int,
        default=1,
        help="If training on single sample, duplicate it N times for more gradient updates per epoch",
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
        default=0.1,
        help=(
            "Quality: minimum waters/residue ratio required per structure. Applied "
            "at cache-write time, so it decides which structures the cache holds."
        ),
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
        default=2.0,
        help=(
            "Water filter: remove waters with normalized B-factor above this "
            "threshold. Baked in at cache-write time, so a warm cache built at a "
            "different value is refused rather than extended."
        ),
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
    # flow-matching prior
    p.add_argument(
        "--sampling_strategy",
        type=str,
        default="uniform_ball",
        choices=["uniform_ball", "scaled_gaussian"],
        help=(
            "Source distribution for the flow prior. Also resolves "
            "--dynamic_edge_policy auto (default: uniform_ball)"
        ),
    )

    # edge construction
    p.add_argument(
        "--dynamic_edge_policy",
        type=str,
        default="auto",
        choices=["auto", *DYNAMIC_EDGE_POLICIES],
        help=(
            "How water-touching edges are built: 'radius' connects everything "
            "within --cutoff, 'knn' takes a fixed neighbour count, "
            "'knn_if_isolated' is radius plus a rescue for stranded waters. "
            "'auto' picks radius under uniform_ball and knn_if_isolated under "
            "scaled_gaussian (default: auto)"
        ),
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff in Angstroms for radius edges (default: 8.0)",
    )
    p.add_argument(
        "--max_neighbors",
        type=int,
        default=256,
        help="Per-source cap on radius query results (default: 256)",
    )
    p.add_argument(
        "--knn_fallback_k",
        type=int,
        default=8,
        help=(
            "Nearest neighbours attached to waters the radius query stranded; "
            "0 disables the rescue. Ignored under --dynamic_edge_policy knn "
            "(default: 8)"
        ),
    )
    p.add_argument(
        "--disable_ww",
        action="store_true",
        help="Ablate water->water edges",
    )
    p.add_argument(
        "--disable_wp",
        action="store_true",
        help="Ablate water->protein edges",
    )
    p.add_argument(
        "--k_pw",
        type=int,
        default=12,
        help="Nearest neighbours for protein->water edges under 'knn' (default: 12)",
    )
    p.add_argument(
        "--k_ww",
        type=int,
        default=8,
        help="Nearest neighbours for water->water edges under 'knn' (default: 8)",
    )
    p.add_argument(
        "--k_wp",
        type=int,
        default=8,
        help="Nearest neighbours for water->protein edges under 'knn' (default: 8)",
    )

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
    p.add_argument(
        "--sample_cache_size",
        type=int,
        default=0,
        help="Per-worker in-process dataset sample LRU cache size (0 disables caching)",
    )
    p.add_argument(
        "--cache_load_mmap",
        action="store_true",
        default=False,
        help="Use mmap-backed torch.load for dataset cache files when supported",
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
        "--lr_decay_epochs",
        type=int,
        default=None,
        help="Cosine T_max in epochs, decoupled from --epochs so the LR can fully "
        "anneal over a chosen window and then hold at eta_min. Defaults to --epochs.",
    )
    p.add_argument(
        "--step_size", type=int, default=50, help="StepLR step size (epochs)"
    )
    p.add_argument("--step_gamma", type=float, default=0.5, help="StepLR gamma")

    # mixed precision / optimizer
    p.add_argument(
        "--use_amp",
        action="store_true",
        help="Run the training forward pass under bfloat16 autocast (CUDA only).",
    )
    p.add_argument(
        "--fused_adamw",
        action="store_true",
        help="Use the fused AdamW implementation (CUDA only).",
    )

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
    p.add_argument(
        "--eval_method",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="Integrator for the sampling eval.",
    )
    p.add_argument(
        "--eval_steps", type=int, default=50, help="Integration steps for the sampling eval."
    )
    p.add_argument(
        "--selection_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "f1", "auc_pr", "blend"],
        help="Metric that selects best.pt. 'val_loss' uses the per-epoch validation "
        "loss; 'f1'/'auc_pr'/'blend' use the sampling-eval metrics, rolling-3 "
        "smoothed to absorb sampling noise. 'blend' = 0.85*F1 + 0.15*AUC-PR.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest epoch checkpoint under save_dir/run_name.",
    )
    p.add_argument(
        "--save_gifs", action="store_true", help="Save trajectory GIFs during eval"
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold in Angstroms for precision/recall (default: 1.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for weight init and data shuffling. Unset leaves them unseeded.",
    )
    p.add_argument(
        "--val_seed",
        type=int,
        default=1234,
        help="Seed for eval-sample selection and eval integration, for reproducible eval.",
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
    if args.sample_cache_size < 0:
        p.error("--sample_cache_size must be >= 0")
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
        "include_ligands": args.include_ligands,
        "sample_cache_size": args.sample_cache_size,
        "cache_load_mmap": args.cache_load_mmap,
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
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        dynamic_edge_policy=args.dynamic_edge_policy,
        # "auto" depends on which prior the run uses, so pass that through.
        sampling_strategy=args.sampling_strategy,
        knn_fallback_k=args.knn_fallback_k,
        disable_ww=args.disable_ww,
        disable_wp=args.disable_wp,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
        k_wp=args.k_wp,
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
    # Fix the eval-sampling RNG so the per-epoch generative metrics are comparable
    # across epochs (the water prior and integration both draw noise).
    torch.manual_seed(args.val_seed)

    # Each rank integrates a disjoint stride of eval_indices; the metric sums are
    # all-reduced below so every rank ends with identical averages.
    rank, world_size = ddp_rank_and_world()
    results = []

    integrate = (
        flow_matcher.euler_integrate
        if args.eval_method == "euler"
        else flow_matcher.rk4_integrate
    )
    for i, idx in enumerate(eval_indices):
        # Shard by global position i, so plot/GIF filenames never collide.
        if i % world_size != rank:
            continue
        graph = val_loader.dataset[idx]
        if graph["water"].num_nodes == 0:
            continue

        # rk4 returns a per-frame trajectory (used for GIFs); euler does not.
        out = integrate(
            graph,
            num_steps=args.eval_steps,
            use_sc=args.use_self_cond,
            device=device,
        )[0]  # integrators return a list; take the single result

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

    # Every rank must reach this, even one whose stride was all zero-water graphs.
    avg_metrics, _ = all_reduce_means(
        {
            f"eval/avg_{key}": sum(r[key] for r in results)
            for key in ("rmsd", "precision", "recall", "f1", "auc_pr")
        },
        len(results),
        device,
    )
    if not avg_metrics:
        return {}
    wandb.log(avg_metrics, step=global_step)
    return avg_metrics


def _needs_grad_sync(step: int, n_batches: int, accum_steps: int) -> bool:
    """
    Whether this micro-step's backward must all-reduce gradients under DDP.

    True on every accumulation boundary, and throughout the epoch's trailing
    partial window: that window ends in an optimizer.step() too, and stepping on
    gradients that were never all-reduced leaves the ranks permanently diverged.
    """
    if (step + 1) % accum_steps == 0:
        return True
    return step >= n_batches - (n_batches % accum_steps)


def train_epoch(
    flow_matcher: FlowMatcher,
    train_loader: DataLoader,
    optimizer: AdamW,
    warmup_scheduler,
    args: argparse.Namespace,
    device: torch.device,
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
        batch = batch.to(device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue

        # Suppress the gradient all-reduce on micro-steps that are not followed by
        # an optimizer.step(), keeping comms at one all-reduce per optimizer step.
        no_sync = ddp_is_active() and not _needs_grad_sync(
            step, len(train_loader), args.grad_accum_steps
        )
        with flow_matcher.model.no_sync() if no_sync else contextlib.nullcontext():
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

    # All-reduce the metric sums once per epoch, so the logged numbers cover every
    # rank's shard. Must run before the zero-batch check below: a rank that skipped
    # all its batches would return early and never join the all-reduce, hanging the
    # rest.
    train_metrics, processed_batches = all_reduce_means(
        {"train/epoch_loss": total_loss, "train/epoch_rmsd": total_rmsd},
        processed_batches,
        device,
    )

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
    return train_metrics, final_global_step, optimizer_step_count


@torch.no_grad()
def val_epoch(
    flow_matcher: FlowMatcher,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Single validation epoch."""
    flow_matcher.model.eval()
    total_loss, total_rmsd = 0.0, 0.0
    skipped_batches = 0
    processed_batches = 0

    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
        batch = batch.to(device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue
        metrics = flow_matcher.validation_step(batch)
        processed_batches += 1
        total_loss += metrics["loss"]
        total_rmsd += metrics["rmsd"]

    # Best-checkpoint selection keys off val/loss, so ranks must agree on it.
    val_metrics, processed_batches = all_reduce_means(
        {"val/loss": total_loss, "val/rmsd": total_rmsd},
        processed_batches,
        device,
    )

    if processed_batches == 0:
        logger.warning(
            f"Epoch {epoch}: skipped all {skipped_batches} val batches (no waters)."
        )
        return {"val/loss": float("inf"), "val/rmsd": float("inf")}

    logger.info(
        f"Epoch {epoch} [Val] processed_batches={processed_batches}, skipped_batches={skipped_batches}"
    )
    return val_metrics


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _latest_epoch_checkpoint(ckpt_dir: Path) -> Path | None:
    """Highest-numbered epoch_<n>.pt in ckpt_dir, or None if there is none."""
    ckpts = list(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))


def save_checkpoint(
    model,
    optimizer,
    warmup_scheduler,
    main_scheduler,
    epoch,
    optimizer_step_count,
    path,
    best=False,
    best_val_loss=None,
    best_sel_score=None,
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
            "best_val_loss": best_val_loss,
            "best_sel_score": best_sel_score,
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
        t_max = args.lr_decay_epochs if args.lr_decay_epochs is not None else args.epochs
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=args.lr * args.eta_min_factor
        )
    elif args.scheduler == "step":
        main_scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )

    return warmup_scheduler, main_scheduler


def _build_cache_shard(
    list_file: str, processed_dir: str, dataset_kwargs: dict
) -> None:
    """
    Pool worker: build the geometry cache for one shard's list.

    Already-cached entries are skipped, and shards hold disjoint keys, so workers
    never write the same file.
    """
    ProteinWaterDataset(
        pdb_list_file=list_file,
        processed_dir=processed_dir,
        preprocess=True,
        **dataset_kwargs,
    )


def build_cache(args: argparse.Namespace) -> None:
    """
    Build the geometry cache for the train+val lists as the sole writer.

    Preprocessing is CPU/PyMOL only, so this runs before the DDP group exists --
    hence race-free. A warm cache is a fast no-op; a cold build is parallelized
    across CPU cores over disjoint key shards.
    """
    dataset_kwargs, _, _ = _build_dataset_config(args)
    ids = set()
    for lst in (args.train_list, args.val_list):
        with open(lst) as f:
            ids.update(line.strip() for line in f if line.strip())
    if not ids:
        return
    sorted_ids = sorted(ids)

    tmpdir = Path(tempfile.mkdtemp(prefix="wf_build_"))
    try:
        union = tmpdir / "union.txt"
        union.write_text("\n".join(sorted_ids) + "\n")
        # Parse-only probe to find which entries still need building.
        probe = ProteinWaterDataset(
            pdb_list_file=str(union),
            processed_dir=args.processed_dir,
            preprocess=False,
            **dataset_kwargs,
        )
        # Entries can repeat a cache_key; dedup so each file is stat-ed once.
        keys = list(dict.fromkeys(entry["cache_key"] for entry in probe.entries))
        missing = [k for k in keys if not (probe.geometry_dir / f"{k}.pt").is_file()]
        if not missing:
            return  # warm cache: nothing to build

        logger.info(f"build_cache: preprocessing {len(missing)} missing entries")
        # One worker per CPU over disjoint key shards. A single shard still goes
        # through the pool, so PyMOL never runs in this (parent) process.
        n_shards = max(1, min(len(missing), os.cpu_count() or 1))
        shard_files = []
        for i in range(n_shards):
            shard = tmpdir / f"shard_{i}.txt"
            shard.write_text("\n".join(missing[i::n_shards]) + "\n")
            shard_files.append(str(shard))
        # spawn (not fork): safe alongside PyMOL's C extension and any threads.
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_shards) as pool:
            pool.starmap(
                _build_cache_shard,
                [(shard, args.processed_dir, dataset_kwargs) for shard in shard_files],
            )
    except Exception:
        logger.exception("build_cache failed; other ranks will block until timeout")
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    """Run the full training pipeline."""
    args = parse_args()

    # Seed weight init and data shuffling when requested; left unseeded otherwise.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Build the cache on rank 0 before the NCCL group exists. A warm cache is a
    # no-op probe; only a first cold build is slow, and coordinating that on the
    # NCCL group would trip its collective watchdog (minutes). A CPU store has no
    # such watchdog, so rank 0 can take as long as it needs; it is then reused as
    # NCCL's rendezvous.
    store = run_once_on_main(lambda: build_cache(args), key="wf_cache_ready")

    # Under torchrun each rank binds its own GPU; a plain launch yields (0, 0, 1).
    rank, local_rank, world_size = setup_distributed(store=store)
    main_proc = is_main_process(rank)
    # Under torchrun each rank owns the GPU that setup_distributed pinned with
    # set_device(local_rank); a plain launch uses --device (CPU if no CUDA). We do
    # not write this back to args, so the recorded config stays rank-independent.
    if ddp_is_active():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.use_amp and device.type != "cuda":
        logger.warning("--use_amp set but device is not CUDA; training without AMP.")

    if args.run_name is None:
        args.run_name = generate_run_name(args)

    run_dir = Path(args.save_dir) / args.run_name
    if main_proc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        (run_dir / "gifs").mkdir(exist_ok=True)
    ddp_barrier()  # other ranks wait until run_dir exists

    # Rank 0 owns the log file; other ranks log to console only, so N processes
    # never interleave writes into it.
    log_file = Path(args.log_file) if args.log_file else run_dir / "train.log"
    setup_logging_for_tqdm(
        level=args.log_level, log_file=str(log_file) if main_proc else None
    )

    logger.info("=" * 60)
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    if ddp_is_active():
        logger.info(
            f"DDP active: rank={rank} local_rank={local_rank} world_size={world_size}"
        )
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
        distributed=ddp_is_active(),
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
        distributed=ddp_is_active(),
        **dataset_kwargs,
    )

    # sample fixed eval indices
    np.random.seed(args.val_seed)
    eval_indices = np.random.choice(
        len(val_loader.dataset),
        min(args.n_eval_samples, len(val_loader.dataset)),
        replace=False,
    ).tolist()

    eval_indices_file = run_dir / "eval_indices.txt"
    if main_proc:
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
    if main_proc:
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to: {config_file}")

    # Non-main ranks run a disabled client, so every wandb.log call site stays a
    # no-op without a per-call guard. None on the main rank defers to WANDB_MODE
    # (default: online); an explicit mode here would override that env var.
    wandb.init(
        project=args.wandb_project,
        dir=args.wandb_dir,
        name=args.run_name,
        config=config_dict,
        mode=None if main_proc else "disabled",
    )

    model = build_model(args, device, encoder_config=encoder_config)
    if ddp_is_active():
        # broadcast_buffers=False is safe (no BatchNorm; LayerNorm has no synced
        # buffers). find_unused_parameters=True because ablated edge types can
        # leave the used-parameter set varying across backwards.
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    # Unwrapped module: parameter access, sanity forward, sampling, and
    # state_dicts. Saving the wrapper would prefix every key with "module.".
    raw_model = getattr(model, "module", model)

    trainable_params, total_params = count_parameters(raw_model)
    logger.info("Model statistics:")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # quick forward pass sanity check for cached embedding encoders
    if _uses_cached_embeddings(args.encoder_type):
        logger.info(f"Testing forward pass with {args.encoder_type.upper()}...")
        raw_model.eval()
        batch = next(iter(train_loader)).to(device)
        with torch.no_grad():
            num_graphs = int(batch["protein"].batch.max().item()) + 1
            t = torch.zeros(num_graphs, device=device)
            v_out = raw_model(batch, t)
            logger.info(f"Forward pass successful! Output shape: {v_out.shape}")
            logger.info(f"Output stats: mean={v_out.mean():.4f}, std={v_out.std():.4f}")
            if v_out.std() < 1e-6:
                logger.warning("Model output is constant! This indicates a problem.")
        raw_model.train()

    flow_matcher = FlowMatcher(
        model=model,
        p_self_cond=args.p_self_cond,
        sampling_strategy=args.sampling_strategy,
        use_distortion=args.use_distortion,
        p_distort=args.p_distort,
        t_distort=args.t_distort,
        sigma_distort=args.sigma_distort,
        use_amp=args.use_amp,
    )

    # fused AdamW is a CUDA-only kernel; it silently requires all params on GPU.
    optimizer = AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=args.fused_adamw and device.type == "cuda",
    )
    warmup_scheduler, main_scheduler = build_scheduler(optimizer, args)

    best_val_loss = float("inf")
    best_sel_score = float("-inf")
    sel_history: list[float] = []
    optimizer_step_count = 0
    start_epoch = 0

    if args.resume:
        ckpt_path = _latest_epoch_checkpoint(run_dir / "checkpoints")
        if ckpt_path is None:
            raise FileNotFoundError(
                f"--resume set but no epoch_*.pt found under {run_dir / 'checkpoints'}."
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if warmup_scheduler is not None and ckpt.get("warmup_scheduler_state_dict"):
            warmup_scheduler.load_state_dict(ckpt["warmup_scheduler_state_dict"])
        if main_scheduler is not None and ckpt.get("main_scheduler_state_dict"):
            main_scheduler.load_state_dict(ckpt["main_scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        optimizer_step_count = ckpt["optimizer_step_count"]
        best_val_loss = ckpt.get("best_val_loss")
        best_val_loss = float("inf") if best_val_loss is None else best_val_loss
        best_sel_score = ckpt.get("best_sel_score")
        best_sel_score = float("-inf") if best_sel_score is None else best_sel_score
        logger.info(f"Resumed from {ckpt_path} at epoch {start_epoch}.")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Without this every epoch replays the same shard order on every rank.
        if ddp_is_active():
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        train_metrics, global_step, optimizer_step_count = train_epoch(
            flow_matcher,
            train_loader,
            optimizer,
            warmup_scheduler,
            args,
            device,
            epoch,
            optimizer_step_count,
        )
        # Log epoch-level metrics with epoch number for per-epoch tracking
        train_metrics["epoch"] = epoch
        wandb.log(train_metrics, step=global_step)

        val_metrics = val_epoch(flow_matcher, val_loader, device, epoch)
        val_metrics["epoch"] = epoch
        wandb.log(val_metrics, step=global_step)

        # Step the main scheduler once per epoch after warmup. A cosine scheduler
        # holds at eta_min once it reaches T_max; stepping past T_max would make the
        # LR climb back up, so stop stepping there (matters when lr_decay_epochs < epochs).
        if main_scheduler is not None and optimizer_step_count >= args.warmup_steps:
            past_cosine_horizon = (
                isinstance(main_scheduler, CosineAnnealingLR)
                and main_scheduler.last_epoch >= main_scheduler.T_max
            )
            if not past_cosine_horizon:
                main_scheduler.step()

        logger.info(
            f"Epoch {epoch}: train_loss={train_metrics['train/epoch_loss']:.4f}, "
            f"val_loss={val_metrics['val/loss']:.4f}, val_rmsd={val_metrics['val/rmsd']:.2f}"
        )

        # Sampling eval runs before selection so a generative --selection_metric can
        # use this epoch's numbers. All ranks enter (it ends in a collective); swap in
        # the unwrapped module so no DDP forward machinery fires during integration.
        eval_metrics = {}
        if epoch % args.eval_every == 0:
            wrapped = flow_matcher.model
            flow_matcher.model = raw_model
            try:
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
            finally:
                flow_matcher.model = wrapped
            if eval_metrics:
                logger.info(
                    f"Eval: RMSD={eval_metrics['eval/avg_rmsd']:.2f}A, "
                    f"Precision={eval_metrics['eval/avg_precision']:.2%}, "
                    f"Recall={eval_metrics['eval/avg_recall']:.2%}, "
                    f"F1={eval_metrics['eval/avg_f1']:.3f}, "
                    f"AUC-PR={eval_metrics['eval/avg_auc_pr']:.3f}"
                )

        # Every quantity below is all-reduced, so all ranks select the same epoch;
        # only rank 0 writes to disk. best_val_loss is kept as checkpoint metadata
        # regardless of which metric drives selection.
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]

        improved = False
        if args.selection_metric == "val_loss":
            sel = -val_metrics["val/loss"]
            if sel > best_sel_score:
                best_sel_score = sel
                improved = True
        elif eval_metrics:  # generative metric, defined only on eval epochs
            if args.selection_metric == "blend":
                raw = (
                    0.85 * eval_metrics["eval/avg_f1"]
                    + 0.15 * eval_metrics["eval/avg_auc_pr"]
                )
            else:
                raw = eval_metrics[f"eval/avg_{args.selection_metric}"]
            sel_history.append(raw)
            sel = sum(sel_history[-3:]) / len(sel_history[-3:])  # rolling-3 smoothed
            if sel > best_sel_score:
                best_sel_score = sel
                improved = True

        if improved and main_proc:
            save_checkpoint(
                raw_model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / "best.pt",
                best=True,
                best_val_loss=best_val_loss,
                best_sel_score=best_sel_score,
            )

        if epoch % args.save_every == 0 and main_proc:
            save_checkpoint(
                raw_model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / f"epoch_{epoch}.pt",
                best_val_loss=best_val_loss,
                best_sel_score=best_sel_score,
            )

        # Realign ranks: rank 0 may have spent extra time writing checkpoints.
        ddp_barrier()

    wandb.finish()
    teardown_distributed()
    logger.info("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed with an unhandled exception.")
        raise
