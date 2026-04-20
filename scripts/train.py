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
    python scripts/train.py train_list=splits/train.txt val_list=splits/val.txt
    python scripts/train.py train_list=... model=slae training.batch_size=8
"""

from datetime import datetime
from pathlib import Path
from typing import cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
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


def generate_run_name(cfg: DictConfig) -> str:
    """Generate a run name from timestamp and key parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers = f"L{cfg.model.flow_layers}"
    hidden = f"h{cfg.model.hidden_s}"
    name = f"{timestamp}_{cfg.model.encoder_type}_{layers}_{hidden}"
    return name


def _extract_quality_config(cfg: DictConfig) -> dict:
    """Extract dataset quality check parameters (always active in preprocessing)."""
    return {
        "max_com_dist": cfg.data.quality.max_com_dist,
        "max_clash_fraction": cfg.data.quality.max_clash_fraction,
        "clash_dist": cfg.data.quality.clash_dist,
        "interface_dist_threshold": cfg.data.quality.interface_dist_threshold,
        "min_water_residue_ratio": cfg.data.quality.min_water_residue_ratio,
    }


def _extract_water_filter_config(cfg: DictConfig) -> dict:
    """Extract per-water filtering parameters (toggleable)."""
    return {
        "edia_dir": cfg.data.edia_dir,
        "max_protein_dist": cfg.data.water_filter.max_protein_dist,
        "min_edia": cfg.data.water_filter.min_edia,
        "max_bfactor_zscore": cfg.data.water_filter.max_bfactor_zscore,
        "filter_by_distance": cfg.data.water_filter.filter_by_distance,
        "filter_by_edia": cfg.data.water_filter.filter_by_edia,
        "filter_by_bfactor": cfg.data.water_filter.filter_by_bfactor,
    }


def _build_dataset_config(cfg: DictConfig) -> tuple[dict, dict, dict]:
    """
    Build grouped dataset configuration from Hydra config.

    Args:
        cfg: Hydra DictConfig

    Returns:
        Tuple of (dataset_kwargs, quality_kwargs, water_filter_kwargs):
            - dataset_kwargs: Merged dict for DataLoader creation
            - quality_kwargs: Structure-level quality check parameters
            - water_filter_kwargs: Per-water filtering parameters
    """
    quality_kwargs = _extract_quality_config(cfg)
    water_filter_kwargs = _extract_water_filter_config(cfg)
    dataset_kwargs = {
        "encoder_type": cfg.model.encoder_type,
        "base_pdb_dir": cfg.data.base_pdb_dir,
        "geometry_cache_name": cfg.data.geometry_cache_name,
        "include_mates": cfg.data.include_mates,
        **quality_kwargs,
        **water_filter_kwargs,
    }
    return dataset_kwargs, quality_kwargs, water_filter_kwargs


def _ignored_water_filter_thresholds(cfg: DictConfig) -> list[str]:
    """
    Identify water filter thresholds that are disabled.

    Args:
        cfg: Hydra DictConfig with data.water_filter settings

    Returns:
        List of threshold parameter names that are disabled (e.g., ['min_edia'])
    """
    ignored = []
    if not cfg.data.water_filter.filter_by_distance:
        ignored.append("max_protein_dist")
    if not cfg.data.water_filter.filter_by_edia:
        ignored.append("min_edia")
    if not cfg.data.water_filter.filter_by_bfactor:
        ignored.append("max_bfactor_zscore")
    return ignored


def _log_dataset_filter_config(cfg: DictConfig, quality_kwargs: dict):
    """
    Log dataset quality check and water filter configuration.

    Args:
        cfg: Hydra DictConfig with filter settings
        quality_kwargs: Structure-level quality check parameters to log
    """
    active_filters = {
        "distance": cfg.data.water_filter.filter_by_distance,
        "edia": cfg.data.water_filter.filter_by_edia,
        "bfactor": cfg.data.water_filter.filter_by_bfactor,
    }
    logger.info(f"Dataset quality checks (always on): {quality_kwargs}")
    logger.info(f"Water filters (toggleable): {active_filters}")

    ignored = _ignored_water_filter_thresholds(cfg)
    if ignored:
        logger.info(f"Ignored water-filter thresholds (disabled): {ignored}")

    if cfg.data.water_filter.filter_by_edia and cfg.data.edia_dir is None:
        logger.info(
            "EDIA filter enabled but edia_dir is not set; EDIA filtering will be skipped."
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


def resolve_encoder_config(cfg: DictConfig, sample_data, node_scalar_in: int):
    """
    Build a registry-friendly encoder config with inferred dimensions.

    Args:
        cfg: Hydra DictConfig containing encoder settings
        sample_data: HeteroData sample used to infer embedding dimensions
        node_scalar_in: Number of input scalar features per node

    Returns:
        dict: Encoder configuration ready for build_encoder(), e.g.:
            - GVP: {"encoder_type": "gvp", "hidden_s": 256, "hidden_v": 64, ...}
            - SLAE: {"encoder_type": "slae", "embedding_key": "embedding", "embedding_dim": 128, ...}
            - ESM: {"encoder_type": "esm", "embedding_key": "embedding", "embedding_dim": 1536, ...}
    """
    encoder_config = {
        "encoder_type": cfg.model.encoder_type,
        "hidden_s": cfg.model.hidden_s,
        "hidden_v": cfg.model.hidden_v,
        "node_scalar_in": node_scalar_in,
        "freeze_encoder": cfg.model.freeze_encoder,
        "encoder_ckpt": cfg.model.encoder_ckpt,
    }

    if _uses_cached_embeddings(cfg.model.encoder_type):
        encoder_config["embedding_key"] = "embedding"
        encoder_config["embedding_dim"] = _resolve_embedding_dim(
            sample_data, cfg.model.encoder_type, cfg.model.embedding_dim
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
    cfg: DictConfig, device: torch.device, encoder_config: dict
) -> FlowWaterGVP:
    """
    Build encoder and flow model using registry-based encoder construction.

    Args:
        cfg: Hydra DictConfig with model hyperparameters
        device: Torch device to place the model on
        encoder_config: Registry-friendly config from resolve_encoder_config()

    Returns:
        FlowWaterGVP: Initialized model with the specified encoder
    """
    logger.info(f"Building model with {cfg.model.encoder_type.upper()} encoder")
    logger.info(f"Resolved encoder config: {encoder_config}")

    encoder = build_encoder(encoder_config, device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(cfg.model.hidden_s, cfg.model.hidden_v),
        layers=cfg.model.flow_layers,
        n_message_gvps=cfg.model.n_message_gvps,
        n_update_gvps=cfg.model.n_update_gvps,
        drop_rate=cfg.model.drop_rate,
        k_pw=cfg.model.k_pw,
        k_ww=cfg.model.k_ww,
    ).to(device)

    return model


def run_eval_sampling(
    flow_matcher, val_loader, cfg, epoch, device, global_step, eval_indices, run_dir
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
            num_steps=cfg.logging.rk4_steps,
            use_sc=cfg.flow.use_self_cond,
            device=device,
            return_trajectory=True,
        )[0]  # rk4_integrate returns a list, get the single result

        # compute metrics
        final_metrics = compute_placement_metrics(
            pred=out["water_pred"],
            true=out["water_true"],
            threshold=cfg.logging.threshold,
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
        if cfg.logging.save_gifs and "trajectory" in out:
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
    cfg: DictConfig,
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
        batch = batch.to(cfg.logging.device)
        if batch["water"].num_nodes == 0:
            skipped_batches += 1
            continue

        metrics = flow_matcher.training_step(
            batch,
            use_self_conditioning=cfg.flow.use_self_cond,
            accumulation_steps=cfg.training.optimizer.grad_accum_steps,
        )

        if metrics["per_sample_info"] is not None:
            per_sample_info = cast(dict, metrics["per_sample_info"])
            per_sample_losses = per_sample_info["losses"].cpu()
            num_graphs = per_sample_info["num_graphs"]

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
        total_loss += cast(float, metrics["loss"])
        total_rmsd += cast(float, metrics["rmsd"])

        # Step optimizer every grad_accum_steps
        if (step + 1) % cfg.training.optimizer.grad_accum_steps == 0:
            if cfg.training.optimizer.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in flow_matcher.model.parameters() if p.requires_grad],
                    max_norm=cfg.training.optimizer.grad_clip,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1

            # Step warmup scheduler per optimizer step
            if (
                warmup_scheduler is not None
                and optimizer_step_count <= cfg.training.scheduler.warmup_steps
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
    if (step + 1) % cfg.training.optimizer.grad_accum_steps != 0:
        if cfg.training.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in flow_matcher.model.parameters() if p.requires_grad],
                max_norm=cfg.training.optimizer.grad_clip,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_count += 1
        if (
            warmup_scheduler is not None
            and optimizer_step_count <= cfg.training.scheduler.warmup_steps
        ):
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
    cfg: DictConfig,
    epoch: int,
) -> dict[str, float]:
    """Single validation epoch."""
    flow_matcher.model.eval()
    total_loss, total_rmsd = 0.0, 0.0
    skipped_batches = 0
    processed_batches = 0

    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
        batch = batch.to(cfg.logging.device)
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


def build_scheduler(optimizer, cfg: DictConfig):
    """
    Build warmup and main learning rate schedulers.

    Supports hybrid stepping: warmup scheduler steps per optimizer step,
    main scheduler steps per epoch after warmup completes.

    Args:
        optimizer: AdamW optimizer instance
        cfg: Hydra DictConfig with scheduler configuration

    Returns:
        Tuple of (warmup_scheduler, main_scheduler), either may be None
    """
    # Warmup scheduler (stepped per optimizer step)
    warmup_scheduler = None
    if cfg.training.scheduler.warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=cfg.training.scheduler.warmup_steps,
        )

    # Main scheduler (stepped per epoch, after warmup)
    main_scheduler = None
    if cfg.training.scheduler.type == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.training.optimizer.lr * cfg.training.scheduler.eta_min_factor,
        )
    elif cfg.training.scheduler.type == "step":
        main_scheduler = StepLR(
            optimizer,
            step_size=cfg.training.scheduler.step_size,
            gamma=cfg.training.scheduler.step_gamma,
        )

    return warmup_scheduler, main_scheduler


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the full training pipeline."""
    device = torch.device(cfg.logging.device if torch.cuda.is_available() else "cpu")

    # Generate run name if not provided
    run_name = cfg.logging.run_name or generate_run_name(cfg)

    # Determine save directory (Hydra changes cwd, so use absolute paths)
    save_dir = Path(cfg.logging.save_dir) if cfg.logging.save_dir else Path.cwd()
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "gifs").mkdir(exist_ok=True)

    log_file = (
        Path(cfg.logging.log_file) if cfg.logging.log_file else run_dir / "train.log"
    )
    setup_logging_for_tqdm(level=cfg.logging.log_level, log_file=str(log_file))

    logger.info("=" * 60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    # Save resolved Hydra config
    config_file = run_dir / "config.yaml"
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Configuration saved to: {config_file}")

    # data loaders
    dataset_kwargs, quality_kwargs, _ = _build_dataset_config(cfg)
    _log_dataset_filter_config(cfg, quality_kwargs)

    train_loader = get_dataloader(
        pdb_list_file=cfg.train_list,
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader.num_workers,
        pin_memory=cfg.training.dataloader.pin_memory,
        prefetch_factor=cfg.training.dataloader.prefetch_factor,
        persistent_workers=cfg.training.dataloader.persistent_workers,
        duplicate_single_sample=cfg.data.duplicate_single_sample,
        **dataset_kwargs,
    )

    val_loader = get_dataloader(
        pdb_list_file=cfg.val_list,
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.dataloader.num_workers,
        pin_memory=cfg.training.dataloader.pin_memory,
        prefetch_factor=cfg.training.dataloader.prefetch_factor,
        persistent_workers=cfg.training.dataloader.persistent_workers,
        duplicate_single_sample=cfg.data.duplicate_single_sample,
        **dataset_kwargs,
    )

    # sample fixed eval indices
    np.random.seed(42)
    eval_indices = np.random.choice(
        len(val_loader.dataset),
        min(cfg.logging.n_eval_samples, len(val_loader.dataset)),
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

    log_encoder_sample_stats(sample_data, cfg.model.encoder_type)
    encoder_config = resolve_encoder_config(
        cfg, sample_data, node_scalar_in=node_scalar_in
    )

    # Create config dict for wandb logging
    config_dict = cast(dict, OmegaConf.to_container(cfg, resolve=True))
    config_dict["active_water_filters"] = {
        "distance": cfg.data.water_filter.filter_by_distance,
        "edia": cfg.data.water_filter.filter_by_edia,
        "bfactor": cfg.data.water_filter.filter_by_bfactor,
    }
    config_dict["ignored_water_filter_thresholds"] = _ignored_water_filter_thresholds(
        cfg
    )
    config_dict["node_scalar_in"] = node_scalar_in
    config_dict["resolved_encoder_config"] = encoder_config

    wandb.init(
        project=cfg.logging.wandb.project,
        dir=cfg.logging.wandb.dir,
        name=run_name,
        config=config_dict,
    )

    model = build_model(cfg, device, encoder_config=encoder_config)
    trainable_params, total_params = count_parameters(model)
    logger.info("Model statistics:")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Total parameters: {total_params:,}")

    # quick forward pass sanity check for cached embedding encoders
    if _uses_cached_embeddings(cfg.model.encoder_type):
        logger.info(f"Testing forward pass with {cfg.model.encoder_type.upper()}...")
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
        p_self_cond=cfg.flow.p_self_cond,
        use_distortion=cfg.flow.use_distortion,
        p_distort=cfg.flow.p_distort,
        t_distort=cfg.flow.t_distort,
        sigma_distort=cfg.flow.sigma_distort,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    warmup_scheduler, main_scheduler = build_scheduler(optimizer, cfg)

    best_val_loss = float("inf")
    optimizer_step_count = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics, global_step, optimizer_step_count = train_epoch(
            flow_matcher,
            train_loader,
            optimizer,
            warmup_scheduler,
            cfg,
            epoch,
            optimizer_step_count,
        )
        # Log epoch-level metrics with epoch number for per-epoch tracking
        train_metrics["epoch"] = epoch
        wandb.log(train_metrics, step=global_step)

        val_metrics = val_epoch(flow_matcher, val_loader, cfg, epoch)
        val_metrics["epoch"] = epoch
        wandb.log(val_metrics, step=global_step)

        # Step main scheduler per epoch (after warmup completes)
        if (
            main_scheduler is not None
            and optimizer_step_count >= cfg.training.scheduler.warmup_steps
        ):
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

        if epoch % cfg.logging.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                warmup_scheduler,
                main_scheduler,
                epoch,
                optimizer_step_count,
                run_dir / "checkpoints" / f"epoch_{epoch}.pt",
            )

        if epoch % cfg.logging.eval_every == 0:
            eval_metrics = run_eval_sampling(
                flow_matcher,
                val_loader,
                cfg,
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
