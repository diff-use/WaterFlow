#!/usr/bin/env python
"""
Train `ConfidenceGVP` on candidates sampled from a trained flow checkpoint.

Stage two of the pipeline, riding on the flow dataset/cache layout:
    1. A candidate generator samples waters from a trained flow checkpoint and
       writes a thin per-PDB file (`<pdb_id>.pt` holding only `candidate_pos`).
    2. This script fits a head with BCE-with-logits on a soft smootherstep target
       of each candidate's nearest-GT distance (or a hard 1[d<=cutoff] label),
       loading the protein graph, embeddings, and GT waters straight from the
       flow caches via `ProteinWaterDataset` + `ConfidenceDataset`.

`best.pt` is selected on candidate-level AUC-PR at the acceptance label, not on
val loss. Architecture, encoder plumbing, and dataset filters all come from the
flow run's `config.json`, which keeps the two stages in lockstep.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch
from tqdm import tqdm

from scripts.inference import _extract_dataset_filter_config
from src.confidence import ConfidenceGVP
from src.confidence_dataset import ConfidenceDataset
from src.constants import NUM_RBF
from src.dataset import ProteinWaterDataset
from src.distributed import (
    all_gather_concat,
    all_reduce_means,
    ddp_barrier,
    ddp_is_active,
    is_main_process,
    setup_distributed,
    teardown_distributed,
)
from src.encoder_base import build_encoder
from src.utils import auc_pr_and_best_f1, setup_logging_for_tqdm


# Everything except the score head. Frozen together, or not at all.
BACKBONE_MODULE_NAMES = (
    "encoder",
    "encoder_to_flow",
    "protein_scalar_encoder",
    "water_scalar_encoder",
    "updater",
)


def load_config(run_dir: Path) -> dict:
    """
    Read a flow run's recorded `config.json`.

    Args:
        run_dir: The flow training run directory.

    Returns:
        The parsed config.
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def _unwrap(model: ConfidenceGVP | DDP) -> ConfidenceGVP:
    """Return the underlying ConfidenceGVP whether or not it is DDP-wrapped."""
    return model.module if isinstance(model, DDP) else model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a confidence model from a pre-built candidate cache."
    )
    # data / cache paths
    p.add_argument(
        "--flow_run_dir",
        type=str,
        required=True,
        help="Flow training run directory (provides encoder/filter config).",
    )
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument(
        "--candidate_dir",
        type=str,
        required=True,
        help="Per-PDB candidate directory (<pdb_id>.pt with `candidate_pos`), "
        "written by scripts/cache_candidates.py.",
    )
    p.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Cache root shared with flow training (geometry + esm). "
        "Protein graph, embeddings, and GT waters load from here.",
    )
    p.add_argument(
        "--base_pdb_dir",
        type=str,
        required=True,
        help="Base PDB dir, as used by flow training.",
    )
    p.add_argument(
        "--geometry_cache_name",
        type=str,
        default=None,
        help="Override geometry cache base name (default: flow config).",
    )
    p.add_argument(
        "--include_mates",
        action="store_true",
        default=None,
        help="Force-include symmetry mates (default: flow config).",
    )
    p.add_argument(
        "--strict_cache",
        action="store_true",
        help="Raise when a structure has no candidate file. Default: skip "
        "it (the dataset logs how many were dropped).",
    )
    p.add_argument(
        "--max_candidates",
        type=int,
        default=None,
        help="Cap candidates per structure (fresh random subsample of the "
        "pooled cloud each epoch). Bounds per-step memory at no cost "
        "to quality, as candidates are scored independently. Unset "
        "scores all of them.",
    )

    # run control
    p.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Parent directory for confidence training runs.",
    )
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Checkpoint to warm-start from (e.g. a flow or joint "
        "flow+confidence run).",
    )
    p.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the pretrained encoder/shared backbone and train only "
        "the score head.",
    )

    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Micro-batches per optimizer step. Effective batch = n_gpus * "
        "batch_size * grad_accum_steps. Lets an uncapped candidate set "
        "run on a tiny micro-batch while holding the effective batch.",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument(
        "--eta_min_factor",
        type=float,
        default=0.01,
        help="eta_min for cosine = lr * eta_min_factor.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
        help="Main LR scheduler. Both run after the linear warmup.",
    )
    p.add_argument(
        "--plateau_factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau: lr <- lr * factor on plateau.",
    )
    p.add_argument(
        "--plateau_patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau: epochs of no val improvement.",
    )
    p.add_argument("--plateau_min_lr", type=float, default=1e-7)

    # wandb
    p.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="If set, log to this wandb project. Omit to disable wandb.",
    )
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Defaults to --run_name when omitted.",
    )

    # loss shape -- smootherstep target, 0.5-crossing on the acceptance radius
    p.add_argument(
        "--r_in",
        type=float,
        default=0.5,
        help="smootherstep plateau radius (A): conf=1 for d<=r_in.",
    )
    p.add_argument(
        "--r_out",
        type=float,
        default=1.5,
        help="smootherstep floor radius (A): conf=0 for d>=r_out. Crossing "
        "sits at (r_in+r_out)/2, width sets steepness; the 0.5/1.5 "
        "default puts it on the 1A acceptance.",
    )
    p.add_argument(
        "--accept_radius",
        type=float,
        default=1.0,
        help="Acceptance radius (A). Defines the binary AUC-PR validation "
        "label and the cutoff for --hard_label.",
    )
    p.add_argument(
        "--hard_label",
        action="store_true",
        help="Train on 1[d<=accept_radius] instead of the soft "
        "smootherstep, yielding a calibrated P(within radius).",
    )
    # model / system
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable bfloat16 autocast mixed precision (CUDA only).",
    )
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def freeze_backbone(model: ConfidenceGVP) -> None:
    """Freeze the pretrained backbone, leaving the score head trainable."""
    for name in BACKBONE_MODULE_NAMES:
        for param in getattr(model, name).parameters():
            param.requires_grad = False
    for param in model.score_head.parameters():
        param.requires_grad = True


def _set_model_mode(
    model: ConfidenceGVP, *, training: bool, freeze_backbone_enabled: bool
) -> None:
    """Set train/eval mode, holding a frozen backbone in eval while the head trains."""
    model.train(training)
    if training and freeze_backbone_enabled:
        for name in BACKBONE_MODULE_NAMES:
            getattr(model, name).eval()


def build_confidence_model(config: dict, device: torch.device) -> ConfidenceGVP:
    """
    Instantiate `ConfidenceGVP` from the flow run's hyperparameters.

    Mirroring the flow's encoder and hidden dims is what lets the head
    warm-start from a checkpoint that shares the same backbone shape.

    Args:
        config: The flow run's recorded config.
        device: Device to build on.

    Returns:
        The model, on `device`.
    """
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


def _warm_start_from(
    model: ConfidenceGVP, ckpt_path: Path, device: torch.device
) -> None:
    """
    Warm-start the shared backbone from a flow (or confidence) checkpoint.

    A flow checkpoint holds the same backbone as `ConfidenceGVP` plus its own
    velocity head. Loading non-strict keeps the matching backbone tensors,
    ignores the flow-only head (extra keys), and leaves the score head at its
    fresh init (missing keys); any shape-mismatched tensor is skipped too.

    Args:
        model: Model to load into, modified in place.
        ckpt_path: Checkpoint to read.
        device: Map location for the load.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    target = model.state_dict()
    compatible = {
        k: v for k, v in state.items() if k in target and target[k].shape == v.shape
    }
    missing = model.load_state_dict(compatible, strict=False).missing_keys
    logger.info(
        f"Warm-started from {ckpt_path}: loaded {len(compatible)} tensors, "
        f"{len(missing)} left at fresh init."
    )


def _build_loader(
    args: argparse.Namespace,
    pdb_list: str,
    shuffle: bool,
    config: dict,
    *,
    distributed: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, DistributedSampler | None]:
    """
    Build a confidence loader over the flow cache layout plus the candidate files.

    The protein graph, embeddings, and GT waters come from the flow caches via
    `ProteinWaterDataset`, using the flow run's encoder type and filters so the
    graph matches what the flow model saw; candidates come from `candidate_dir`.

    Args:
        args: Parsed CLI arguments.
        pdb_list: Split file to load.
        shuffle: Shuffle the data. Also selects the train-only candidate cap.
        config: The flow run's recorded config.
        distributed: Shard across ranks with a `DistributedSampler`.
        drop_last: Drop the trailing partial batch. Set on the train loader
            under DDP so every rank runs the same number of optimizer steps.

    Returns:
        (loader, sampler); sampler is None when not distributed.
    """
    include_mates = (
        args.include_mates
        if args.include_mates is not None
        else config.get("include_mates", False)
    )
    ds_kwargs = dict(
        pdb_list_file=pdb_list,
        processed_dir=args.processed_dir,
        base_pdb_dir=args.base_pdb_dir,
        encoder_type=config.get("encoder_type", "gvp"),
        include_mates=include_mates,
        # Also picks the cache directory, so it has to track the flow run.
        include_ligands=config.get("include_ligands", True),
        geometry_cache_name=args.geometry_cache_name
        or config.get("geometry_cache_name", "geometry"),
        preprocess=True,
        **_extract_dataset_filter_config(config),
    )
    ds = ConfidenceDataset(
        flow_dataset=ProteinWaterDataset(**ds_kwargs),
        candidate_dir=args.candidate_dir,
        r_in=args.r_in,
        r_out=args.r_out,
        hard_label=args.hard_label,
        accept_radius=args.accept_radius,
        # Cap on the train loader only: it bounds backward memory and the draw is
        # i.i.d., while val stays a fixed full set so its metric is comparable.
        max_candidates=args.max_candidates if shuffle else None,
        strict=args.strict_cache,
    )
    sampler = (
        DistributedSampler(ds, shuffle=shuffle, drop_last=drop_last)
        if distributed
        else None
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        # DataLoader forbids shuffle=True alongside a sampler; the sampler shuffles.
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=lambda b: Batch.from_data_list(b),
    )
    return loader, sampler


def train_one_epoch(
    model: ConfidenceGVP | DDP,
    loader: DataLoader,
    optimizer: AdamW,
    warmup_scheduler,
    device: torch.device,
    args: argparse.Namespace,
    step_counter: int,
    use_amp: bool = False,
    wandb_run=None,
) -> tuple[float, int]:
    """
    Run one training epoch.

    Args:
        model: The model, DDP-wrapped or bare.
        loader: Train loader.
        optimizer: AdamW over the trainable parameters.
        warmup_scheduler: LinearLR stepped per optimizer step, or None.
        device: Compute device.
        args: Parsed CLI arguments.
        step_counter: Optimizer steps taken so far, for warmup and wandb.
        wandb_run: Active wandb run on rank 0, or None.

    Returns:
        (epoch mean loss over candidates, updated step_counter).
    """
    _set_model_mode(
        _unwrap(model), training=True, freeze_backbone_enabled=args.freeze_backbone
    )
    total_loss, total_n = 0.0, 0
    params = [p for group in optimizer.param_groups for p in group["params"]]
    accum = max(1, args.grad_accum_steps)
    # Bound to the wrapper, since only DDP defers the gradient all-reduce.
    no_sync = model.no_sync if isinstance(model, DDP) else None
    n_batches = len(loader)  # Equal across ranks: DistributedSampler + drop_last.

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="train", leave=False)
    for micro, batch in enumerate(pbar, start=1):
        batch = batch.to(device)
        target = batch["water"].target_confidence
        # Under DDP every rank must run one backward per micro-batch or the
        # gradient all-reduce hangs, so an empty candidate set backprops a zero
        # loss touching all params rather than skipping.
        is_empty = target.numel() == 0
        # Step every `accum` micro-batches and on the last one, so a trailing
        # partial window still steps; the stepping backward runs outside no_sync
        # and all-reduces the whole accumulated gradient.
        is_boundary = (micro % accum == 0) or (micro == n_batches)
        sync_ctx = (
            no_sync() if (no_sync is not None and not is_boundary) else nullcontext()
        )

        with sync_ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if is_empty:
                    loss = sum(p.sum() for p in params) * 0.0
                else:
                    preds = model(batch, return_logits=True)
                    loss = F.binary_cross_entropy_with_logits(preds, target)
            (loss / accum).backward()

        if is_boundary:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_counter += 1
            if warmup_scheduler is not None and step_counter <= args.warmup_steps:
                warmup_scheduler.step()

        if is_empty:
            continue

        total_loss += loss.item() * target.size(0)
        total_n += target.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        # wandb_run is rank-0 only (guarded at init); log once per optimizer step.
        if wandb_run is not None and is_boundary:
            wandb_run.log(
                {
                    "train/step_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/step": step_counter,
                },
                step=step_counter,
            )

    # Average the epoch loss across ranks once, so every rank logs the same
    # number and picks the best checkpoint from the same value.
    means, _ = all_reduce_means({"loss": total_loss}, total_n, device)
    return means.get("loss", 0.0), step_counter


@torch.no_grad()
def validate(
    model: ConfidenceGVP | DDP,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    Score the validation split.

    Args:
        model: The model, DDP-wrapped or bare.
        loader: Val loader.
        device: Compute device.
        args: Parsed CLI arguments.

    Returns:
        loss, mae, auc_pr, and best_f1, identical on every rank.
    """
    _set_model_mode(
        _unwrap(model), training=False, freeze_backbone_enabled=args.freeze_backbone
    )
    total_loss, total_n, abs_err = 0.0, 0, 0.0
    score_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    # Ranks only sync after the loop (one gather), so skipping an empty batch
    # here cannot desync them.
    for batch in tqdm(loader, desc="val", leave=False):
        batch = batch.to(device)
        target = batch["water"].target_confidence
        if target.numel() == 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds = model(batch, return_logits=True)
            loss = F.binary_cross_entropy_with_logits(preds, target)
        # MAE is reported in probability space, for interpretability.
        probs = torch.sigmoid(preds)
        abs_err += (probs - target).abs().sum().item()
        total_loss += loss.item() * target.size(0)
        total_n += target.size(0)
        score_chunks.append(probs.detach().float().cpu())
        label_chunks.append(batch["water"].within_accept_radius.detach().float().cpu())

    # Reduce sums so the scheduler and best-checkpoint logic see identical
    # metrics on every rank. DistributedSampler pads val with at most
    # world_size-1 duplicates, a negligible bias on the mean.
    means, _ = all_reduce_means({"loss": total_loss, "mae": abs_err}, total_n, device)
    # AUC-PR ranks candidates globally, so pool the pairs rather than the metric.
    scores = all_gather_concat(
        torch.cat(score_chunks) if score_chunks else torch.empty(0)
    )
    labels = all_gather_concat(
        torch.cat(label_chunks) if label_chunks else torch.empty(0)
    )
    auc_pr, best_f1 = auc_pr_and_best_f1(scores, labels)
    return {
        "loss": means.get("loss", 0.0),
        "mae": means.get("mae", 0.0),
        "auc_pr": auc_pr,
        "best_f1": best_f1,
    }


def save_ckpt(
    path: Path,
    model: ConfidenceGVP,
    optimizer: AdamW,
    epoch: int,
    val_metrics: dict,
    args: argparse.Namespace,
) -> None:
    """Write an unwrapped checkpoint, so it reloads under single-GPU inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    main_proc = is_main_process(rank)
    distributed = ddp_is_active()
    device = (
        torch.device(f"cuda:{local_rank}")
        if distributed
        else torch.device(args.device if torch.cuda.is_available() else "cpu")
    )

    run_dir = Path(args.save_dir) / args.run_name
    # Rank 0 owns the run dir; the others wait so it exists before they touch it.
    if main_proc:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    ddp_barrier()
    setup_logging_for_tqdm(
        level=args.log_level,
        log_file=str(run_dir / "train.log") if main_proc else None,
    )
    if distributed:
        logger.info(
            f"DDP active: rank {rank}/{world_size} (local_rank {local_rank}), "
            f"device {device}."
        )
    logger.info(f"Run directory: {run_dir}")

    flow_config = load_config(Path(args.flow_run_dir))
    if main_proc:
        with open(run_dir / "config.json", "w") as f:
            json.dump(
                {"flow_config": flow_config, "confidence_args": vars(args)}, f, indent=2
            )

    # drop_last on train keeps every rank on the same step count (DDP lockstep).
    train_loader, train_sampler = _build_loader(
        args,
        args.train_list,
        True,
        flow_config,
        distributed=distributed,
        drop_last=distributed,
    )
    val_loader, _ = _build_loader(
        args,
        args.val_list,
        False,
        flow_config,
        distributed=distributed,
        drop_last=False,
    )
    logger.info(
        f"Train samples: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}"
    )

    model = build_confidence_model(flow_config, device)
    if args.init_from is not None:
        _warm_start_from(model, Path(args.init_from), device)
    if args.freeze_backbone:
        freeze_backbone(model)
        logger.info(f"Frozen backbone modules: {', '.join(BACKBONE_MODULE_NAMES)}")

    raw_model = model
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(
            "No trainable parameters remain after applying freeze settings."
        )
    logger.info(
        f"Model parameters: trainable={sum(p.numel() for p in trainable_params):,} / "
        f"total={sum(p.numel() for p in raw_model.parameters()):,}"
    )

    if distributed:
        # broadcast_buffers=False is safe (LayerNorm only, no running stats);
        # find_unused_parameters covers --freeze_backbone and variant configs.
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    warmup_scheduler = (
        LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup_steps
        )
        if args.warmup_steps > 0
        else None
    )
    if args.scheduler == "cosine":
        main_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * args.eta_min_factor
        )
    else:
        main_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    # bfloat16 autocast has fp32-equivalent range, so no loss scaling is needed.
    use_amp = args.use_amp
    if use_amp and device.type != "cuda":
        logger.warning("--use_amp requested but device is not CUDA; ignoring.")
        use_amp = False
    elif use_amp:
        logger.info("AMP enabled: bfloat16 autocast.")

    # Rank 0 only, so every downstream wandb_run guard is implicitly a rank guard.
    wandb_run = None
    if args.wandb_project is not None and main_proc:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.run_name,
            dir=str(run_dir),
            config={
                "confidence_args": vars(args),
                "flow_config_summary": {
                    k: flow_config.get(k)
                    for k in (
                        "encoder_type",
                        "hidden_s",
                        "hidden_v",
                        "flow_layers",
                        "cutoff",
                        "max_neighbors",
                        "n_message_gvps",
                        "n_update_gvps",
                    )
                },
            },
        )
        logger.info(f"wandb run: {wandb_run.name} ({wandb_run.url})")

    # best.pt tracks candidate-level AUC-PR at the acceptance label -- the ranking
    # metric the deliverable cares about -- not val loss.
    best_aucpr = float("-inf")
    step_counter = 0
    for epoch in range(args.epochs):
        # Reshuffle each rank's shard differently per epoch (DDP requirement).
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, step_counter = train_one_epoch(
            model,
            train_loader,
            optimizer,
            warmup_scheduler,
            device,
            args,
            step_counter,
            use_amp,
            wandb_run,
        )
        # Metrics are all-reduced inside, so every rank steps the scheduler on the
        # same value and their learning rates stay in sync.
        val_metrics = validate(model, val_loader, device, args, use_amp)
        if step_counter > args.warmup_steps:
            if isinstance(main_scheduler, ReduceLROnPlateau):
                main_scheduler.step(val_metrics["loss"])
            else:
                main_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"epoch {epoch:3d} | lr={lr:.2e} | train={train_loss:.4f} | "
            f"val={val_metrics['loss']:.4f} | mae={val_metrics['mae']:.4f} | "
            f"auc_pr={val_metrics['auc_pr']:.4f} | f1={val_metrics['best_f1']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_metrics["loss"],
                    "val/mae": val_metrics["mae"],
                    "val/auc_pr": val_metrics["auc_pr"],
                    "val/best_f1": val_metrics["best_f1"],
                    "lr": lr,
                },
                step=step_counter,
            )

        aucpr = val_metrics["auc_pr"]
        is_best = aucpr == aucpr and aucpr > best_aucpr  # `== aucpr` rejects nan
        if main_proc:
            ckpt_dir = run_dir / "checkpoints"
            save_ckpt(
                ckpt_dir / "last.pt", raw_model, optimizer, epoch, val_metrics, args
            )
            if is_best:
                best_aucpr = aucpr
                save_ckpt(
                    ckpt_dir / "best.pt", raw_model, optimizer, epoch, val_metrics, args
                )
                logger.info(f"  new best (val auc_pr = {best_aucpr:.4f})")
        elif is_best:
            best_aucpr = aucpr

    if wandb_run is not None:
        wandb_run.finish()
    teardown_distributed()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
