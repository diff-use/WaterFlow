# train.py

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import get_dataloader
from src.encoder_base import build_encoder
from src.flow import FlowMatcher, FlowWaterGVP
from src.utils import (
    compute_placement_metrics,
    compute_rmsd,
    create_trajectory_gif,
    plot_3d_frame,
)


def generate_run_name(args):
    """Generate a run name from timestamp and key parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoder_type = "slae" if args.use_slae else "gvp"
    layers = f"L{args.flow_layers}"
    hidden = f"h{args.hidden_s}"
    name = f"{timestamp}_{encoder_type}_{layers}_{hidden}"
    return name


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument("--processed_dir", type=str, default="/home/srivasv/flow_cache/")
    p.add_argument("--base_pdb_dir", type=str, default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data")
    p.add_argument("--include_mates", action="store_true", help="Include symmetry mate atoms as protein nodes")
    p.add_argument("--duplicate_single_sample", type=int, default=1,
                   help="If training on single sample, duplicate it N times for more gradient updates per epoch")

    # model
    p.add_argument("--encoder_ckpt", type=str, default=None)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--hidden_s", type=int, default=256)
    p.add_argument("--hidden_v", type=int, default=64)
    p.add_argument("--flow_layers", type=int, default=3)
    p.add_argument("--k_pw", type=int, default=16)
    p.add_argument("--k_ww", type=int, default=16)

    # SLAE encoder options
    p.add_argument("--use_slae", action="store_true", help="Use SLAE encoder instead of GVP")
    p.add_argument("--slae_dim", type=int, default=128, help="SLAE embedding dimension")

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)

    # flow matching
    p.add_argument("--use_self_cond", action="store_true")
    p.add_argument("--p_self_cond", type=float, default=0.5)
    p.add_argument("--use_distortion", action="store_true")
    p.add_argument("--p_distort", type=float, default=0.2)
    p.add_argument("--t_distort", type=float, default=0.5)
    p.add_argument("--sigma_distort", type=float, default=0.5)

    # checkpointing
    p.add_argument("--save_dir", type=str, default="/home/srivasv/flow_checkpoints")
    p.add_argument("--run_name", type=str, default=None, help="Name for this run (auto-generated if not provided)")
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_eval_samples", type=int, default=3)
    p.add_argument("--rk4_steps", type=int, default=100)
    p.add_argument("--save_gifs", action="store_true", help="Save trajectory GIFs during eval")

    # wandb
    p.add_argument("--wandb_project", type=str, default="water-flow")
    p.add_argument("--wandb_run", type=str, default=None)
    p.add_argument("--wandb_dir", type=str, default="/home/srivasv/wandb_logs")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_model(args, device, node_scalar_in=16):
    """Build encoder and flow model using registry-based encoder construction."""
    encoder_type = 'slae' if args.use_slae else 'gvp'
    logger.info(f"Building model with {encoder_type.upper()} encoder")

    if args.use_slae:
        logger.info(f"  SLAE dim: {args.slae_dim}")

    encoder_config = {
        'encoder_type': encoder_type,
        'hidden_s': args.hidden_s,
        'hidden_v': args.hidden_v,
        'node_scalar_in': node_scalar_in,
        'freeze_encoder': args.freeze_encoder,
        'slae_dim': args.slae_dim,
        'encoder_ckpt': args.encoder_ckpt,
    }

    encoder = build_encoder(encoder_config, device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(args.hidden_s, args.hidden_v),
        edge_scalar_dim=32,
        layers=args.flow_layers,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
    ).to(device)

    return model


def run_eval_sampling(flow_matcher, val_loader, args, epoch, device, global_step, eval_indices, run_dir):
    """Run RK4 integration on fixed eval samples and log results.

    Args:
        eval_indices: Fixed list of dataset indices to evaluate (sampled once at start)
        run_dir: Path to run directory for saving outputs
    """
    flow_matcher.model.eval()
    results = []

    for i, idx in enumerate(eval_indices):
        graph = val_loader.dataset[idx]
        if graph['water'].num_nodes == 0:
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
            pred=out['water_pred'],
            true=out['water_true'],
            threshold=1.0
        )

        final_rmsd = compute_rmsd(out['water_pred'], out['water_true'])

        results.append({
            'rmsd': final_rmsd,
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'f1': final_metrics['f1'],
            'auc_pr': final_metrics['auc_pr']
        })

        # plot final frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_frame(
            ax,
            out['protein_pos'],
            None,
            out['water_pred'],
            out['water_true'],
            title=f"Epoch {epoch} Sample {i} | RMSD={final_rmsd:.2f}Å | F1={final_metrics['f1']:.3f}"
        )

        plot_path = run_dir / "plots" / f"epoch{epoch}_sample{i}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # save GIF if requested
        if args.save_gifs and 'trajectory' in out:
            gif_path = run_dir / "gifs" / f"epoch{epoch}_sample{i}.gif"
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            create_trajectory_gif(
                trajectory=out['trajectory'],
                protein_pos=out['protein_pos'],
                water_true=out['water_true'],
                save_path=str(gif_path),
                title=f"Epoch {epoch} Sample {i}",
                fps=10,
                pdb_id=graph.pdb_id
            )

    if results:
        avg_metrics = {
            "eval/avg_rmsd": np.mean([r['rmsd'] for r in results]),
            "eval/avg_precision": np.mean([r['precision'] for r in results]),
            "eval/avg_recall": np.mean([r['recall'] for r in results]),
            "eval/avg_f1": np.mean([r['f1'] for r in results]),
            "eval/avg_auc_pr": np.mean([r['auc_pr'] for r in results]),
        }
        wandb.log(avg_metrics, step=global_step)
        return avg_metrics
    return {}


def train_epoch(flow_matcher, train_loader, optimizer, args, epoch):
    """Single training epoch."""
    flow_matcher.model.train()
    total_loss, total_rmsd = 0.0, 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        batch = batch.to(args.device)
        if batch['water'].num_nodes == 0:
            continue

        metrics = flow_matcher.training_step(
            batch, optimizer,
            grad_clip=args.grad_clip,
            use_self_conditioning=args.use_self_cond,
        )

        # print per-sample losses if batch loss exceeded 100.0
        if metrics['per_sample_info'] is not None:
            per_sample_losses = metrics['per_sample_info']['losses'].cpu()
            num_graphs = metrics['per_sample_info']['num_graphs']

            if hasattr(batch, 'pdb_id'):
                # pdb_id might be a list when batched
                pdb_ids = batch.pdb_id if isinstance(batch.pdb_id, list) else [batch.pdb_id]
                logger.info(f"\n{'='*60}")
                logger.warning(f"WARNING: Batch loss {metrics['loss']:.2f} exceeded 100.0!")
                logger.info(f"Per-sample losses ({num_graphs} samples):")
                for i in range(num_graphs):
                    pdb_id = pdb_ids[i] if i < len(pdb_ids) else 'unknown'
                    sample_loss = per_sample_losses[i].item()
                    logger.info(f"  [{i}] {pdb_id}: {sample_loss:.2f}")
                logger.info(f"{'='*60}")

        total_loss += metrics['loss']
        total_rmsd += metrics['rmsd']
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}", rmsd=f"{metrics['rmsd']:.2f}")

        global_step = (epoch - 1) * len(train_loader) + step
        wandb.log({
            "train/iter_loss": metrics['loss'],
            "train/iter_rmsd": metrics['rmsd'],
        }, step=global_step)

    n = len(train_loader)

    final_global_step = (epoch - 1) * len(train_loader) + len(train_loader) - 1
    return {'train/epoch_loss': total_loss / n, 'train/epoch_rmsd': total_rmsd / n}, final_global_step

@torch.no_grad()
def val_epoch(flow_matcher, val_loader, args, epoch):
    """Single validation epoch."""
    flow_matcher.model.eval()
    total_loss, total_rmsd = 0.0, 0.0
    
    for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
        batch = batch.to(args.device)
        if batch['water'].num_nodes == 0:
            continue
        metrics = flow_matcher.validation_step(batch)
        total_loss += metrics['loss']
        total_rmsd += metrics['rmsd']
    
    n = len(val_loader)
    return {'val/loss': total_loss / n, 'val/rmsd': total_rmsd / n}


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def check_slae_embeddings(loader, device):
    """Check if SLAE embeddings are present and compute statistics."""
    logger.info("\nChecking SLAE embeddings in dataset...")
    batch = next(iter(loader))
    batch = batch.to(device)

    if 'slae_embedding' not in batch['protein']:
        logger.warning("  WARNING: No SLAE embeddings found in data!")
        logger.info("  Please run scripts/precompute_slae_embeddings.py first")
        return False

    emb = batch['protein'].slae_embedding
    logger.info(f"  SLAE embedding shape: {emb.shape}")
    logger.info(f"  SLAE embedding stats: mean={emb.mean():.4f}, std={emb.std():.4f}, min={emb.min():.4f}, max={emb.max():.4f}")

    # check if embeddings are all zeros or constant
    if emb.std() < 1e-6:
        logger.warning("  WARNING: SLAE embeddings appear to be constant/zero!")
        return False

    return True

def save_checkpoint(model, optimizer, scheduler, epoch, path, best=False):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)
    logger.info(f"{'Best ' if best else ''}Checkpoint saved: {path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = generate_run_name(args)

    run_dir = Path(args.save_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "gifs").mkdir(exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Run name: {args.run_name}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"{'='*60}\n")

    import json
    config_file = run_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to: {config_file}\n")
    
    # dataloaders
    train_loader = get_dataloader(
        args.train_list, args.processed_dir,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        base_pdb_dir=args.base_pdb_dir,
        include_mates=args.include_mates,
        duplicate_single_sample=args.duplicate_single_sample,
    )
    val_loader = get_dataloader(
        args.val_list, args.processed_dir,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        base_pdb_dir=args.base_pdb_dir,
        include_mates=args.include_mates,
        duplicate_single_sample=1,  # Don't duplicate for validation
    )

    # sample fixed eval indices (same proteins evaluated every epoch)
    np.random.seed(42)  
    eval_indices = np.random.choice(
        len(val_loader.dataset),
        min(args.n_eval_samples, len(val_loader.dataset)),
        replace=False
    ).tolist()

    # save eval indices for reproducibility
    eval_indices_file = run_dir / "eval_indices.txt"
    with open(eval_indices_file, 'w') as f:
        f.write("# Fixed evaluation sample indices\n")
        for idx in eval_indices:
            graph = val_loader.dataset[idx]
            pdb_id = getattr(graph, 'pdb_id', 'unknown')
            f.write(f"{idx}\t{pdb_id}\n")
    logger.info(f"Fixed eval indices saved to: {eval_indices_file}")
    logger.info(f"Evaluating on {len(eval_indices)} proteins at each eval epoch\n")

    wandb.init(
        project=args.wandb_project,
        dir=args.wandb_dir,
        name=args.wandb_run,
        config=vars(args),
    )

    # detect input dimension from data (16 for element_onehot, 37 for atom37 format)
    sample_data = train_loader.dataset[0]
    node_scalar_in = sample_data['protein'].x.shape[-1]
    logger.info(f"Detected protein input dimension: {node_scalar_in}")

    model = build_model(args, device, node_scalar_in=node_scalar_in)
    trainable_params, total_params = count_parameters(model)
    logger.info(f"\nModel statistics:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")

    # check SLAE embeddings if using SLAE mode
    if args.use_slae:
        embeddings_ok = check_slae_embeddings(train_loader, device)
        if not embeddings_ok:
            logger.error("\nERROR: SLAE embeddings are missing or invalid!")
            logger.info("Please run: python scripts/precompute_slae_embeddings.py \\")
            logger.info(f"  --train_list {args.train_list} \\")
            logger.info(f"  --val_list {args.val_list} \\")
            logger.info(f"  --processed_dir {args.processed_dir}")
            return

        # test forward pass to check if adapter is working
        logger.info("\nTesting forward pass with SLAE...")
        model.eval()
        batch = next(iter(train_loader)).to(device)
        with torch.no_grad():
            # determine number of graphs in batch
            num_graphs = int(batch['protein'].batch.max().item()) + 1
            t = torch.zeros(num_graphs, device=device)
            try:
                v_out = model(batch, t)
                logger.info(f"  Forward pass successful! Output shape: {v_out.shape}")
                logger.info(f"  Output stats: mean={v_out.mean():.4f}, std={v_out.std():.4f}")
                if v_out.std() < 1e-6:
                    logger.warning("  WARNING: Model output is constant! This indicates a problem.")
            except Exception as e:
                logger.error(f"  ERROR in forward pass: {e}")
                return
        model.train()
    
    # flow matcher
    flow_matcher = FlowMatcher(
        model=model,
        p_self_cond=args.p_self_cond,
        use_distortion=args.use_distortion,
        p_distort=args.p_distort,
        t_distort=args.t_distort,
        sigma_distort=args.sigma_distort,
    )
    
    # optimizer & scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        
        # train
        train_metrics, global_step = train_epoch(flow_matcher, train_loader, optimizer, args, epoch)
        wandb.log(train_metrics, step=global_step)

        # val
        val_metrics = val_epoch(flow_matcher, val_loader, args, epoch)
        wandb.log(val_metrics, step=global_step)
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=global_step)

        scheduler.step()

        logger.info(f"Epoch {epoch}: train_loss={train_metrics['train/epoch_loss']:.4f}, "
              f"val_loss={val_metrics['val/loss']:.4f}, val_rmsd={val_metrics['val/rmsd']:.2f}")

        # save best
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            save_checkpoint(model, optimizer, scheduler, epoch,
                            run_dir / "checkpoints" / "best.pt", best=True)

        # periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            run_dir / "checkpoints" / f"epoch_{epoch}.pt")

        # eval sampling
        if epoch % args.eval_every == 0:
            eval_metrics = run_eval_sampling(
                flow_matcher, val_loader, args, epoch, device, global_step, eval_indices, run_dir
            )
            if eval_metrics:
                logger.info(f"  Eval: RMSD={eval_metrics['eval/avg_rmsd']:.2f}Å, "
                      f"Precision={eval_metrics['eval/avg_precision']:.2%}, "
                      f"Recall={eval_metrics['eval/avg_recall']:.2%}, "
                      f"F1={eval_metrics['eval/avg_f1']:.3f}, "
                      f"AUC-PR={eval_metrics['eval/avg_auc_pr']:.3f}")
    
    wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
