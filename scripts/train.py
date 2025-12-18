# train.py

import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from src.dataset import get_dataloader
from src.encoder import ProteinGVPEncoder, load_encoder_from_checkpoint
from src.flow import FlowWaterGVP, FlowMatcher
from src.utils import plot_3d_frame


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument("--processed_dir", type=str, default="/home/srivasv/flow_cache/")
    p.add_argument("--base_pdb_dir", type=str, default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data")

    # model
    p.add_argument("--encoder_ckpt", type=str, default=None)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--hidden_s", type=int, default=256)
    p.add_argument("--hidden_v", type=int, default=64)
    p.add_argument("--flow_layers", type=int, default=4)
    p.add_argument("--k_pw", type=int, default=12)
    p.add_argument("--k_ww", type=int, default=12)

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
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
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_eval_samples", type=int, default=3)
    p.add_argument("--rk4_steps", type=int, default=100)

    # wandb
    p.add_argument("--wandb_project", type=str, default="water-flow")
    p.add_argument("--wandb_run", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def build_model(args, device):
    """Build encoder and flow model."""
    if args.encoder_ckpt:
        encoder, enc_args = load_encoder_from_checkpoint(
            args.encoder_ckpt, device=device, freeze=args.freeze_encoder
        )
    else:
        encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(args.hidden_s, args.hidden_v),
            edge_scalar_in=16,
            pooled_dim=128,
        ).to(device)
    
    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(args.hidden_s, args.hidden_v),
        edge_scalar_dim=32,
        layers=args.flow_layers,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    
    return model


def run_eval_sampling(flow_matcher, val_loader, args, epoch, device):
    """Run RK4 integration on a few samples and log results."""
    flow_matcher.model.eval()
    results = []
    
    sample_indices = np.random.choice(len(val_loader.dataset), min(args.n_eval_samples, len(val_loader.dataset)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        graph = val_loader.dataset[idx]
        if graph['water'].num_nodes == 0:
            continue
            
        out = flow_matcher.rk4_integrate(
            graph,
            num_steps=args.rk4_steps,
            use_sc=args.use_self_cond,
            device=device,
            return_trajectory=True,
        )
        
        final_rmsd = out['rmsd'][-1]
        final_cov = out['coverage'][-1]
        final_prec = out['precision'][-1]
        
        results.append({'rmsd': final_rmsd, 'coverage': final_cov, 'precision': final_prec})
        
        # plot final frame
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_frame(
            ax,
            out['protein_pos'],
            None,
            out['water_pred'],
            out['water_true'],
            title=f"Epoch {epoch} Sample {i} | RMSD={final_rmsd:.2f}Å"
        )
        
        plot_path = Path(args.save_dir) / "plots" / f"epoch{epoch}_sample{i}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        wandb.log({f"eval/sample_{i}": wandb.Image(str(plot_path))}, step=epoch)
    
    if results:
        avg_metrics = {
            "eval/avg_rmsd": np.mean([r['rmsd'] for r in results]),
            "eval/avg_coverage": np.mean([r['coverage'] for r in results]),
            "eval/avg_precision": np.mean([r['precision'] for r in results]),
        }
        wandb.log(avg_metrics, step=epoch)
        return avg_metrics
    return {}


def train_epoch(flow_matcher, train_loader, optimizer, args, epoch):
    """Single training epoch."""
    flow_matcher.model.train()
    total_loss, total_rmsd = 0.0, 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        batch = batch.to(args.device)
        if batch['water'].num_nodes == 0:
            continue
            
        metrics = flow_matcher.training_step(
            batch, optimizer, 
            grad_clip=args.grad_clip,
            use_self_conditioning=args.use_self_cond,
        )
        
        total_loss += metrics['loss']
        total_rmsd += metrics['rmsd']
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}", rmsd=f"{metrics['rmsd']:.2f}")
    
    n = len(train_loader)
    return {'train/loss': total_loss / n, 'train/rmsd': total_rmsd / n}


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


def save_checkpoint(model, optimizer, scheduler, epoch, path, best=False):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)
    print(f"{'Best ' if best else ''}Checkpoint saved: {path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # wandb init
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=vars(args),
    )
    
    # dataloaders
    train_loader = get_dataloader(
        args.train_list, args.processed_dir,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        base_pdb_dir=args.base_pdb_dir,
    )
    val_loader = get_dataloader(
        args.val_list, args.processed_dir,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        base_pdb_dir=args.base_pdb_dir,
    )
    
    # model
    model = build_model(args, device)
    wandb.watch(model, log_freq=100)
    
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
        train_metrics = train_epoch(flow_matcher, train_loader, optimizer, args, epoch)
        wandb.log(train_metrics, step=epoch)
        
        # val
        val_metrics = val_epoch(flow_matcher, val_loader, args, epoch)
        wandb.log(val_metrics, step=epoch)
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: train_loss={train_metrics['train/loss']:.4f}, "
              f"val_loss={val_metrics['val/loss']:.4f}, val_rmsd={val_metrics['val/rmsd']:.2f}")
        
        # save best
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            save_checkpoint(model, optimizer, scheduler, epoch,
                            Path(args.save_dir) / "best.pt", best=True)
        
        # periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            Path(args.save_dir) / f"epoch_{epoch}.pt")
        
        # eval sampling
        if epoch % args.eval_every == 0:
            eval_metrics = run_eval_sampling(flow_matcher, val_loader, args, epoch, device)
            if eval_metrics:
                print(f"  Eval: RMSD={eval_metrics['eval/avg_rmsd']:.2f}, "
                      f"Cov={eval_metrics['eval/avg_coverage']:.2%}, "
                      f"Prec={eval_metrics['eval/avg_precision']:.2%}")
    
    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()