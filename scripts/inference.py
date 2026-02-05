# inference.py

"""
Inference script for WaterFlow model.

Takes a text file of PDB paths, runs trajectory integration (euler/rk4),
and outputs plots, gifs, and metrics for each PDB.

Usage:
    python scripts/inference.py \
        --run_dir /path/to/run_directory \
        --pdb_list /path/to/pdb_list.txt \
        --output_dir /path/to/output \
        --method rk4 \
        --num_steps 100 \
        --save_gifs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import ProteinWaterDataset
from src.encoder_base import build_encoder
from src.flow import FlowWaterGVP, FlowMatcher
from src.utils import (
    plot_3d_frame,
    create_trajectory_gif,
    compute_placement_metrics,
    compute_rmsd,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run WaterFlow inference on PDB files")

    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (contains config.json and checkpoints/)",
    )
    p.add_argument(
        "--pdb_list",
        type=str,
        required=True,
        help="Text file with PDB entries (one per line, format: pdb_id_final or pdb_id_final_chainID)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference outputs",
    )

    # data arguments
    p.add_argument(
        "--processed_dir",
        type=str,
        default="/home/srivasv/flow_cache/",
        help="Directory for cached preprocessed .pt files",
    )
    p.add_argument(
        "--base_pdb_dir",
        type=str,
        default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data",
        help="Base directory containing PDB subdirectories for dataset creation",
    )
    p.add_argument(
        "--include_mates",
        action="store_true",
        help="Include symmetry mate atoms as protein nodes",
    )

    # checkpoint arguments
    p.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename within run_dir/checkpoints/ (default: best.pt)",
    )

    # integration arguments
    p.add_argument(
        "--method",
        type=str,
        default="rk4",
        choices=["euler", "rk4"],
        help="Integration method (default: rk4)",
    )
    p.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of integration steps (default: 100)",
    )
    p.add_argument(
        "--use_sc",
        action="store_true",
        help="Use self-conditioning during integration",
    )

    p.add_argument(
        "--save_gifs",
        action="store_true",
        help="Save trajectory GIFs (slower but useful for visualization)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Distance threshold in Angstroms for precision/recall (default: 1.0)",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)",
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of proteins to process in parallel (default: 8)",
    )

    p.add_argument(
        "--water_ratio",
        type=float,
        default=None,
        help="Sample num_residues * water_ratio waters instead of using ground truth count. "
             "E.g., --water_ratio 0.5 samples 50 waters for a 100-residue protein.",
    )

    args = p.parse_args()

    return args


def load_config(run_dir: Path) -> dict:
    """Load training config from run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def build_model_from_config(config: dict, device: torch.device) -> nn.Module:
    """Build model architecture from config using registry-based encoder construction."""
    use_slae = config.get("use_slae", False)
    hidden_s = config.get("hidden_s", 256)
    hidden_v = config.get("hidden_v", 64)
    flow_layers = config.get("flow_layers", 5)
    k_pw = config.get("k_pw", 24)
    k_ww = config.get("k_ww", 24)
    freeze_encoder = config.get("freeze_encoder", False)

    # Build encoder config for registry
    encoder_config = {
        'encoder_type': 'slae' if use_slae else 'gvp',
        'hidden_s': hidden_s,
        'hidden_v': hidden_v,
        'node_scalar_in': config.get("node_scalar_in", 16),
        'freeze_encoder': freeze_encoder,
        'slae_dim': config.get("slae_dim", 128),
        'encoder_ckpt': config.get("encoder_ckpt"),
    }

    encoder = build_encoder(encoder_config, device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(hidden_s, hidden_v),
        edge_scalar_dim=32,
        layers=flow_layers,
        k_pw=k_pw,
        k_ww=k_ww,
    ).to(device)

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    """Load model weights from checkpoint."""
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
    water_ratio: float = None,
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
        water_preds = flow_matcher.euler_integrate(
            graphs,
            num_steps=num_steps,
            use_sc=use_sc,
            device=device,
            water_ratio=water_ratio,
        )
        # build result dicts similar to rk4
        results = []
        for graph, water_pred in zip(graphs, water_preds):
            results.append({
                "protein_pos": graph["protein"].pos.numpy(),
                "water_true": graph["water"].pos.numpy(),
                "water_pred": water_pred,
                "trajectory": None,
                "pdb_id": getattr(graph, 'pdb_id', None),
            })

    return results


def save_plot(
    result: dict,
    pdb_id: str,
    output_path: Path,
    metrics: dict,
):
    """Save 3D visualization plot."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    title = (
        f"{pdb_id} | RMSD={metrics['rmsd']:.2f}Å | "
        f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} F1={metrics['f1']:.3f}"
    )

    plot_3d_frame(
        ax,
        result["protein_pos"],
        None,  # no separate mate positions
        result["water_pred"],
        result["water_true"],
        title=title,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    # setup paths
    run_dir = Path(args.run_dir)
    output_root = Path(args.output_dir)
    output_dir = output_root / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    if args.save_gifs:
        (output_dir / "gifs").mkdir(exist_ok=True)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load config and build model
    print(f"\nLoading model from: {run_dir}")
    config = load_config(run_dir)
    model = build_model_from_config(config, device)

    # load checkpoint
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    epoch = load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")

    # Create FlowMatcher
    flow_matcher = FlowMatcher(
        model=model,
        p_self_cond=config.get("p_self_cond", 0.5),
    )

    # Load dataset
    print(f"\nLoading PDBs from: {args.pdb_list}")

    # Determine include_mates from args or config
    include_mates = args.include_mates or config.get("include_mates", False)

    dataset = ProteinWaterDataset(
        pdb_list_file=args.pdb_list,
        processed_dir=args.processed_dir,
        base_pdb_dir=args.base_pdb_dir,
        include_mates=include_mates,
        preprocess=True,
    )

    print(f"Found {len(dataset)} PDB entries")

    # run inference
    print(f"\nRunning inference with method={args.method}, steps={args.num_steps}")
    print(f"Self-conditioning: {args.use_sc}")
    print(f"Threshold for metrics: {args.threshold}Å")
    print(f"Batch size: {args.batch_size}")
    if args.water_ratio is not None:
        print(f"Water ratio: {args.water_ratio} (sampling num_residues × {args.water_ratio} waters)")
    else:
        print("Water ratio: None (using ground truth water count)")
    print("-" * 60)

    all_metrics = []

    # collect valid graphs (those with waters for ground truth comparison)
    valid_graphs = []
    skipped_pdbs = []
    for idx in range(len(dataset)):
        graph = dataset[idx]
        if graph["water"].num_nodes == 0:
            skipped_pdbs.append(graph.pdb_id)
        else:
            valid_graphs.append(graph)

    if skipped_pdbs:
        print(f"Skipping {len(skipped_pdbs)} PDBs with no water molecules")

    # process in batches
    num_batches = (len(valid_graphs) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(valid_graphs))
        batch_graphs = valid_graphs[start_idx:end_idx]

        # run batched inference
        batch_results = run_inference_batch(
            flow_matcher,
            batch_graphs,
            method=args.method,
            num_steps=args.num_steps,
            use_sc=args.use_sc,
            device=args.device,
            water_ratio=args.water_ratio,
        )

        # process each result in the batch
        for result in batch_results:
            pdb_id = result.get("pdb_id", f"unknown_{len(all_metrics)}")
            water_true = result["water_true"]
            water_pred = result["water_pred"]

            # compute metrics
            metrics = compute_placement_metrics(
                pred=water_pred,
                true=water_true,
                threshold=args.threshold,
            )
            metrics["rmsd"] = compute_rmsd(water_pred, water_true)
            metrics["pdb_id"] = pdb_id
            metrics["n_waters_true"] = water_true.shape[0]
            metrics["n_waters_pred"] = water_pred.shape[0]

            all_metrics.append(metrics)

            plot_path = output_dir / "plots" / f"{pdb_id}.png"
            save_plot(result, pdb_id, plot_path, metrics)

            # save GIF if requested and trajectory available
            if args.save_gifs and result.get("trajectory") is not None:
                gif_path = output_dir / "gifs" / f"{pdb_id}.gif"
                create_trajectory_gif(
                    trajectory=result["trajectory"],
                    protein_pos=result["protein_pos"],
                    water_true=water_true,
                    save_path=str(gif_path),
                    title="",
                    fps=10,
                    pdb_id=pdb_id,
                )

            # print per-sample metrics
            tqdm.write(
                f"  {pdb_id}: RMSD={metrics['rmsd']:.2f}Å | "
                f"P={metrics['precision']:.2%} R={metrics['recall']:.2%} "
                f"F1={metrics['f1']:.3f} AUC-PR={metrics['auc_pr']:.3f}"
            )

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
            "avg_n_waters_true": float(np.mean([m["n_waters_true"] for m in all_metrics])),
            "avg_n_waters_pred": float(np.mean([m["n_waters_pred"] for m in all_metrics])),
        }

        print("\n" + "=" * 60)
        print("SUMMARY METRICS")
        print("=" * 60)
        print(f"  Samples processed: {summary['n_samples']}")
        print(f"  Avg waters (true):  {summary['avg_n_waters_true']:.1f}")
        print(f"  Avg waters (pred):  {summary['avg_n_waters_pred']:.1f}")
        print(f"  Avg RMSD:      {summary['avg_rmsd']:.3f} ± {summary['std_rmsd']:.3f} Å")
        print(f"  Avg Precision: {summary['avg_precision']:.3%}")
        print(f"  Avg Recall:    {summary['avg_recall']:.3%}")
        print(f"  Avg F1:        {summary['avg_f1']:.4f}")
        print(f"  Avg AUC-PR:    {summary['avg_auc_pr']:.4f}")
        print("=" * 60)

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "per_sample": all_metrics,
                    "config": {
                        "run_dir": str(run_dir),
                        "checkpoint": args.checkpoint,
                        "method": args.method,
                        "num_steps": args.num_steps,
                        "use_sc": args.use_sc,
                        "threshold": args.threshold,
                        "include_mates": include_mates,
                        "water_ratio": args.water_ratio,
                    },
                },
                f,
                indent=2,
            )
        print(f"\nMetrics saved to: {metrics_path}")

    else:
        print("\nNo valid samples processed.")

    print(f"Plots saved to: {output_dir / 'plots'}")
    if args.save_gifs:
        print(f"GIFs saved to: {output_dir / 'gifs'}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
