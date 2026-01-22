"""
Visualize graph construction for WaterFlow.

Creates publication-quality figures showing:
1. Protein radius graph with residue coloring and mate highlighting
2. Water KNN edges with sampled edges for clarity
3. 2D schematic explaining the graph construction concept

Usage:
    python scripts/visualize_graph_construction.py \
        --pdb_list splits/single_train.txt \
        --output_dir figures/graph_construction \
        --neighborhood_radius 10.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

from src.dataset import ProteinWaterDataset
from src.flow import build_knn_edges

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def parse_args():
    p = argparse.ArgumentParser(description="Visualize graph construction")

    p.add_argument("--pdb_list", type=str, default="splits/single_train.txt")
    p.add_argument("--processed_dir", type=str, default="/home/srivasv/flow_cache/")
    p.add_argument("--base_pdb_dir", type=str, default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data")
    p.add_argument("--output_dir", type=str, default="figures/graph_construction")
    p.add_argument("--focal_atom", type=int, default=None)
    p.add_argument("--neighborhood_radius", type=float, default=10.0)
    p.add_argument("--mate_distance_cutoff", type=float, default=6.0,
                   help="Only show mates within this distance of ASU atoms")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--k_pw", type=int, default=12)
    p.add_argument("--k_ww", type=int, default=8)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--edge_sample_frac", type=float, default=0.15,
                   help="Fraction of background edges to sample for display")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def get_residue_colors(residue_indices, cmap_name='tab20'):
    """Assign colors to residues using a colormap."""
    unique_res = np.unique(residue_indices)
    cmap = cm.get_cmap(cmap_name)
    colors = {r: cmap(i % 20) for i, r in enumerate(unique_res)}
    return np.array([colors[r] for r in residue_indices])


def find_focal_atom_near_waters(protein_pos, water_pos, num_asu_atoms):
    """Find a good focal atom near waters with many neighbors."""
    if water_pos.numel() == 0:
        asu_pos = protein_pos[:num_asu_atoms]
        center = asu_pos.mean(dim=0)
        dists_to_center = torch.norm(asu_pos - center, dim=1)
        return torch.argmin(dists_to_center).item()

    asu_pos = protein_pos[:num_asu_atoms]
    dists_to_water = torch.cdist(asu_pos, water_pos)
    min_dists_to_water = dists_to_water.min(dim=1).values
    near_water_mask = min_dists_to_water < 5.0

    if near_water_mask.sum() > 0:
        near_water_indices = torch.where(near_water_mask)[0]
        neighbor_counts = []
        for idx in near_water_indices:
            dist_to_others = torch.norm(protein_pos - protein_pos[idx], dim=1)
            n_neighbors = (dist_to_others < 8.0).sum().item() - 1
            neighbor_counts.append(n_neighbors)
        best_local_idx = torch.tensor(neighbor_counts).argmax().item()
        return near_water_indices[best_local_idx].item()

    return torch.argmin(min_dists_to_water).item()


def filter_mates_by_distance(protein_pos, num_asu_atoms, mate_cutoff):
    """
    Return mask for mate atoms that are within mate_cutoff of any ASU atom.
    """
    if num_asu_atoms >= protein_pos.size(0):
        return torch.ones(protein_pos.size(0), dtype=torch.bool)

    asu_pos = protein_pos[:num_asu_atoms]
    mate_pos = protein_pos[num_asu_atoms:]

    # Compute distances from each mate to closest ASU atom
    dists = torch.cdist(mate_pos, asu_pos)
    min_dists = dists.min(dim=1).values

    # Mask for close mates
    close_mates = min_dists < mate_cutoff

    # Build full mask
    full_mask = torch.ones(protein_pos.size(0), dtype=torch.bool)
    full_mask[num_asu_atoms:] = close_mates

    return full_mask


def sample_edges(edge_list, frac, rng, important_indices=None):
    """
    Sample a fraction of edges, always keeping edges involving important_indices.
    """
    if important_indices is None:
        important_indices = set()

    important_edges = []
    other_edges = []

    for edge in edge_list:
        if edge[0] in important_indices or edge[1] in important_indices:
            important_edges.append(edge)
        else:
            other_edges.append(edge)

    # Sample from other edges
    n_sample = max(1, int(len(other_edges) * frac))
    if len(other_edges) > n_sample:
        sampled_indices = rng.choice(len(other_edges), n_sample, replace=False)
        other_edges = [other_edges[i] for i in sampled_indices]

    return important_edges + other_edges


def plot_protein_radius_graph(
    protein_pos: torch.Tensor,
    edge_index: torch.Tensor,
    residue_indices: torch.Tensor,
    num_asu_atoms: int,
    focal_atom: int,
    neighborhood_radius: float,
    mate_cutoff: float,
    cutoff: float,
    edge_sample_frac: float,
    save_path: str,
    pdb_id: str = "",
    rng: np.random.Generator = None,
):
    """Plot protein radius graph with residue coloring."""
    if rng is None:
        rng = np.random.default_rng(42)

    pos = protein_pos.numpy()
    res_idx = residue_indices.numpy()
    focal_pos = pos[focal_atom]

    # Filter atoms in neighborhood
    dists_to_focal = np.linalg.norm(pos - focal_pos, axis=1)
    in_neighborhood = dists_to_focal < neighborhood_radius

    # Also filter mates by distance to ASU
    mate_mask = filter_mates_by_distance(protein_pos, num_asu_atoms, mate_cutoff).numpy()
    in_neighborhood = in_neighborhood & mate_mask

    neighborhood_indices = np.where(in_neighborhood)[0]
    global_to_local = {g: l for l, g in enumerate(neighborhood_indices)}

    # Filter edges
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    all_edges = []
    edge_involves_mate = []

    for s, d in zip(src, dst):
        if s in global_to_local and d in global_to_local:
            local_s, local_d = global_to_local[s], global_to_local[d]
            s_is_mate = s >= num_asu_atoms
            d_is_mate = d >= num_asu_atoms
            all_edges.append((local_s, local_d))
            edge_involves_mate.append(s_is_mate or d_is_mate)

    local_pos = pos[neighborhood_indices]
    local_res = res_idx[neighborhood_indices]
    local_focal = global_to_local[focal_atom]
    is_asu = neighborhood_indices < num_asu_atoms

    # Find focal neighbors
    focal_neighbors = set()
    focal_edges = []
    for i, (s, d) in enumerate(all_edges):
        if s == local_focal:
            focal_neighbors.add(d)
            focal_edges.append(i)
        if d == local_focal:
            focal_neighbors.add(s)
            focal_edges.append(i)
    focal_edges = set(focal_edges)

    # Sample non-focal edges
    non_focal_edges = [(i, e, edge_involves_mate[i]) for i, e in enumerate(all_edges) if i not in focal_edges]
    n_sample = max(5, int(len(non_focal_edges) * edge_sample_frac))
    if len(non_focal_edges) > n_sample:
        sampled_idx = rng.choice(len(non_focal_edges), n_sample, replace=False)
        non_focal_edges = [non_focal_edges[i] for i in sampled_idx]

    # Create figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Colors
    res_colors = get_residue_colors(local_res)
    mate_color = '#d35400'
    edge_asu_color = '#7f8c8d'
    edge_mate_color = '#e67e22'

    # Plot sampled non-focal edges (very subtle)
    for _, (s, d), involves_mate in non_focal_edges:
        xs = [local_pos[s, 0], local_pos[d, 0]]
        ys = [local_pos[s, 1], local_pos[d, 1]]
        zs = [local_pos[s, 2], local_pos[d, 2]]
        c = edge_mate_color if involves_mate else edge_asu_color
        ax.plot(xs, ys, zs, c=c, alpha=0.08, linewidth=0.3, zorder=1)

    # Plot focal edges prominently
    for i in focal_edges:
        s, d = all_edges[i]
        xs = [local_pos[s, 0], local_pos[d, 0]]
        ys = [local_pos[s, 1], local_pos[d, 1]]
        zs = [local_pos[s, 2], local_pos[d, 2]]
        c = edge_mate_color if edge_involves_mate[i] else '#2c3e50'
        ax.plot(xs, ys, zs, c=c, alpha=0.9, linewidth=1.8, zorder=2)

    # Plot background ASU atoms with residue colors
    bg_mask = is_asu & (np.arange(len(local_pos)) != local_focal) & \
              np.array([i not in focal_neighbors for i in range(len(local_pos))])
    if bg_mask.sum() > 0:
        ax.scatter(
            local_pos[bg_mask, 0], local_pos[bg_mask, 1], local_pos[bg_mask, 2],
            c=res_colors[bg_mask], alpha=0.5, s=25, edgecolors='white', linewidths=0.3, zorder=3
        )

    # Plot background mate atoms
    bg_mate_mask = ~is_asu & np.array([i not in focal_neighbors for i in range(len(local_pos))])
    if bg_mate_mask.sum() > 0:
        ax.scatter(
            local_pos[bg_mate_mask, 0], local_pos[bg_mate_mask, 1], local_pos[bg_mate_mask, 2],
            c=mate_color, alpha=0.6, s=30, marker='s', edgecolors='white', linewidths=0.3, zorder=3
        )

    # Plot focal neighbors - ASU (with residue colors)
    neighbor_asu = [i for i in focal_neighbors if is_asu[i]]
    if neighbor_asu:
        ax.scatter(
            local_pos[neighbor_asu, 0], local_pos[neighbor_asu, 1], local_pos[neighbor_asu, 2],
            c=res_colors[neighbor_asu], alpha=1.0, s=100,
            edgecolors='#2c3e50', linewidths=1.5, zorder=5
        )

    # Plot focal neighbors - mates
    neighbor_mate = [i for i in focal_neighbors if not is_asu[i]]
    if neighbor_mate:
        ax.scatter(
            local_pos[neighbor_mate, 0], local_pos[neighbor_mate, 1], local_pos[neighbor_mate, 2],
            c=mate_color, alpha=1.0, s=100, marker='s',
            edgecolors='#2c3e50', linewidths=1.5, zorder=5
        )

    # Plot focal atom
    ax.scatter(
        [local_pos[local_focal, 0]], [local_pos[local_focal, 1]], [local_pos[local_focal, 2]],
        c='#e74c3c', alpha=1.0, s=200, marker='*',
        edgecolors='black', linewidths=1.5, zorder=10
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='Protein atoms (by residue)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=mate_color,
               markersize=8, label='Symmetry mates'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c',
               markersize=12, label='Focal atom'),
        Line2D([0], [0], color='#2c3e50', linewidth=2, label='Edges (ASU)'),
        Line2D([0], [0], color=edge_mate_color, linewidth=2, label='Edges (to mates)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')

    n_asu_neighbors = len(neighbor_asu)
    n_mate_neighbors = len(neighbor_mate)
    ax.set_title(f'Radius Graph (r={cutoff}Å)\n{n_asu_neighbors} protein + {n_mate_neighbors} mate neighbors',
                 fontweight='bold')

    # Set view and limits
    ax.view_init(elev=20, azim=45)
    max_range = neighborhood_radius * 0.9
    ax.set_xlim(focal_pos[0] - max_range, focal_pos[0] + max_range)
    ax.set_ylim(focal_pos[1] - max_range, focal_pos[1] + max_range)
    ax.set_zlim(focal_pos[2] - max_range, focal_pos[2] + max_range)

    # Remove grid for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved protein radius graph to: {save_path}")


def plot_water_knn_edges(
    protein_pos: torch.Tensor,
    water_pos: torch.Tensor,
    residue_indices: torch.Tensor,
    num_asu_atoms: int,
    mate_cutoff: float,
    k_pw: int,
    k_ww: int,
    edge_sample_frac: float,
    save_path: str,
    pdb_id: str = "",
    neighborhood_center: np.ndarray = None,
    neighborhood_radius: float = 15.0,
    rng: np.random.Generator = None,
):
    """Plot water KNN edges with sampled edges for clarity."""
    if rng is None:
        rng = np.random.default_rng(42)

    if water_pos.numel() == 0:
        print("No waters to visualize")
        return

    # Build KNN edges
    ei_pw = build_knn_edges(protein_pos, water_pos, k=k_pw)
    ei_wp = build_knn_edges(water_pos, protein_pos, k=k_pw)
    ei_wp_reversed = ei_wp.flip(0)
    ei_pw_union = torch.cat([ei_pw, ei_wp_reversed], dim=1).unique(dim=1)
    ei_ww = build_knn_edges(water_pos, water_pos, k=k_ww)

    pos_p = protein_pos.numpy()
    pos_w = water_pos.numpy()
    res_idx = residue_indices.numpy()

    if neighborhood_center is None:
        neighborhood_center = pos_w.mean(axis=0)

    # Filter waters in neighborhood
    dists_w = np.linalg.norm(pos_w - neighborhood_center, axis=1)
    in_neighborhood_w = dists_w < neighborhood_radius
    water_indices = np.where(in_neighborhood_w)[0]

    if len(water_indices) == 0:
        water_indices = np.arange(min(15, len(pos_w)))

    # Filter proteins - only those close to displayed waters and within mate cutoff
    mate_mask = filter_mates_by_distance(protein_pos, num_asu_atoms, mate_cutoff).numpy()
    dists_p = np.linalg.norm(pos_p - neighborhood_center, axis=1)
    in_neighborhood_p = (dists_p < neighborhood_radius * 1.1) & mate_mask
    protein_indices = np.where(in_neighborhood_p)[0]

    global_to_local_p = {g: l for l, g in enumerate(protein_indices)}
    global_to_local_w = {g: l for l, g in enumerate(water_indices)}

    local_pos_p = pos_p[protein_indices]
    local_pos_w = pos_w[water_indices]
    local_res = res_idx[protein_indices]
    is_asu_p = protein_indices < num_asu_atoms

    # Filter and collect p->w edges
    pw_src, pw_dst = ei_pw_union[0].numpy(), ei_pw_union[1].numpy()
    pw_edges = []
    for s, d in zip(pw_src, pw_dst):
        if s in global_to_local_p and d in global_to_local_w:
            pw_edges.append((global_to_local_p[s], global_to_local_w[d], s < num_asu_atoms))

    # Filter w->w edges
    ww_src, ww_dst = ei_ww[0].numpy(), ei_ww[1].numpy()
    ww_edges = []
    for s, d in zip(ww_src, ww_dst):
        if s in global_to_local_w and d in global_to_local_w:
            ww_edges.append((global_to_local_w[s], global_to_local_w[d]))

    # Sample edges
    n_pw_sample = max(20, int(len(pw_edges) * edge_sample_frac * 2))
    if len(pw_edges) > n_pw_sample:
        sampled_idx = rng.choice(len(pw_edges), n_pw_sample, replace=False)
        pw_edges = [pw_edges[i] for i in sampled_idx]

    n_ww_sample = max(10, int(len(ww_edges) * edge_sample_frac * 3))
    if len(ww_edges) > n_ww_sample:
        sampled_idx = rng.choice(len(ww_edges), n_ww_sample, replace=False)
        ww_edges = [ww_edges[i] for i in sampled_idx]

    # Create figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Colors
    res_colors = get_residue_colors(local_res) if len(local_res) > 0 else []
    mate_color = '#d35400'
    water_color = '#e74c3c'
    pw_edge_color = '#3498db'
    ww_edge_color = '#9b59b6'

    # Plot P-W edges
    for s, d, is_asu in pw_edges:
        xs = [local_pos_p[s, 0], local_pos_w[d, 0]]
        ys = [local_pos_p[s, 1], local_pos_w[d, 1]]
        zs = [local_pos_p[s, 2], local_pos_w[d, 2]]
        c = pw_edge_color if is_asu else mate_color
        ax.plot(xs, ys, zs, c=c, alpha=0.5, linewidth=0.8, zorder=1)

    # Plot W-W edges
    for s, d in ww_edges:
        xs = [local_pos_w[s, 0], local_pos_w[d, 0]]
        ys = [local_pos_w[s, 1], local_pos_w[d, 1]]
        zs = [local_pos_w[s, 2], local_pos_w[d, 2]]
        ax.plot(xs, ys, zs, c=ww_edge_color, alpha=0.8, linewidth=1.5, zorder=2)

    # Plot protein atoms
    if len(local_pos_p) > 0:
        asu_mask = is_asu_p
        if asu_mask.sum() > 0:
            ax.scatter(
                local_pos_p[asu_mask, 0], local_pos_p[asu_mask, 1], local_pos_p[asu_mask, 2],
                c=res_colors[asu_mask] if len(res_colors) > 0 else '#3498db',
                alpha=0.5, s=20, edgecolors='white', linewidths=0.2, zorder=3
            )
        mate_mask = ~asu_mask
        if mate_mask.sum() > 0:
            ax.scatter(
                local_pos_p[mate_mask, 0], local_pos_p[mate_mask, 1], local_pos_p[mate_mask, 2],
                c=mate_color, alpha=0.6, s=25, marker='s', edgecolors='white', linewidths=0.2, zorder=3
            )

    # Plot water molecules
    ax.scatter(
        local_pos_w[:, 0], local_pos_w[:, 1], local_pos_w[:, 2],
        c=water_color, alpha=0.95, s=80,
        edgecolors='#c0392b', linewidths=1.0, marker='o', zorder=5
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='Protein (by residue)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=mate_color,
               markersize=8, label='Symmetry mates'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=water_color,
               markeredgecolor='#c0392b', markersize=10, label='Waters'),
        Line2D([0], [0], color=pw_edge_color, linewidth=2, label=f'Protein-Water (k={k_pw})'),
        Line2D([0], [0], color=ww_edge_color, linewidth=2, label=f'Water-Water (k={k_ww})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'KNN Graph Construction\n{len(local_pos_w)} waters shown', fontweight='bold')

    ax.view_init(elev=20, azim=45)

    all_pos = np.vstack([local_pos_p, local_pos_w]) if len(local_pos_p) > 0 else local_pos_w
    center = all_pos.mean(axis=0)
    max_range = max(np.ptp(all_pos, axis=0)) / 2 + 2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved water KNN edges to: {save_path}")


def plot_2d_schematic(save_path: str, cutoff: float = 8.0, k_pw: int = 12, k_ww: int = 8):
    """
    Create a clean 2D schematic showing the graph construction concept.
    Two panels: (A) Protein Radius Graph, (B) Water KNN (both P-W and W-W)
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    rng = np.random.default_rng(123)

    # Panel A: Radius graph on protein
    ax = axes[0]

    # Generate clustered protein atoms (residues) - closer together for inter-residue edges
    # Use fixed seed for reproducibility
    rng_a = np.random.default_rng(42)

    residue_centers = np.array([[0.25, 0.48], [0.40, 0.62], [0.55, 0.48], [0.40, 0.34]])
    protein_pos = []
    protein_res = []
    for i, center in enumerate(residue_centers):
        n_atoms = 7  # More atoms per residue
        atoms = center + rng_a.normal(0, 0.04, (n_atoms, 2))
        protein_pos.extend(atoms)
        protein_res.extend([i] * n_atoms)
    protein_pos = np.array(protein_pos)
    protein_res = np.array(protein_res)

    # Add symmetry mate cluster - close to purple residue (residue 2)
    mate_positions = np.array([
        [0.68, 0.55], [0.72, 0.48], [0.75, 0.58], [0.70, 0.42], [0.78, 0.50]
    ])
    mate_pos = mate_positions + rng_a.normal(0, 0.015, mate_positions.shape)

    # Colors for residues
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#1abc9c']

    radius = 0.18

    # Focal atoms: 2 from protein (different residues) + 1 from mates
    protein_focal_indices = [3, 17]  # One from blue (res 0), one from purple (res 2)
    mate_focal_idx = 1  # One mate as focal

    # Collect all edges and neighbors
    all_protein_neighbors = set()
    all_mate_neighbors_from_protein = set()
    all_protein_neighbors_from_mate = set()
    edges_drawn = set()

    # Draw from protein focal atoms
    for focal_idx in protein_focal_indices:
        fp = protein_pos[focal_idx]

        # Draw radius circle
        circle = plt.Circle(fp, radius, fill=False, color='#95a5a6',
                           linestyle='--', linewidth=1.2, alpha=0.5)
        ax.add_patch(circle)

        # Find protein neighbors within radius
        dists = np.linalg.norm(protein_pos - fp, axis=1)
        neighbors = np.where((dists < radius) & (dists > 0))[0]
        all_protein_neighbors.update(neighbors)

        # Draw edges to protein neighbors
        for n in neighbors:
            edge_key = ('p', min(focal_idx, n), max(focal_idx, n))
            if edge_key not in edges_drawn:
                ax.plot([fp[0], protein_pos[n, 0]], [fp[1], protein_pos[n, 1]],
                       color='#2c3e50', linewidth=1.5, alpha=0.7, zorder=1)
                edges_drawn.add(edge_key)

        # Check mate neighbors from this protein focal
        mate_dists = np.linalg.norm(mate_pos - fp, axis=1)
        m_neighbors = np.where(mate_dists < radius)[0]
        all_mate_neighbors_from_protein.update(m_neighbors)
        for n in m_neighbors:
            edge_key = ('pm', focal_idx, n)
            if edge_key not in edges_drawn:
                ax.plot([fp[0], mate_pos[n, 0]], [fp[1], mate_pos[n, 1]],
                       color='#e67e22', linewidth=1.5, alpha=0.7, zorder=1)
                edges_drawn.add(edge_key)

    # Draw from mate focal atom
    mate_fp = mate_pos[mate_focal_idx]
    circle = plt.Circle(mate_fp, radius, fill=False, color='#d35400',
                       linestyle='--', linewidth=1.2, alpha=0.5)
    ax.add_patch(circle)

    # Mate to protein edges
    dists_to_protein = np.linalg.norm(protein_pos - mate_fp, axis=1)
    p_neighbors = np.where(dists_to_protein < radius)[0]
    all_protein_neighbors_from_mate.update(p_neighbors)
    for n in p_neighbors:
        edge_key = ('mp', mate_focal_idx, n)
        if edge_key not in edges_drawn:
            ax.plot([mate_fp[0], protein_pos[n, 0]], [mate_fp[1], protein_pos[n, 1]],
                   color='#e67e22', linewidth=1.5, alpha=0.7, zorder=1)
            edges_drawn.add(edge_key)

    # Mate to mate edges
    dists_to_mates = np.linalg.norm(mate_pos - mate_fp, axis=1)
    mm_neighbors = np.where((dists_to_mates < radius) & (dists_to_mates > 0))[0]
    for n in mm_neighbors:
        edge_key = ('mm', min(mate_focal_idx, n), max(mate_focal_idx, n))
        if edge_key not in edges_drawn:
            ax.plot([mate_fp[0], mate_pos[n, 0]], [mate_fp[1], mate_pos[n, 1]],
                   color='#e67e22', linewidth=1.5, alpha=0.7, zorder=1)
            edges_drawn.add(edge_key)

    # Plot all residue atoms
    for i in range(4):
        mask = protein_res == i
        ax.scatter(protein_pos[mask, 0], protein_pos[mask, 1],
                  c=colors[i], s=65, edgecolors='white', linewidths=0.4, zorder=3)

    # Plot mates (non-focal)
    non_focal_mates = [i for i in range(len(mate_pos)) if i != mate_focal_idx]
    ax.scatter(mate_pos[non_focal_mates, 0], mate_pos[non_focal_mates, 1],
              c='#e67e22', s=65, marker='s', edgecolors='white', linewidths=0.4, zorder=3)

    # Highlight protein neighbors (not focal atoms)
    all_highlighted_protein = all_protein_neighbors | all_protein_neighbors_from_mate
    neighbor_not_focal = [n for n in all_highlighted_protein if n not in protein_focal_indices]
    if neighbor_not_focal:
        ax.scatter(protein_pos[neighbor_not_focal, 0], protein_pos[neighbor_not_focal, 1],
                  c=[colors[protein_res[n]] for n in neighbor_not_focal], s=90,
                  edgecolors='#2c3e50', linewidths=1.5, zorder=4)

    # Highlight mate neighbors (not focal mate)
    all_highlighted_mates = all_mate_neighbors_from_protein | set(mm_neighbors)
    mate_neighbor_list = [n for n in all_highlighted_mates if n != mate_focal_idx]
    if mate_neighbor_list:
        ax.scatter(mate_pos[mate_neighbor_list, 0], mate_pos[mate_neighbor_list, 1],
                  c='#e67e22', s=90, marker='s', edgecolors='#2c3e50', linewidths=1.5, zorder=4)

    # Highlight protein focal atoms with black border
    for focal_idx in protein_focal_indices:
        ax.scatter([protein_pos[focal_idx, 0]], [protein_pos[focal_idx, 1]],
                  c=colors[protein_res[focal_idx]], s=110,
                  edgecolors='black', linewidths=2, zorder=5)

    # Highlight mate focal atom with black border
    ax.scatter([mate_fp[0]], [mate_fp[1]], c='#e67e22', s=110,
              marker='s', edgecolors='black', linewidths=2, zorder=5)

    # Add radius annotation
    ax.annotate(f'r = {cutoff}Å', xy=(0.65, 0.75), fontsize=10, color='#7f8c8d',
               fontweight='bold')

    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0.15, 0.82)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(A) Protein Radius Graph', fontweight='bold', fontsize=12)

    # Panel B: Combined Water KNN (both P-W and W-W)
    ax = axes[1]

    # Create a protein surface (curved arrangement)
    n_protein = 25
    theta = np.linspace(-0.3, np.pi + 0.3, n_protein)
    protein_x = 0.5 + 0.35 * np.cos(theta)
    protein_y = 0.5 + 0.35 * np.sin(theta)
    # Add some noise
    protein_x += rng.normal(0, 0.02, n_protein)
    protein_y += rng.normal(0, 0.02, n_protein)
    protein_pos_b = np.column_stack([protein_x, protein_y])

    # Waters - some near protein surface, some clustered together
    water_pos = np.array([
        # Near protein surface
        [0.5, 0.15], [0.3, 0.25], [0.7, 0.25], [0.2, 0.4], [0.8, 0.4],
        # Water cluster in the middle
        [0.45, 0.45], [0.55, 0.45], [0.5, 0.55], [0.4, 0.55], [0.6, 0.55],
        [0.45, 0.35], [0.55, 0.35],
    ])

    # Draw P-W KNN edges (k=4 for more edges)
    k_pw_viz = 4
    for w_idx, w in enumerate(water_pos):
        dists = np.linalg.norm(protein_pos_b - w, axis=1)
        nearest = np.argsort(dists)[:k_pw_viz]
        for p_idx in nearest:
            ax.plot([w[0], protein_pos_b[p_idx, 0]],
                   [w[1], protein_pos_b[p_idx, 1]],
                   color='#3498db', linewidth=1.0, alpha=0.5, zorder=1)

    # Draw W-W KNN edges (k=4 for more edges)
    k_ww_viz = 4
    for i, w1 in enumerate(water_pos):
        dists = np.linalg.norm(water_pos - w1, axis=1)
        dists[i] = np.inf  # exclude self
        nearest = np.argsort(dists)[:k_ww_viz]
        for j in nearest:
            ax.plot([w1[0], water_pos[j, 0]],
                   [w1[1], water_pos[j, 1]],
                   color='#9b59b6', linewidth=1.5, alpha=0.7, zorder=2)

    # Plot protein atoms
    ax.scatter(protein_pos_b[:, 0], protein_pos_b[:, 1], c='#7f8c8d', s=50,
              edgecolors='white', linewidths=0.3, zorder=3)

    # Plot waters
    ax.scatter(water_pos[:, 0], water_pos[:, 1], c='#e74c3c', s=100,
              edgecolors='#c0392b', linewidths=1, zorder=4)

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(B) Water KNN Graph', fontweight='bold', fontsize=12)

    # Add common legend at bottom
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=9, label='Protein atoms'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e67e22',
               markersize=9, label='Symmetry mates'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markeredgecolor='#c0392b', markersize=10, label='Water molecules'),
        Line2D([0], [0], color='#2c3e50', linewidth=2, label='Radius edges'),
        Line2D([0], [0], color='#3498db', linewidth=2, label='Protein-Water KNN'),
        Line2D([0], [0], color='#9b59b6', linewidth=2, label='Water-Water KNN'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
              frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D schematic to: {save_path}")


def plot_trajectory_multipanel(save_path: str):
    """
    Create a clean multi-panel schematic showing flow-based inference.
    Waters evolve from random noise (t=0) to final positions (t=1).
    """
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))

    rng = np.random.default_rng(456)

    # Create binding pocket using parametric curve (no scipy needed)
    t_curve = np.linspace(0, np.pi, 60)

    # Base U-shape with irregularities
    pocket_x_smooth = 0.5 + 0.38 * np.cos(t_curve)
    pocket_y_smooth = 0.15 + 0.75 * np.sin(t_curve)

    # Add bumps and indentations for realism
    pocket_x_smooth += 0.04 * np.sin(5 * t_curve) + 0.02 * np.sin(8 * t_curve)
    pocket_y_smooth += 0.03 * np.cos(6 * t_curve) + 0.02 * np.cos(9 * t_curve)

    # Protein atoms along pocket wall
    atom_indices = np.linspace(2, len(pocket_x_smooth) - 3, 14, dtype=int)
    atom_pos = np.column_stack([pocket_x_smooth[atom_indices], pocket_y_smooth[atom_indices]])
    atom_pos += rng.normal(0, 0.018, atom_pos.shape)

    # Final water positions (clustered in pocket)
    final_waters = np.array([
        [0.35, 0.38], [0.50, 0.32], [0.65, 0.38],
        [0.28, 0.52], [0.45, 0.48], [0.55, 0.48], [0.72, 0.52],
        [0.38, 0.62], [0.62, 0.62],
    ])

    # Starting positions - random across the panel
    start_waters = rng.uniform(0.08, 0.92, (len(final_waters), 2))

    # Timesteps
    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Water colors - gradient from light cyan to deep blue
    water_colors = ['#b8d4e8', '#85c1e2', '#4aa3d4', '#2185c5', '#155a8a']

    for idx, t in enumerate(timesteps):
        ax = axes[idx]

        # Interpolate positions with decreasing noise
        if t < 1.0:
            noise_scale = 0.05 * (1 - t) ** 1.3
            rng_t = np.random.default_rng(456 + idx * 100)
            noise = rng_t.normal(0, noise_scale, final_waters.shape)
            water_pos = (1 - t) * start_waters + t * final_waters + noise
        else:
            water_pos = final_waters.copy()

        # Draw protein body (outside pocket is filled gray)
        ax.fill_between([0, 1], [0, 0], [1, 1], color='#dcdcdc', zorder=1)

        # Draw pocket interior (white)
        pocket_fill_x = np.concatenate([[pocket_x_smooth[0]], pocket_x_smooth, [pocket_x_smooth[-1]]])
        pocket_fill_y = np.concatenate([[0], pocket_y_smooth, [0]])
        ax.fill(pocket_fill_x, pocket_fill_y, color='#fafafa', zorder=2)

        # Draw pocket boundary
        ax.plot(pocket_x_smooth, pocket_y_smooth, color='#4a5568', linewidth=2.8,
                solid_capstyle='round', zorder=3)

        # Draw protein atoms
        ax.scatter(atom_pos[:, 0], atom_pos[:, 1], c='#4a5568', s=35,
                  edgecolors='white', linewidths=0.3, zorder=4)

        # Draw waters
        ax.scatter(water_pos[:, 0], water_pos[:, 1], c=water_colors[idx], s=70,
                  edgecolors='#0d5a8c', linewidths=1.0, zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        ax.set_title(f't = {t}', fontsize=12, fontweight='bold', pad=8)

    # Simple legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a5568',
               markersize=8, label='Protein'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#45b0d8',
               markeredgecolor='#0d5a8c', markersize=9, label='Water'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, wspace=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory multi-panel to: {save_path}")


def plot_trajectory_single(save_path: str):
    """
    Create a single-panel schematic showing flow-based inference trajectory.
    Waters colored by timestep with trajectory lines showing paths.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    rng = np.random.default_rng(789)

    # Define protein surface as a curved shape
    theta = np.linspace(0.15, np.pi - 0.15, 50)
    protein_x = 0.5 + 0.4 * np.cos(theta)
    protein_y = 0.52 + 0.4 * np.sin(theta)

    # Add some protein atoms along the surface
    protein_atoms_idx = np.linspace(0, len(theta)-1, 15, dtype=int)
    protein_atoms = np.column_stack([protein_x[protein_atoms_idx], protein_y[protein_atoms_idx]])

    # Ground truth water positions
    ground_truth = np.array([
        [0.32, 0.38], [0.50, 0.28], [0.68, 0.38],
        [0.25, 0.52], [0.50, 0.48], [0.75, 0.52],
        [0.38, 0.62], [0.62, 0.62], [0.50, 0.40],
    ])

    # Random starting positions (t=0)
    random_start = rng.uniform(0.12, 0.88, (len(ground_truth), 2))

    # Draw protein surface as filled region
    protein_fill_x = np.concatenate([protein_x, [protein_x[-1], protein_x[0]]])
    protein_fill_y = np.concatenate([protein_y, [0.98, 0.98]])
    ax.fill(protein_fill_x, protein_fill_y, color='#f8f9fa', alpha=0.9, zorder=1)

    # Draw protein surface line
    ax.plot(protein_x, protein_y, color='#7f8c8d', linewidth=4, solid_capstyle='round', zorder=2)

    # Draw protein atoms
    ax.scatter(protein_atoms[:, 0], protein_atoms[:, 1], c='#7f8c8d', s=60,
              edgecolors='white', linewidths=0.5, zorder=3)

    # Timesteps for trajectory
    timesteps = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cmap = plt.cm.YlOrRd  # Yellow to red colormap for time progression

    # Draw trajectories for each water molecule
    for w_idx in range(len(ground_truth)):
        trajectory = []
        for t in np.linspace(0, 1, 20):
            noise_scale = 0.05 * (1 - t)
            noise = rng.normal(0, noise_scale, 2) if t < 1.0 else np.zeros(2)
            pos = (1 - t) * random_start[w_idx] + t * ground_truth[w_idx] + noise
            trajectory.append(pos)
        trajectory = np.array(trajectory)

        # Draw trajectory line with gradient
        for i in range(len(trajectory) - 1):
            t_val = i / (len(trajectory) - 1)
            color = cmap(0.2 + 0.7 * t_val)
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],
                   color=color, linewidth=1.5, alpha=0.6, zorder=4)

    # Draw water positions at key timesteps
    for t_idx, t in enumerate(timesteps):
        if t < 1.0:
            noise_scale = 0.05 * (1 - t)
            noise = rng.normal(0, noise_scale, ground_truth.shape)
            water_pos = (1 - t) * random_start + t * ground_truth + noise
        else:
            water_pos = ground_truth.copy()

        color = cmap(0.2 + 0.7 * t)
        size = 40 + t * 40  # Size increases with time
        alpha = 0.5 + t * 0.5

        if t == 1.0:
            # Final positions - larger and prominent
            ax.scatter(water_pos[:, 0], water_pos[:, 1], c=[color], s=100,
                      edgecolors='#c0392b', linewidths=1.5, zorder=7, alpha=1.0)
        elif t == 0.0:
            # Starting positions - hollow
            ax.scatter(water_pos[:, 0], water_pos[:, 1], facecolors=[color],
                      edgecolors='#f39c12', s=50, linewidths=1, zorder=5, alpha=0.7)
        else:
            ax.scatter(water_pos[:, 0], water_pos[:, 1], c=[color], s=size,
                      edgecolors='none', zorder=6, alpha=alpha)

    # Draw ground truth positions (hollow green circles)
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1],
              facecolors='none', edgecolors='#27ae60', s=120, linewidths=2.5,
              zorder=8, linestyle='--')

    # Add time colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.08,
                        shrink=0.6, aspect=25)
    cbar.set_label('Time (t)', fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7f8c8d',
               markersize=10, label='Protein'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#27ae60', markeredgewidth=2.5, markersize=10, label='Ground truth'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.9),
               markeredgecolor='#c0392b', markersize=10, label='Final prediction (t=1)'),
        Line2D([0], [0], color=cmap(0.5), linewidth=2, label='Trajectory'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.05, 0.99)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Flow-Based Water Placement Trajectory', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory single-panel to: {save_path}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {args.pdb_list}")
    dataset = ProteinWaterDataset(
        pdb_list_file=args.pdb_list,
        processed_dir=args.processed_dir,
        base_pdb_dir=args.base_pdb_dir,
        include_mates=True,
        cutoff=args.cutoff,
        preprocess=True,
    )

    if len(dataset) == 0:
        print("No samples found in dataset")
        return

    sample_idx = min(args.sample_idx, len(dataset) - 1)
    data = dataset[sample_idx]
    pdb_id = getattr(data, 'pdb_id', f'sample_{sample_idx}')

    print(f"\nVisualizing sample: {pdb_id}")
    print(f"  Protein atoms: {data['protein'].num_nodes}")
    print(f"  ASU atoms: {data.num_asu_protein_atoms}")
    print(f"  Mate atoms: {data['protein'].num_nodes - data.num_asu_protein_atoms}")
    print(f"  Waters: {data['water'].num_nodes}")

    protein_pos = data['protein'].pos
    water_pos = data['water'].pos
    pp_edge_index = data['protein', 'pp', 'protein'].edge_index
    residue_indices = data['protein'].residue_index
    num_asu = data.num_asu_protein_atoms

    if args.focal_atom is not None:
        focal_atom = args.focal_atom
    else:
        focal_atom = find_focal_atom_near_waters(protein_pos, water_pos, num_asu)

    print(f"  Focal atom: {focal_atom}")

    # Plot 1: Protein radius graph
    plot_protein_radius_graph(
        protein_pos=protein_pos,
        edge_index=pp_edge_index,
        residue_indices=residue_indices,
        num_asu_atoms=num_asu,
        focal_atom=focal_atom,
        neighborhood_radius=args.neighborhood_radius,
        mate_cutoff=args.mate_distance_cutoff,
        cutoff=args.cutoff,
        edge_sample_frac=args.edge_sample_frac,
        save_path=str(output_dir / f"{pdb_id}_protein_radius_graph.png"),
        pdb_id=pdb_id,
        rng=rng,
    )

    # Plot 2: Water KNN edges
    focal_pos = protein_pos[focal_atom].numpy()
    plot_water_knn_edges(
        protein_pos=protein_pos,
        water_pos=water_pos,
        residue_indices=residue_indices,
        num_asu_atoms=num_asu,
        mate_cutoff=args.mate_distance_cutoff,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
        edge_sample_frac=args.edge_sample_frac,
        save_path=str(output_dir / f"{pdb_id}_water_knn_edges.png"),
        pdb_id=pdb_id,
        neighborhood_center=focal_pos,
        neighborhood_radius=args.neighborhood_radius,
        rng=rng,
    )

    # Plot 3: 2D schematic
    plot_2d_schematic(
        save_path=str(output_dir / "graph_construction_schematic.png"),
        cutoff=args.cutoff,
        k_pw=args.k_pw,
        k_ww=args.k_ww,
    )

    # Plot 4: Trajectory multi-panel
    plot_trajectory_multipanel(
        save_path=str(output_dir / "trajectory_multipanel.png"),
    )

    # Plot 5: Trajectory single-panel
    plot_trajectory_single(
        save_path=str(output_dir / "trajectory_single.png"),
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
