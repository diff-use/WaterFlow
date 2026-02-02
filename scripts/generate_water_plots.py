"""
Generate whole-dataset distribution and correlation plots for water quality metrics.

Generates:
- EDIA distribution histograms (all waters + mean per PDB)
- RSCC distribution histograms (all waters + mean per PDB)
- B-factor distribution histograms (all waters + mean per PDB)
- EDIA vs B-factor correlation scatter/hexbin

Usage:
    uv run scripts/generate_water_plots.py --output-dir figures/water_quality
    uv run scripts/generate_water_plots.py --skip-bfactor --output-dir figures/water_quality
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biotite.structure.io.pdb import PDBFile
from tqdm import tqdm


def load_all_water_data(edia_dir: Path) -> pd.DataFrame:
    """Load all EDIA CSV files and filter for water (HOH) residues only."""
    all_data = []

    csv_files = list(edia_dir.rglob("*_residue_stats.csv"))
    print(f"Found {len(csv_files)} EDIA CSV files")

    for csv_file in tqdm(csv_files, desc="Loading EDIA data"):
        try:
            df = pd.read_csv(csv_file)
            # Filter for water molecules only
            water_df = df[df["compID"] == "HOH"].copy()
            if len(water_df) > 0:
                # Add PDB ID from filename
                pdb_id = csv_file.stem.replace("_residue_stats", "")
                water_df["pdb_id"] = pdb_id
                all_data.append(water_df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not all_data:
        raise ValueError("No water data found in any CSV files")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined)} water molecules from {len(all_data)} PDBs")
    return combined


def extract_water_bfactors_from_pdb(pdb_path: Path) -> pd.DataFrame | None:
    """Extract B-factors for water molecules from a PDB file using biotite.

    B-factors are normalized using whole-PDB statistics (mean and std of all atoms)
    to account for structure-to-structure variation.

    Args:
        pdb_path: Path to the PDB file

    Returns:
        DataFrame with columns: pdb_id, chain_id, res_id, b_factor, b_factor_normalized
        Returns None if extraction fails
    """
    try:
        pdb_file = PDBFile.read(pdb_path)
        atoms = pdb_file.get_structure(model=1, altloc="occupancy", extra_fields=["b_factor"])

        #compute whole-PDB B-factor statistics for normalization
        all_bfactors = atoms.b_factor
        pdb_mean = np.mean(all_bfactors)
        pdb_std = np.std(all_bfactors)

        #filter for water molecules (HOH or WAT)
        water_mask = (atoms.res_name == "HOH") | (atoms.res_name == "WAT")
        water_atoms = atoms[water_mask]

        if len(water_atoms) == 0:
            return None

        #extract PDB ID from filename (e.g., "3ilf_final.pdb" -> "3ilf")
        pdb_id = pdb_path.stem.replace("_final", "")

        #build DataFrame with one row per unique water residue
        #water molecules have one oxygen atom, so we take unique (chain, res_id) pairs
        records = []
        seen = set()
        for i in range(len(water_atoms)):
            chain_id = water_atoms.chain_id[i]
            res_id = water_atoms.res_id[i]
            key = (chain_id, res_id)
            if key not in seen:
                seen.add(key)
                raw_bfactor = water_atoms.b_factor[i]
                #z-score using whole-PDB statistics
                normalized = (raw_bfactor - pdb_mean) / pdb_std if pdb_std > 0 else 0.0
                records.append({
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "b_factor": raw_bfactor,
                    "b_factor_normalized": normalized,
                })

        return pd.DataFrame(records)

    except Exception as e:
        print(f"Error extracting B-factors from {pdb_path}: {e}")
        return None


def _extract_bfactors_worker(args: tuple) -> pd.DataFrame | None:
    """Worker function for parallel B-factor extraction."""
    pdb_path, pdb_id = args
    return extract_water_bfactors_from_pdb(pdb_path)


def load_all_bfactors(pdb_dir: Path, pdb_ids: list[str], num_workers: int = 4) -> pd.DataFrame:
    """Load B-factors for all PDB IDs in parallel.

    Args:
        pdb_dir: Directory containing PDB files (organized as pdb_dir/pdb_id/pdb_id_final.pdb)
        pdb_ids: List of PDB IDs to process
        num_workers: Number of parallel workers

    Returns:
        DataFrame with columns: pdb_id, chain_id, res_id, b_factor
    """
    #build list of (pdb_path, pdb_id) tuples
    tasks = []
    for pdb_id in pdb_ids:
        pdb_path = pdb_dir / pdb_id / f"{pdb_id}_final.pdb"
        if pdb_path.exists():
            tasks.append((pdb_path, pdb_id))

    print(f"Found {len(tasks)} PDB files out of {len(pdb_ids)} requested")

    all_bfactors = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_extract_bfactors_worker, task): task[1] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting B-factors"):
            result = future.result()
            if result is not None:
                all_bfactors.append(result)

    if not all_bfactors:
        raise ValueError("No B-factor data extracted from any PDB files")

    combined = pd.concat(all_bfactors, ignore_index=True)
    print(f"Extracted B-factors for {len(combined)} water molecules from {len(all_bfactors)} PDBs")
    return combined


def merge_edia_with_bfactors(edia_df: pd.DataFrame, bfactor_df: pd.DataFrame) -> pd.DataFrame:
    """Merge EDIA data with B-factor data.

    Matching is done on (pdb_id, chain, residue_number):
    - EDIA: pdb_strandID (chain), pdb_seqNum (residue number)
    - PDB: chain_id, res_id

    Args:
        edia_df: DataFrame with EDIA data (must have pdb_id, pdb_strandID, pdb_seqNum)
        bfactor_df: DataFrame with B-factor data (must have pdb_id, chain_id, res_id,
                    b_factor, b_factor_normalized)

    Returns:
        Merged DataFrame with b_factor and b_factor_normalized columns added
    """
    #rename B-factor columns to match EDIA column names
    bfactor_renamed = bfactor_df.rename(columns={
        "chain_id": "pdb_strandID",
        "res_id": "pdb_seqNum",
    })

    #merge on the matching key
    merged = edia_df.merge(
        bfactor_renamed[["pdb_id", "pdb_strandID", "pdb_seqNum", "b_factor", "b_factor_normalized"]],
        on=["pdb_id", "pdb_strandID", "pdb_seqNum"],
        how="left",
    )

    #report match statistics
    n_total = len(merged)
    n_matched = merged["b_factor"].notna().sum()
    match_rate = 100 * n_matched / n_total if n_total > 0 else 0
    print(f"B-factor match rate: {n_matched}/{n_total} ({match_rate:.1f}%)")

    return merged


def plot_ediam_waters(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of EDIAm for all water molecules."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["EDIAm"].dropna(), bins=50, edgecolor="black", alpha=0.7, color="steelblue")

    # Add threshold lines
    ax.axvline(x=0.4, color="red", linestyle="--", linewidth=2, label="EDIAm = 0.4")
    ax.axvline(x=0.8, color="orange", linestyle="--", linewidth=2, label="EDIAm = 0.8")

    # Add statistics
    n_total = len(df)
    mean_val = df["EDIAm"].mean()
    median_val = df["EDIAm"].median()

    # Count waters in each threshold region
    n_low = (df["EDIAm"] < 0.4).sum()
    n_mid = ((df["EDIAm"] >= 0.4) & (df["EDIAm"] < 0.8)).sum()
    n_high = (df["EDIAm"] >= 0.8).sum()

    textstr = (
        f"n = {n_total:,}\n"
        f"mean = {mean_val:.3f}\n"
        f"median = {median_val:.3f}\n"
        f"─────────────\n"
        f"< 0.4: {n_low:,} ({100*n_low/n_total:.1f}%)\n"
        f"0.4–0.8: {n_mid:,} ({100*n_mid/n_total:.1f}%)\n"
        f"≥ 0.8: {n_high:,} ({100*n_high/n_total:.1f}%)"
    )
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("EDIAm Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Water EDIAm Scores (All Waters)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "01_ediam_waters.png", dpi=150)
    plt.close(fig)
    print("Saved: 01_ediam_waters.png")


def plot_ediam_pdbs(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of mean EDIAm per PDB."""
    pdb_means = df.groupby("pdb_id")["EDIAm"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(pdb_means.dropna(), bins=50, edgecolor="black", alpha=0.7, color="steelblue")

    # Add threshold lines
    ax.axvline(x=0.4, color="red", linestyle="--", linewidth=2, label="EDIAm = 0.4")
    ax.axvline(x=0.8, color="orange", linestyle="--", linewidth=2, label="EDIAm = 0.8")

    # Add statistics
    n_pdbs = len(pdb_means)
    mean_val = pdb_means.mean()
    median_val = pdb_means.median()

    # Count PDBs in each threshold region
    n_low = (pdb_means < 0.4).sum()
    n_mid = ((pdb_means >= 0.4) & (pdb_means < 0.8)).sum()
    n_high = (pdb_means >= 0.8).sum()

    textstr = (
        f"n = {n_pdbs:,} PDBs\n"
        f"mean = {mean_val:.3f}\n"
        f"median = {median_val:.3f}\n"
        f"─────────────\n"
        f"< 0.4: {n_low:,} ({100*n_low/n_pdbs:.1f}%)\n"
        f"0.4–0.8: {n_mid:,} ({100*n_mid/n_pdbs:.1f}%)\n"
        f"≥ 0.8: {n_high:,} ({100*n_high/n_pdbs:.1f}%)"
    )
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Mean EDIAm Score", fontsize=12)
    ax.set_ylabel("Number of PDBs", fontsize=12)
    ax.set_title("Distribution of Mean Water EDIAm (Per PDB)", fontsize=14)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "02_ediam_pdbs.png", dpi=150)
    plt.close(fig)
    print("Saved: 02_ediam_pdbs.png")


def plot_rsccs_waters(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of RSCCS for all water molecules."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df["RSCCS"].dropna(), bins=50, edgecolor="black", alpha=0.7, color="teal")

    # Add statistics
    n_total = df["RSCCS"].notna().sum()
    mean_val = df["RSCCS"].mean()
    median_val = df["RSCCS"].median()
    textstr = f"n = {n_total:,}\nmean = {mean_val:.3f}\nmedian = {median_val:.3f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("RSCCS Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Water RSCCS Scores (All Waters)", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / "03_rsccs_waters.png", dpi=150)
    plt.close(fig)
    print("Saved: 03_rsccs_waters.png")


def plot_rsccs_pdbs(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of mean RSCCS per PDB."""
    pdb_means = df.groupby("pdb_id")["RSCCS"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(pdb_means.dropna(), bins=50, edgecolor="black", alpha=0.7, color="teal")

    # Add statistics
    n_pdbs = len(pdb_means)
    mean_val = pdb_means.mean()
    median_val = pdb_means.median()
    textstr = f"n = {n_pdbs:,} PDBs\nmean = {mean_val:.3f}\nmedian = {median_val:.3f}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Mean RSCCS Score", fontsize=12)
    ax.set_ylabel("Number of PDBs", fontsize=12)
    ax.set_title("Distribution of Mean Water RSCCS (Per PDB)", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / "04_rsccs_pdbs.png", dpi=150)
    plt.close(fig)
    print("Saved: 04_rsccs_pdbs.png")


def plot_bfactor_waters(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of normalized B-factor for all water molecules.

    B-factors are normalized per-PDB (z-score using whole-PDB mean/std).
    Shows cutoff at 5.0 (high B-factor = worse quality).
    """
    bfactor_data = df["b_factor_normalized"].dropna()

    if len(bfactor_data) == 0:
        print("Skipped: 05_bfactor_waters.png (no B-factor data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(bfactor_data, bins=50, edgecolor="black", alpha=0.7, color="coral")

    # Add cutoff line at 5.0
    cutoff = 5.0
    ax.axvline(x=cutoff, color="red", linestyle="--", linewidth=2, label=f"cutoff = {cutoff}")

    # Add statistics
    n_total = len(bfactor_data)
    mean_val = bfactor_data.mean()
    median_val = bfactor_data.median()

    # Count waters above cutoff
    n_above = (bfactor_data >= cutoff).sum()

    textstr = (
        f"n = {n_total:,}\n"
        f"mean = {mean_val:.2f}\n"
        f"median = {median_val:.2f}\n"
        f"─────────────\n"
        f"≥ {cutoff}: {n_above:,} ({100*n_above/n_total:.1f}%)"
    )
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Normalized B-factor (z-score)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Normalized Water B-factors (All Waters)", fontsize=14)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "05_bfactor_waters.png", dpi=150)
    plt.close(fig)
    print("Saved: 05_bfactor_waters.png")


def plot_bfactor_pdbs(df: pd.DataFrame, output_dir: Path):
    """Plot histogram of mean normalized B-factor per PDB.

    B-factors are normalized per-PDB (z-score using whole-PDB mean/std).
    Shows cutoff at 2.5 (high B-factor = worse quality).
    """
    # Filter to rows with B-factor data
    df_with_bfactor = df[df["b_factor_normalized"].notna()]

    if len(df_with_bfactor) == 0:
        print("Skipped: 06_bfactor_pdbs.png (no B-factor data)")
        return

    pdb_means = df_with_bfactor.groupby("pdb_id")["b_factor_normalized"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(pdb_means, bins=50, edgecolor="black", alpha=0.7, color="coral")

    # Add cutoff line at 2.5
    cutoff = 2.5
    ax.axvline(x=cutoff, color="red", linestyle="--", linewidth=2, label=f"cutoff = {cutoff}")

    # Add statistics
    n_pdbs = len(pdb_means)
    mean_val = pdb_means.mean()
    median_val = pdb_means.median()

    # Count PDBs above cutoff
    n_above = (pdb_means >= cutoff).sum()

    textstr = (
        f"n = {n_pdbs:,} PDBs\n"
        f"mean = {mean_val:.2f}\n"
        f"median = {median_val:.2f}\n"
        f"─────────────\n"
        f"≥ {cutoff}: {n_above:,} ({100*n_above/n_pdbs:.1f}%)"
    )
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Mean Normalized B-factor (z-score)", fontsize=12)
    ax.set_ylabel("Number of PDBs", fontsize=12)
    ax.set_title("Distribution of Mean Normalized Water B-factor (Per PDB)", fontsize=14)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "06_bfactor_pdbs.png", dpi=150)
    plt.close(fig)
    print("Saved: 06_bfactor_pdbs.png")


def plot_ediam_bfactor_correlation(df: pd.DataFrame, output_dir: Path):
    """Plot EDIA vs normalized B-factor scatter with hexbin overlay."""
    # Filter to rows with both EDIA and normalized B-factor data
    df_valid = df[df["EDIAm"].notna() & df["b_factor_normalized"].notna()]

    if len(df_valid) == 0:
        print("Skipped: 07_ediam_bfactor_correlation.png (no matched data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Scatter plot
    ax1 = axes[0]
    ax1.scatter(df_valid["b_factor_normalized"], df_valid["EDIAm"], alpha=0.1, s=5, c="steelblue")

    # Add correlation coefficient
    corr = df_valid["EDIAm"].corr(df_valid["b_factor_normalized"])
    ax1.text(0.02, 0.98, f"r = {corr:.3f}\nn = {len(df_valid):,}",
             transform=ax1.transAxes, fontsize=12,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax1.set_xlabel("Normalized B-factor (z-score)", fontsize=12)
    ax1.set_ylabel("EDIAm Score", fontsize=12)
    ax1.set_title("EDIAm vs Normalized B-factor (Scatter)", fontsize=14)

    # Right: Hexbin density plot
    ax2 = axes[1]
    hb = ax2.hexbin(df_valid["b_factor_normalized"], df_valid["EDIAm"], gridsize=50, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax2, label="Count")

    ax2.set_xlabel("Normalized B-factor (z-score)", fontsize=12)
    ax2.set_ylabel("EDIAm Score", fontsize=12)
    ax2.set_title("EDIAm vs Normalized B-factor (Density)", fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / "07_ediam_bfactor_correlation.png", dpi=150)
    plt.close(fig)
    print("Saved: 07_ediam_bfactor_correlation.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate whole-dataset distribution and correlation plots for water quality metrics"
    )
    parser.add_argument(
        "--edia-dir",
        type=Path,
        default=Path("/sb/wankowicz_lab/data/srivasv/edia_results"),
        help="Directory containing EDIA CSV files",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("/sb/wankowicz_lab/data/srivasv/pdb_redo_data"),
        help="Directory containing PDB files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/water_quality"),
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for B-factor extraction",
    )
    parser.add_argument(
        "--skip-bfactor",
        action="store_true",
        help="Skip B-factor extraction (generate only EDIA/RSCC plots)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load EDIA water data
    print("Loading EDIA water data...")
    df = load_all_water_data(args.edia_dir)

    # Extract and merge B-factors if not skipped
    if not args.skip_bfactor:
        print("\nExtracting B-factors from PDB files...")
        pdb_ids = df["pdb_id"].unique().tolist()
        bfactor_df = load_all_bfactors(args.pdb_dir, pdb_ids, args.num_workers)

        print("\nMerging EDIA and B-factor data...")
        df = merge_edia_with_bfactors(df, bfactor_df)

    # Generate plots
    print("\nGenerating plots...")

    # EDIA plots
    plot_ediam_waters(df, args.output_dir)
    plot_ediam_pdbs(df, args.output_dir)

    # RSCC plots
    plot_rsccs_waters(df, args.output_dir)
    plot_rsccs_pdbs(df, args.output_dir)

    # B-factor plots (only if B-factors were extracted)
    if not args.skip_bfactor:
        plot_bfactor_waters(df, args.output_dir)
        plot_bfactor_pdbs(df, args.output_dir)
        plot_ediam_bfactor_correlation(df, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
