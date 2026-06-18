"""
Diagnostic visualization for crystal-contact / symmetry-mate handling.

Renders, for a PDB, what the symmetry expansion produces and what the cleanup
(protein-only mates + coordinate dedup) keeps, so the behavior described in the
crystal-contact audit is inspectable rather than asserted.

Key concepts made visible:
  - ASU water   = deposited HOH/WAT, supported by electron density -> PREDICTION TARGET.
  - Crystal/mate water = an ASU water pushed into a neighbor copy by a symmetry
    operator; carries no independent density (a symmetry image of a target water).
    On a special position it lands ~0 A from the ASU water it copies (a label leak).
  - Mate protein = the neighbor molecule's surface = genuine crystal-contact context.

Panels (per PDB):
  1. Raw symmetry expansion, colored by composition (ASU protein / ASU water /
     mate protein / mate water / mate ion-ligand).
  2. Coincidences: mate atoms removed by the cleanup (special-position copies and
     redundant symmetry images) highlighted on top of the ASU.
  3. After cleanup: ASU + surviving mate-protein context only.

Usage:
    python -m scripts.inspect_crystal_contacts \
        --pdb tests/test_files/8dzt/8dzt_final.pdb --out out/
    python -m scripts.inspect_crystal_contacts --all --out out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from src.dataset import (  # noqa: E402
    dedup_mate_atoms,
    get_crystal_contacts_pymol,
    parse_asu_with_biotite,
)


# Amino-acid residue names treated as protein (mirrors biotite filter_amino_acids
# intent, incl. common modified residues), for classifying raw mate atoms.
WATER_RESN = {"HOH", "WAT"}
AA_RESN = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL", "HYP",
}

DEDUP_TOL = 0.3
COINCIDENT_TOL = 0.1


def _classify_raw_mates(coords: np.ndarray, atoms: list):
    """Split raw mate atoms into (protein, water, other) coordinate arrays."""
    prot, water, other = [], [], []
    for c, a in zip(coords, atoms):
        resn = str(getattr(a, "resn", "")).upper()
        if resn in WATER_RESN:
            water.append(c)
        elif resn in AA_RESN:
            prot.append(c)
        else:
            other.append(c)

    def stack(x):
        return np.asarray(x) if x else np.zeros((0, 3))

    return stack(prot), stack(water), stack(other)


def _scatter(ax, pts, *, c, s, label, marker="o", alpha=0.6, depthshade=True):
    if pts is None or len(pts) == 0:
        return
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=c, s=s, marker=marker, alpha=alpha, label=label, depthshade=depthshade,
    )


def _finalize(ax, title):
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, markerscale=1.5)


def _removed_mask(kept_coords: np.ndarray, all_coords: np.ndarray) -> np.ndarray:
    """Boolean mask over all_coords for atoms NOT present in kept_coords."""
    if len(all_coords) == 0:
        return np.zeros(0, dtype=bool)
    if len(kept_coords) == 0:
        return np.ones(len(all_coords), dtype=bool)
    tree = cKDTree(kept_coords)
    d, _ = tree.query(all_coords, k=1)
    return d > 1e-6  # removed atoms have no exact match among kept


def inspect_pdb(pdb_path: Path, out_dir: Path, cutoff: float = 5.0) -> dict:
    """Render the 3 diagnostic panels and return a text-summary dict."""
    asu_protein, asu_water = parse_asu_with_biotite(str(pdb_path))

    raw = get_crystal_contacts_pymol(str(pdb_path), cutoff, protein_only=False)
    prot = get_crystal_contacts_pymol(str(pdb_path), cutoff, protein_only=True)

    reference = (
        np.concatenate([asu_protein.coord, asu_water.coord], axis=0)
        if len(asu_water)
        else asu_protein.coord
    )
    kept_coords, _ = dedup_mate_atoms(
        prot["mate_coords"], prot["mate_atoms"], reference, tol=DEDUP_TOL
    )

    # Center everything on the ASU protein centroid for a stable view.
    center = asu_protein.coord.mean(axis=0, keepdims=True)

    def ctr(x):
        return (np.asarray(x) - center) if len(x) else np.zeros((0, 3))

    asu_p = ctr(asu_protein.coord)
    asu_w = ctr(asu_water.coord)
    raw_p, raw_w, raw_o = _classify_raw_mates(raw["mate_coords"], raw["mate_atoms"])
    raw_p, raw_w, raw_o = ctr(raw_p), ctr(raw_w), ctr(raw_o)

    prot_mate = prot["mate_coords"]
    removed = _removed_mask(kept_coords, prot_mate)
    removed_pts = ctr(prot_mate[removed]) if len(prot_mate) else np.zeros((0, 3))
    kept_pts = ctr(kept_coords)

    # ---- figure ----
    fig = plt.figure(figsize=(21, 7))
    pdb_id = pdb_path.parent.name

    ax1 = fig.add_subplot(131, projection="3d")
    _scatter(ax1, asu_p, c="lightgray", s=6, alpha=0.25, label="ASU protein")
    _scatter(ax1, asu_w, c="red", s=28, marker="*", alpha=0.9, label="ASU water (target)")
    _scatter(ax1, raw_p, c="royalblue", s=8, alpha=0.5, label="mate protein")
    _scatter(ax1, raw_w, c="cyan", s=22, marker="*", alpha=0.9, label="mate water (copy)")
    _scatter(ax1, raw_o, c="orange", s=24, marker="^", alpha=0.9, label="mate ion/ligand")
    _finalize(ax1, f"{pdb_id}: raw symmetry expansion")

    ax2 = fig.add_subplot(132, projection="3d")
    _scatter(ax2, asu_p, c="lightgray", s=6, alpha=0.2, label="ASU protein")
    _scatter(ax2, asu_w, c="red", s=28, marker="*", alpha=0.8, label="ASU water (target)")
    _scatter(ax2, kept_pts, c="royalblue", s=6, alpha=0.25, label="mate (kept)")
    _scatter(
        ax2, removed_pts, c="magenta", s=60, marker="X", alpha=1.0,
        depthshade=False, label="mate removed (dup/leak)",
    )
    _finalize(ax2, f"{pdb_id}: coincidences removed by cleanup")

    ax3 = fig.add_subplot(133, projection="3d")
    _scatter(ax3, asu_p, c="lightgray", s=6, alpha=0.3, label="ASU protein")
    _scatter(ax3, asu_w, c="red", s=28, marker="*", alpha=0.9, label="ASU water (target)")
    _scatter(ax3, kept_pts, c="royalblue", s=8, alpha=0.6, label="mate protein (context)")
    _finalize(ax3, f"{pdb_id}: after protein-only + dedup")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdb_id}_crystal_contacts.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # ---- text summary (mirrors audit probes) ----
    def coincident_pairs(c):
        return len(cKDTree(c).query_pairs(COINCIDENT_TOL)) if len(c) > 1 else 0

    # target waters with a mate-protein node copy within DEDUP_TOL (label leak #2)
    leaks = 0
    if len(asu_water) and len(prot_mate):
        d, _ = cKDTree(prot_mate).query(asu_water.coord, k=1)
        leaks = int((d < DEDUP_TOL).sum())

    return {
        "pdb": pdb_id,
        "asu_protein_atoms": len(asu_protein),
        "asu_waters": len(asu_water),
        "raw_mate_atoms": int(raw["mate_coords"].shape[0]),
        "raw_mate_water": int(len(raw_w)),
        "raw_mate_ion_ligand": int(len(raw_o)),
        "protein_only_mate_atoms": int(prot_mate.shape[0]),
        "after_dedup_mate_atoms": int(len(kept_coords)),
        "coincident_pairs_pre": coincident_pairs(prot_mate),
        "coincident_pairs_post": coincident_pairs(kept_coords),
        "target_water_leak_nodes": leaks,
        "image": str(out_path),
    }


def _print_summary(s: dict) -> None:
    print(f"\n=== {s['pdb']} ===")
    print(f"  ASU: protein_atoms={s['asu_protein_atoms']} waters(targets)={s['asu_waters']}")
    print(
        f"  raw mates={s['raw_mate_atoms']} "
        f"(of which water={s['raw_mate_water']} ion/ligand={s['raw_mate_ion_ligand']})"
    )
    print(
        f"  protein-only mates={s['protein_only_mate_atoms']} "
        f"-> after dedup={s['after_dedup_mate_atoms']} "
        f"(removed {s['protein_only_mate_atoms'] - s['after_dedup_mate_atoms']})"
    )
    print(
        f"  coincident mate pairs <{COINCIDENT_TOL}A: "
        f"pre={s['coincident_pairs_pre']} post={s['coincident_pairs_post']}"
    )
    print(f"  target waters with a mate-node copy <{DEDUP_TOL}A (label leak): {s['target_water_leak_nodes']}")
    print(f"  -> {s['image']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", type=str, help="Path to a *_final.pdb file")
    ap.add_argument(
        "--all", action="store_true",
        help="Sweep all tests/test_files/*/*_final.pdb",
    )
    ap.add_argument("--out", type=str, default="out/crystal_contacts")
    ap.add_argument("--cutoff", type=float, default=5.0)
    ap.add_argument(
        "--test-dir", type=str, default="tests/test_files",
        help="Directory scanned by --all",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.all:
        pdbs = sorted(Path(args.test_dir).glob("*/*_final.pdb"))
        if not pdbs:
            ap.error(f"No *_final.pdb found under {args.test_dir}")
    elif args.pdb:
        pdbs = [Path(args.pdb)]
    else:
        ap.error("Provide --pdb <path> or --all")

    for pdb in pdbs:
        try:
            summary = inspect_pdb(pdb, out_dir, cutoff=args.cutoff)
            _print_summary(summary)
        except Exception as exc:  # noqa: BLE001 - diagnostics should not hard-fail a sweep
            print(f"\n=== {pdb.parent.name} === FAILED: {exc}")


if __name__ == "__main__":
    main()
