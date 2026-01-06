import os
import errno
import signal
import contextlib
from pathlib import Path

import numpy as np
import torch
import pyrosetta
from pyrosetta import Pose
from pyrosetta.rosetta import core
from pyrosetta.rosetta.core.scoring.methods import EnergyMethodOptions
from tqdm import tqdm


INPUT_DIR       = Path("/sb/wankowicz_lab/data/srivasv/af2_structures/organized_pdb")
LIST_FILE       = Path("../all_pdbs.list")
OUTPUT_DIR      = Path("/sb/wankowicz_lab/data/srivasv/af2_energies")

TIMEOUT_SECONDS = 200           # per‑file wall‑clock limit
MAX_RESIDUES    = 1000           # crop length to reduce memory
ROSETTA_FLAGS   = "-mute all"   # quiet stdout


class TimeoutException(RuntimeError):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    def _handler(signum, frame):
        raise TimeoutException(errno.ETIMEDOUT, "Timed out")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def crop_pose(pose: Pose, max_len: int = MAX_RESIDUES):
    """Delete residues (all chains) after *max_len* to save memory."""
    if pose.size() <= max_len:
        return
    for res in range(pose.size(), max_len, -1):
        pose.delete_residue_slow(res)


# ───────────────────────────── per‑file work ──────────────────────────────
def count_residues_quick(pdb_path: Path, limit=MAX_RESIDUES + 1) -> int:
    """
    Fast text scan that stops once it is sure the file exceeds *limit* residues.
    Works for both PDB and mmCIF produced by AF3.
    """
    seen = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                key = line[21:22] + line[22:27]  # chain + resseq
                seen.add(key)
                if len(seen) >= limit:
                    return len(seen)
    return len(seen)

def process_pdb(pdb_path: Path) -> bool:
    out_path = OUTPUT_DIR / pdb_path.with_suffix(".pt").name
    if out_path.exists():
        return True  # skip
    if count_residues_quick(pdb_path) > MAX_RESIDUES:
        print(f"[SKIP]  {pdb_path.name} has >{MAX_RESIDUES} residues")
        return False
    try:
        with time_limit(TIMEOUT_SECONDS):
            pose = Pose()
            pyrosetta.rosetta.core.import_pose.pose_from_file(pose, str(pdb_path))

            crop_pose(pose)  # truncate long proteins

            scorefxn = pyrosetta.get_fa_scorefxn()
            emo = EnergyMethodOptions()
            emo.hbond_options().decompose_bb_hb_into_pair_energies(True)
            scorefxn.set_energy_method_options(emo)
            scorefxn(pose)

            energy_graph = pose.energies().energy_graph()
            L = pose.size()
            tensor = np.zeros((L, L, 5), dtype=np.float32)

            atr, rep = core.scoring.ScoreType.fa_atr, core.scoring.ScoreType.fa_rep
            ele, sol = core.scoring.ScoreType.fa_elec, core.scoring.ScoreType.fa_sol
            hbs = (
                core.scoring.ScoreType.hbond_sr_bb,
                core.scoring.ScoreType.hbond_lr_bb,
                core.scoring.ScoreType.hbond_bb_sc,
                core.scoring.ScoreType.hbond_sc,
            )

            for i in range(1, L + 1):
                for j in range(i + 1, L + 1):
                    edge = energy_graph.find_energy_edge(i, j)
                    if edge is None:
                        continue
                    emap = edge.fill_energy_map()
                    a, r = emap[atr], emap[rep]
                    e, s = emap[ele], emap[sol]
                    h = sum(emap[x] for x in hbs)
                    if a or r or e or s or h:
                        tensor[i - 1, j - 1] = (a, r, h, e, s)

            torch.save(torch.tensor(tensor), out_path)
            del pose, energy_graph, tensor
            import gc; gc.collect()
            return True

    except TimeoutException:
        print(f"[TIMEOUT] {pdb_path.name} >{TIMEOUT_SECONDS}s — skipped.")
    except Exception as exc:
        print(f"[ERROR]   {pdb_path.name}: {exc}")
    return False


def main():
    # initialise PyRosetta once (no multiprocessing)
    pyrosetta.init(ROSETTA_FLAGS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with LIST_FILE.open() as fh:
        pdb_ids = [ln.strip() for ln in fh if ln.strip()]

    pdb_paths = [INPUT_DIR / pid for pid in pdb_ids]

    print(pdb_ids[0])
    print(pdb_paths[0])

    ok, skipped = 0, 0
    with tqdm(total=len(pdb_paths), unit="pdb") as bar:
        for pdb in pdb_paths:
            if process_pdb(pdb):
                ok += 1
            else:
                skipped += 1
            bar.update(1)

    print(f"\n {ok} tensors written, {skipped} files skipped or failed.")


if __name__ == "__main__":
    main()
