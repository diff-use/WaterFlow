import os
from pathlib import Path

import pytest
import torch

TEST_DIR = Path(__file__).parent


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def pdb_base_dir():
    """Base directory for test PDB files.

    Override via the TEST_DATA_DIR environment variable (e.g. in CI).
    """
    env = os.environ.get("TEST_DATA_DIR")
    if env:
        return Path(env)
    return TEST_DIR / "test_files"


@pytest.fixture
def pdb_path(pdb_base_dir):
    """Factory fixture that resolves a PDB path, skipping if missing."""
    def _resolve(pdb_id: str) -> str:
        path = pdb_base_dir / pdb_id / f"{pdb_id}_final.pdb"
        if not path.exists():
            pytest.skip(f"PDB file not found: {path}")
        return str(path)
    return _resolve


@pytest.fixture
def pdb_6eey(pdb_path):
    """6eey - standard PDB that passes all quality checks."""
    return pdb_path("6eey")


@pytest.fixture
def pdb_2b5w(pdb_path):
    """2b5w - fails COM distance check."""
    return pdb_path("2b5w")


@pytest.fixture
def pdb_8dzt(pdb_path):
    """8dzt - fails water clash check at 2% threshold with 2A distance."""
    return pdb_path("8dzt")


@pytest.fixture
def pdb_1deu(pdb_path):
    """1deu - has insertion codes (52 residues with ins_code='P')."""
    return pdb_path("1deu")
