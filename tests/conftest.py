import os
from pathlib import Path

import pytest
import torch


TEST_DIR = Path(__file__).parent
ENV_PDB_DIR = os.environ.get("ENV_PDB_DIR")
PDB_BASE_DIR = Path(ENV_PDB_DIR) if ENV_PDB_DIR else TEST_DIR / "test_files"


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def pdb_base_dir():
    """Wrapper of constant PDB_BASE_DIR."""
    return PDB_BASE_DIR


def _resolve_pdb_path(pdb_id):
    """Resolves a PDB path, skipping if missing."""
    path = PDB_BASE_DIR / pdb_id / f"{pdb_id}_final.pdb"
    if not path.exists():
        pytest.skip(f"PDB file not found: {path}")
    return str(path)


@pytest.fixture
def pdb_6eey():
    """6eey - standard PDB that passes all quality checks."""
    return _resolve_pdb_path("6eey")


@pytest.fixture
def pdb_2b5w():
    """2b5w - fails COM distance check."""
    return _resolve_pdb_path("2b5w")


@pytest.fixture
def pdb_8dzt():
    """8dzt - fails water clash check at 2% threshold with 2A distance."""
    return _resolve_pdb_path("8dzt")


@pytest.fixture
def pdb_1deu():
    """1deu - has insertion codes (52 residues with ins_code='P')."""
    return _resolve_pdb_path("1deu")
