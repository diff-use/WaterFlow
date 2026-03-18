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


# ============== Shared encoder fixtures ==============


@pytest.fixture
def base_encoder(device):
    """Base ProteinGVPEncoder for flow model tests."""
    from src.gvp_encoder import ProteinGVPEncoder

    return ProteinGVPEncoder(
        node_scalar_in=16,
        hidden_dims=(64, 8),
        n_edge_scalar_in=16,
        pool_residue=False,
    ).to(device)


@pytest.fixture
def gvp_encoder(base_encoder):
    """Wrapped GVPEncoder for flow model tests."""
    from src.gvp_encoder import GVPEncoder

    return GVPEncoder(encoder=base_encoder, freeze=False)


# ============== Dataset test fixtures ==============


@pytest.fixture
def create_mock_dataset(tmp_path, pdb_base_dir):
    """Factory fixture to create mock ProteinWaterDataset instances."""
    from src.dataset import ProteinWaterDataset

    def _create(
        pdb_ids=None,
        encoder_type="gvp",
        include_mates=True,
        preprocess=False,
        **kwargs,
    ):
        if pdb_ids is None:
            pdb_ids = ["6eey"]

        list_file = tmp_path / f"list_{encoder_type}.txt"
        list_file.write_text("\n".join(f"{pdb_id}_final" for pdb_id in pdb_ids))

        return ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / f"processed_{encoder_type}"),
            base_pdb_dir=str(pdb_base_dir),
            encoder_type=encoder_type,
            include_mates=include_mates,
            preprocess=preprocess,
            **kwargs,
        )

    return _create
