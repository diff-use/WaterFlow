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


def _resolve_test_path(pdb_id, suffix):
    """Resolve a test input file (`{pdb_id}_final{suffix}`), raising if missing."""
    path = PDB_BASE_DIR / pdb_id / f"{pdb_id}_final{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    return str(path)


@pytest.fixture
def pdb_6eey():
    """6eey - standard PDB that passes all quality checks."""
    return _resolve_test_path("6eey", ".pdb")


@pytest.fixture
def cif_6eey():
    """6eey - CIF format of standard test structure."""
    return _resolve_test_path("6eey", ".cif")


@pytest.fixture(scope="session")
def parsed_pdb_6eey():
    """Parsed (protein, water) atoms from the 6eey PDB, parsed once per session."""
    from src.dataset import parse_asu_with_biotite

    return parse_asu_with_biotite(_resolve_test_path("6eey", ".pdb"))


@pytest.fixture(scope="session")
def parsed_cif_6eey():
    """Parsed (protein, water) atoms from the 6eey CIF, parsed once per session."""
    from src.dataset import parse_asu_with_biotite

    return parse_asu_with_biotite(_resolve_test_path("6eey", ".cif"))


@pytest.fixture
def edia_6eey():
    """6eey EDIA JSON file with water quality scores from PDB-REDO."""
    return _resolve_test_path("6eey", ".json")


@pytest.fixture
def pdb_2b5w():
    """2b5w - fails COM distance check."""
    return _resolve_test_path("2b5w", ".pdb")


@pytest.fixture
def pdb_8dzt():
    """8dzt - fails water clash check at 2% threshold with 2A distance."""
    return _resolve_test_path("8dzt", ".pdb")


@pytest.fixture
def pdb_1deu():
    """1deu - has insertion codes (52 residues with ins_code='P')."""
    return _resolve_test_path("1deu", ".pdb")


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
