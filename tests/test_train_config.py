from argparse import Namespace

import pytest
import torch
from torch_geometric.data import HeteroData

from scripts.inference import build_model_from_config
from scripts.train import (
    _required_embedding_field,
    _resolve_embedding_dim,
    _uses_cached_embeddings,
    parse_args,
    resolve_encoder_config,
)
from src.encoder_base import build_encoder
from src.flow import FlowWaterGVP


@pytest.fixture
def sample_cached_embedding_data(device):
    data = HeteroData()
    data["protein"].x = torch.randn(8, 16, device=device)
    data["protein"].pos = torch.randn(8, 3, device=device)
    data["protein"].batch = torch.zeros(8, dtype=torch.long, device=device)
    data["protein"].embedding = torch.randn(8, 128, device=device)
    data["protein"].embedding_type = "slae"
    return data


def test_required_embedding_field_uses_generic_key():
    assert _required_embedding_field("gvp") is None
    assert _required_embedding_field("slae") == "embedding"
    assert _required_embedding_field("esm") == "embedding"


def test_uses_cached_embeddings_matches_encoder_type():
    assert _uses_cached_embeddings("gvp") is False
    assert _uses_cached_embeddings("slae") is True
    assert _uses_cached_embeddings("esm") is True


def test_resolve_embedding_dim_reads_generic_field(sample_cached_embedding_data):
    dim = _resolve_embedding_dim(sample_cached_embedding_data, "slae", None)
    assert dim == 128


def test_resolve_embedding_dim_raises_when_embedding_missing(device):
    data = HeteroData()
    data["protein"].x = torch.randn(4, 16, device=device)
    data["protein"].pos = torch.randn(4, 3, device=device)

    with pytest.raises(ValueError, match=r"protein\.embedding"):
        _resolve_embedding_dim(data, "slae", None)


def test_resolve_embedding_dim_raises_on_embedding_type_mismatch(
    sample_cached_embedding_data,
):
    with pytest.raises(ValueError, match="embedding_type"):
        _resolve_embedding_dim(sample_cached_embedding_data, "esm", None)


def test_resolve_encoder_config_uses_embedding_dim(sample_cached_embedding_data):
    args = Namespace(
        encoder_type="slae",
        hidden_s=256,
        hidden_v=64,
        freeze_encoder=False,
        encoder_ckpt=None,
        embedding_dim=None,
    )

    config = resolve_encoder_config(args, sample_cached_embedding_data, 16)

    assert config["embedding_key"] == "embedding"
    assert config["embedding_dim"] == 128
    assert "embedding_dim" in config


def test_resolve_encoder_config_applies_embedding_override(
    sample_cached_embedding_data,
):
    args = Namespace(
        encoder_type="slae",
        hidden_s=256,
        hidden_v=64,
        freeze_encoder=False,
        encoder_ckpt=None,
        embedding_dim=128,
    )

    config = resolve_encoder_config(args, sample_cached_embedding_data, 16)

    assert config["embedding_dim"] == 128


def test_cached_encoder_model_construction_succeeds(
    sample_cached_embedding_data, device
):
    args = Namespace(
        encoder_type="slae",
        hidden_s=256,
        hidden_v=64,
        freeze_encoder=False,
        encoder_ckpt=None,
        embedding_dim=None,
    )

    encoder_config = resolve_encoder_config(args, sample_cached_embedding_data, 16)
    encoder = build_encoder(encoder_config, device)
    model = FlowWaterGVP(encoder=encoder)

    assert model.encoder.output_dims == (128, 0)


def test_inference_build_model_from_config_uses_embedding_dim(device):
    config = {
        "encoder_type": "slae",
        "hidden_s": 128,
        "hidden_v": 32,
        "flow_layers": 2,
        "node_scalar_in": 16,
        "embedding_dim": 128,
        "k_pw": 8,
        "k_ww": 8,
    }

    model = build_model_from_config(config, device)

    assert model.encoder.output_dims == (128, 0)


def test_parse_args_rejects_embedding_dim_for_gvp(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_list",
            "train.txt",
            "--val_list",
            "val.txt",
            "--encoder_type",
            "gvp",
            "--embedding_dim",
            "128",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_dataset_defaults_match_train_defaults(monkeypatch):
    """Verify dataset.py defaults match train.py argparse defaults."""
    import inspect

    from src.dataset import ProteinWaterDataset

    monkeypatch.setattr(
        "sys.argv", ["train.py", "--train_list", "t.txt", "--val_list", "v.txt"]
    )
    args = parse_args()

    sig = inspect.signature(ProteinWaterDataset.__init__)
    dataset_defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    assert args.min_water_residue_ratio == dataset_defaults["min_water_residue_ratio"]
    assert args.max_protein_dist == dataset_defaults["max_protein_dist"]
    assert args.max_com_dist == dataset_defaults["max_com_dist"]


def test_inference_extracts_filter_config_from_training_config():
    """Verify inference correctly extracts filter params from training config."""
    from scripts.inference import _extract_dataset_filter_config

    training_config = {
        "min_water_residue_ratio": 0.7,
        "max_protein_dist": 4.5,
        "filter_by_edia": False,
    }

    extracted = _extract_dataset_filter_config(training_config)

    assert extracted["min_water_residue_ratio"] == 0.7
    assert extracted["max_protein_dist"] == 4.5
    assert extracted["filter_by_edia"] is False
    assert extracted["max_com_dist"] == 25.0  # default
    assert extracted["min_edia"] == 0.4  # default
