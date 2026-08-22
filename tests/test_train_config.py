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
    run_eval_sampling,
)
from src.encoder_base import build_encoder
from src.flow import FlowMatcher, FlowWaterGVP


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

    # Cached encoder fuses the embedding + element one-hot to hidden_s width.
    assert model.encoder.output_dims == (256, 0)


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


def test_inference_build_model_from_config_replays_recorded_edge_policy(device):
    """Every recorded config carries "auto". Replaying one must build a model,
    not raise, and must land on the radius path those runs actually used."""
    config = {
        "encoder_type": "slae",
        "hidden_s": 128,
        "hidden_v": 32,
        "flow_layers": 2,
        "node_scalar_in": 16,
        "embedding_dim": 128,
        "dynamic_edge_policy": "auto",
        "knn_fallback_k": 8,
        "cutoff": 8.0,
        "max_neighbors": 256,
        "disable_ww": True,
        "disable_wp": True,
    }

    model = build_model_from_config(config, device)

    assert model.updater.dynamic_edge_policy == "radius"
    assert set(model.updater.etypes) == {
        ("protein", "pw", "water"),
        ("protein", "pp", "protein"),
    }


def test_inference_build_model_from_config_rescues_for_scaled_gaussian(device):
    """A run that recorded scaled_gaussian sampling resolves "auto" to
    knn_if_isolated, so the rebuilt model must carry the isolated-water rescue
    even though its dynamic_edge_policy still reads "radius"."""
    config = {
        "encoder_type": "slae",
        "hidden_s": 128,
        "hidden_v": 32,
        "flow_layers": 2,
        "node_scalar_in": 16,
        "embedding_dim": 128,
        "dynamic_edge_policy": "auto",
        "sampling_strategy": "scaled_gaussian",
        "knn_fallback_k": 8,
        "cutoff": 8.0,
        "max_neighbors": 256,
        "disable_ww": True,
        "disable_wp": True,
    }

    model = build_model_from_config(config, device)

    assert model.updater.dynamic_edge_policy == "radius"
    assert model.updater.rescue_isolated


def test_parse_args_rejects_embedding_dim_for_gvp(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_list", "train.txt",
            "--val_list", "val.txt",
            "--processed_dir", "cache",
            "--base_pdb_dir", "pdbs",
            "--encoder_type", "gvp",
            "--embedding_dim", "128",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_dataset_defaults_match_train_defaults(monkeypatch):
    """Verify dataset.py defaults match train.py argparse defaults."""
    import inspect

    from src.dataset import ProteinWaterDataset

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_list", "t.txt",
            "--val_list", "v.txt",
            "--processed_dir", "cache",
            "--base_pdb_dir", "pdbs",
        ],
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
    assert args.include_ligands == dataset_defaults["include_ligands"]


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


def test_eval_rng_is_isolated(device, gvp_encoder, tmp_path):
    """--val_seed fixes the eval draws but must not leak into the training stream."""
    graph = HeteroData()
    graph["protein"].pos = torch.randn(10, 3, device=device)
    graph["protein"].x = torch.randn(10, 16, device=device)
    graph["protein"].batch = torch.zeros(10, dtype=torch.long, device=device)
    graph["water"].pos = torch.randn(5, 3, device=device)
    graph["water"].x = torch.randn(5, 16, device=device)
    graph["water"].batch = torch.zeros(5, dtype=torch.long, device=device)
    graph["protein", "pp", "protein"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long, device=device
    )
    model = FlowWaterGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(device)
    fm = FlowMatcher(model)
    args = Namespace(
        val_seed=1234, eval_method="euler", eval_steps=3, threshold=1.0, save_gifs=False
    )
    loader = Namespace(dataset=[graph])

    def eval_once():
        return run_eval_sampling(
            fm, loader, args, 1, device, eval_indices=[0], run_dir=tmp_path
        )

    # Training's next draw is the same with or without an eval in between.
    torch.manual_seed(7)
    expected = torch.rand(3, device=device)
    torch.manual_seed(7)
    first = eval_once()
    assert torch.equal(torch.rand(3, device=device), expected)

    # Eval is pinned to val_seed regardless of the outer RNG state.
    torch.manual_seed(99)
    second = eval_once()
    assert first and second == pytest.approx(first, rel=1e-4)
