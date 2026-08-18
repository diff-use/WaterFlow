"""Tests for scripts/predict_waters.py -- end-to-end water prediction.

Unit tests cover the pure pieces (selection, model build, lenient load, path
collection, frame recovery). The integration test runs the whole pipeline with
tiny untrained gvp models, so it needs no trained checkpoints or embeddings.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.predict_waters import (
    _collect_struc_paths,
    _kept_atoms_and_center,
    build_confidence_model,
    load_state_dict_lenient,
    parse_args,
    predict_structures,
    select_waters,
)
from src.confidence import ConfidenceGVP
from src.flow import FlowMatcher, FlowWaterGVP


@pytest.mark.unit
class TestSelectWaters:
    def test_confidence_threshold_filters_before_clustering(self):
        # far-apart candidates; the 0.1 one is below threshold and must not survive
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.1])
        sel_pos, sel_conf = select_waters(pos, conf, mode="confidence", threshold=0.5)
        assert sel_pos.shape[0] == 1
        assert torch.allclose(sel_pos[0], pos[0])

    def test_no_threshold_keeps_all_clusters(self):
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.8])
        sel_pos, _ = select_waters(pos, conf, mode="confidence", threshold=None)
        assert sel_pos.shape[0] == 2

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            select_waters(torch.zeros(1, 3), torch.ones(1), mode="bogus")

    def test_density_keeps_top_n_by_confidence(self):
        # Four far-apart candidates -> four singleton clusters, so the kept
        # count is set by the density formula alone, not by clustering merges.
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
        conf = torch.tensor([0.2, 0.9, 0.5, 0.7])
        # floor(0.6 * 5) = 3 kept, highest confidence first
        sel_pos, sel_conf = select_waters(
            pos, conf, mode="density", density_ratio=0.6, num_asu_residues=5
        )
        assert sel_pos.shape[0] == 3
        assert torch.allclose(sel_conf, torch.tensor([0.9, 0.7, 0.5]))

    def test_density_applies_no_cutoff(self):
        # A near-zero confidence candidate still survives in density mode: it is
        # kept because it lands within the top-N count, not filtered by a cutoff.
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.01])
        sel_pos, _ = select_waters(
            pos, conf, mode="density", density_ratio=1.0, num_asu_residues=2
        )
        assert sel_pos.shape[0] == 2

    def test_density_requires_ratio_and_residue_count(self):
        with pytest.raises(ValueError, match="density"):
            select_waters(torch.zeros(1, 3), torch.ones(1), mode="density")


@pytest.mark.unit
class TestSelectionCLI:
    """--selection wiring: each mode owns one knob and rejects the other's."""

    @staticmethod
    def _argv(*extra: str) -> list[str]:
        return [
            "predict_waters",
            "--flow_run_dir", "f",
            "--confidence_run_dir", "c",
            "--struc", "s.pdb",
            "--out_dir", "o",
            *extra,
        ]

    def test_confidence_fills_default_threshold(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._argv("--selection", "confidence"))
        args = parse_args()
        assert args.threshold == 0.5 and args.density_ratio is None

    def test_density_fills_default_ratio(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", self._argv("--selection", "density"))
        args = parse_args()
        assert args.density_ratio == 0.6 and args.threshold is None

    def test_confidence_rejects_density_ratio(self, monkeypatch):
        argv = self._argv("--selection", "confidence", "--density_ratio", "0.6")
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            parse_args()

    def test_density_rejects_threshold(self, monkeypatch):
        argv = self._argv("--selection", "density", "--threshold", "0.5")
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            parse_args()


@pytest.mark.unit
class TestModelBuildAndLoad:
    def test_build_confidence_model(self):
        cfg = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1}
        model = build_confidence_model(cfg, torch.device("cpu"))
        assert isinstance(model, ConfidenceGVP)

    def test_lenient_load_round_trips(self, tmp_path):
        cfg = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1}
        m1 = build_confidence_model(cfg, torch.device("cpu"))
        ckpt = tmp_path / "best.pt"
        torch.save({"model_state_dict": m1.state_dict()}, ckpt)

        m2 = build_confidence_model(cfg, torch.device("cpu"))
        load_state_dict_lenient(m2, ckpt, torch.device("cpu"))  # no raise
        assert not m2.training  # switched to eval

    def test_missing_checkpoint_raises(self, tmp_path):
        cfg = {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1}
        model = build_confidence_model(cfg, torch.device("cpu"))
        with pytest.raises(FileNotFoundError):
            load_state_dict_lenient(model, tmp_path / "nope.pt", torch.device("cpu"))


@pytest.mark.unit
class TestInputsAndFrame:
    def test_single_struc_path(self, pdb_6eey):
        paths = _collect_struc_paths(SimpleNamespace(struc=pdb_6eey, pdb_list=None))
        assert paths == [pdb_6eey]

    def test_pdb_list_resolves_files(self, pdb_6eey, tmp_path):
        base = Path(pdb_6eey).parent.parent
        lst = tmp_path / "list.txt"
        lst.write_text("6eey_final\n")
        paths = _collect_struc_paths(
            SimpleNamespace(struc=None, pdb_list=str(lst), base_pdb_dir=str(base))
        )
        assert len(paths) == 1 and Path(paths[0]).stem == "6eey_final"

    def test_kept_atoms_drop_waters_and_center_is_protein_centroid(self, pdb_4h0b):
        from src.dataset import parse_asu_with_biotite

        kept, center = _kept_atoms_and_center(pdb_4h0b)
        assert int((kept.res_name == "HOH").sum()) == 0
        protein, _w, _lig = parse_asu_with_biotite(pdb_4h0b)
        assert np.allclose(center, protein.coord.mean(axis=0))


@pytest.mark.integration
class TestEndToEnd:
    def test_pipeline_writes_predicted_structure(self, pdb_6eey, gvp_encoder, tmp_path):
        """Whole pipeline on tiny untrained gvp models: graph -> sample -> score ->
        cluster -> select -> un-center -> write. No checkpoints or embeddings."""
        device = torch.device("cpu")
        flow_model = FlowWaterGVP(
            encoder=gvp_encoder, hidden_dims=(64, 8), layers=1
        ).to(device)
        flow_matcher = FlowMatcher(model=flow_model, sampling_strategy="uniform_ball")
        conf_model = build_confidence_model(
            {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1},
            device,
        )

        out_dir = tmp_path / "out"
        args = SimpleNamespace(
            processed_dir=None,
            include_mates=False,
            method="euler",
            num_steps=2,
            water_ratio=1.0,
            selection="confidence",
            threshold=0.0,  # keep all, so the write path is exercised
            density_ratio=None,
            out_dir=str(out_dir),
            out_format=".pdb",
        )

        predict_structures(
            [pdb_6eey],
            flow_matcher,
            conf_model,
            {"encoder_type": "gvp"},
            args,
            device,
        )

        pdb_out = out_dir / "6eey_final_pred.pdb"
        coords_out = out_dir / "6eey_final_waters.txt"
        assert pdb_out.exists() and coords_out.exists()
        # the written structure has HOH waters, and coords are back in the input frame
        from biotite.structure.io.pdb import PDBFile

        written = PDBFile.read(str(pdb_out)).get_structure(model=1)
        n_waters = int((written.res_name == "HOH").sum())
        assert n_waters > 0
        assert np.loadtxt(coords_out).reshape(-1, 3).shape[0] == n_waters
