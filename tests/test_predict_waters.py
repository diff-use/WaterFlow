"""Tests for scripts/predict_waters.py -- end-to-end water prediction.

Unit tests cover the pure pieces (selection, model build, lenient load, path
collection, frame recovery). The integration test runs the whole pipeline with
tiny untrained gvp models, so it needs no trained checkpoints or embeddings.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.predict_waters import (
    _collect_struc_paths,
    _input_frame,
    load_state_dict_lenient,
    predict_structures,
    select_waters,
)
from src.confidence import build_confidence_model, ConfidenceGVP
from src.dataset import parse_asu_with_biotite
from src.flow import FlowMatcher, FlowWaterGVP
from src.structure_io import read_space_group


@pytest.mark.unit
def test_select_waters_unknown_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        select_waters(torch.zeros(1, 3), torch.ones(1), mode="density")


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

    def test_pdb_list_resolves_names_with_and_without_ext(self, pdb_6eey, tmp_path):
        base = Path(pdb_6eey).parent
        lst = tmp_path / "list.txt"
        # one entry carries an extension, one omits it; both resolve to a file
        lst.write_text(f"{Path(pdb_6eey).name}\n6eey_final\n")
        paths = _collect_struc_paths(
            SimpleNamespace(struc=None, pdb_list=str(lst), base_pdb_dir=str(base))
        )
        assert len(paths) == 2
        assert all(Path(p).stem == "6eey_final" for p in paths)

    def test_pdb_list_warns_on_missing(self, tmp_path):
        lst = tmp_path / "list.txt"
        lst.write_text("does_not_exist\n")
        paths = _collect_struc_paths(
            SimpleNamespace(struc=None, pdb_list=str(lst), base_pdb_dir=str(tmp_path))
        )
        assert paths == []

    def test_input_frame(self, pdb_4h0b):
        kept, center, space_group = _input_frame(pdb_4h0b)
        protein, _w, lig = parse_asu_with_biotite(pdb_4h0b)
        assert int((kept.res_name == "HOH").sum()) == 0
        assert len(kept) == len(protein) + len(lig)
        assert np.allclose(center, protein.coord.mean(axis=0))
        assert space_group == "P 6"


@pytest.mark.integration
class TestEndToEnd:
    def test_pipeline_writes_predicted_structure(self, pdb_4h0b, gvp_encoder, tmp_path):
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
            confidence_threshold=0.0,  # keep all, so the write path is exercised
            out_dir=str(out_dir),
            out_format=".pdb",
        )

        predict_structures(
            [pdb_4h0b],
            flow_matcher,
            conf_model,
            {"encoder_type": "gvp"},
            args,
            device,
        )

        pdb_out = out_dir / "4h0b_final_pred.pdb"
        coords_out = out_dir / "4h0b_final_waters.txt"
        assert pdb_out.exists() and coords_out.exists()
        from biotite.structure.io.pdb import PDBFile

        pdb_file = PDBFile.read(str(pdb_out))
        written = pdb_file.get_structure(model=1)
        is_water = written.res_name == "HOH"
        n_waters = int(is_water.sum())
        assert n_waters > 0

        # Protein + ligand atoms are written unchanged, in the input frame, with
        # the input unit cell and space group.
        protein, _w, lig = parse_asu_with_biotite(pdb_4h0b)
        kept = protein + lig
        assert np.allclose(written.coord[~is_water], kept.coord, atol=1e-3)
        assert np.allclose(written.box, kept.box, atol=1e-3)
        assert read_space_group(str(pdb_out)) == read_space_group(pdb_4h0b)

        # Water rows in the txt match the written waters: x y z in the input
        # frame and confidence in [0, 1].
        rows = np.loadtxt(coords_out).reshape(-1, 4)
        assert rows.shape[0] == n_waters
        assert np.allclose(rows[:, :3], written.coord[is_water], atol=1e-3)
        assert ((rows[:, 3] >= 0) & (rows[:, 3] <= 1)).all()
