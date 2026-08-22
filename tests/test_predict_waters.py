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
    _check_embeddings,
    _collect_struc_paths,
    _input_frame,
    load_state_dict_lenient,
    parse_args,
    predict_structures,
    select_waters,
)
from src.confidence import build_confidence_model, ConfidenceGVP
from src.dataset import parse_asu_with_biotite
from src.flow import FlowMatcher, FlowWaterGVP
from src.structure_io import read_space_group


@pytest.mark.unit
class TestSelectWaters:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            select_waters(torch.zeros(1, 3), torch.ones(1), mode="bogus")

    def test_density_keeps_top_n_by_confidence(self):
        # Four far-apart candidates: four singleton clusters, so the kept count
        # comes from the density formula alone. floor(0.6 * 5) = 3.
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0], [20.0, 0, 0], [30.0, 0, 0]])
        conf = torch.tensor([0.2, 0.9, 0.5, 0.7])
        sel_pos, sel_conf = select_waters(
            pos, conf, mode="density", density_ratio=0.6, num_asu_residues=5
        )
        assert sel_pos.shape[0] == 3
        assert torch.allclose(sel_conf, torch.tensor([0.9, 0.7, 0.5]))

    def test_density_has_no_cutoff(self):
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.01])
        sel_pos, _ = select_waters(
            pos, conf, mode="density", density_ratio=1.0, num_asu_residues=2
        )
        assert sel_pos.shape[0] == 2

    def test_density_requires_ratio_and_residue_count(self):
        with pytest.raises(ValueError, match="density"):
            select_waters(torch.zeros(1, 3), torch.ones(1), mode="density")

    @pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
    def test_density_rejects_bad_ratio(self, ratio):
        with pytest.raises(ValueError, match="density_ratio"):
            select_waters(
                torch.zeros(1, 3),
                torch.ones(1),
                mode="density",
                density_ratio=ratio,
                num_asu_residues=5,
            )

    def test_nothing_survives(self):
        pos = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
        conf = torch.tensor([0.9, 0.8])
        # confidence: threshold above every score
        sel_pos, sel_conf = select_waters(pos, conf, mode="confidence", threshold=1.0)
        assert sel_pos.shape == (0, 3) and sel_conf.shape == (0,)
        # density: floor(0.1 * 5) = 0
        sel_pos, sel_conf = select_waters(
            pos, conf, mode="density", density_ratio=0.1, num_asu_residues=5
        )
        assert sel_pos.shape == (0, 3) and sel_conf.shape == (0,)


@pytest.mark.unit
class TestSelectionCLI:
    """Each selection mode owns one knob and rejects the other's."""

    @staticmethod
    def _parse(*extra: str):
        base = ["--ckpt_dir", "c"]
        base += ["--struc", "s.pdb", "--out_dir", "o"]
        return parse_args(base + list(extra))

    def test_confidence_default_threshold(self):
        args = self._parse()
        assert args.confidence_threshold == 0.5 and args.density_ratio is None

    def test_density_default_ratio(self):
        args = self._parse("--selection", "density")
        assert args.density_ratio == 0.6 and args.confidence_threshold is None

    def test_confidence_rejects_density_ratio(self):
        with pytest.raises(SystemExit):
            self._parse("--density_ratio", "0.6")

    def test_density_rejects_threshold(self):
        with pytest.raises(SystemExit):
            self._parse("--selection", "density", "--confidence_threshold", "0.5")

    def test_threshold_range(self):
        with pytest.raises(SystemExit):
            self._parse("--confidence_threshold", "1.5")

    @pytest.mark.parametrize("ratio", ["0", "-1", "nan", "inf"])
    def test_density_ratio_range(self, ratio):
        with pytest.raises(SystemExit):
            self._parse("--selection", "density", "--density_ratio", ratio)


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


@pytest.mark.unit
class TestCheckEmbeddings:
    def test_gvp_skips_check(self):
        # gvp needs no embeddings, so nothing is required even with no cache.
        _check_embeddings(["/no/such/protein.cif"], "gvp", None)

    def test_missing_esm_embedding_raises_naming_files(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            _check_embeddings(["a/protein.cif", "b/other.pdb"], "esm", str(tmp_path))
        msg = str(exc.value)
        assert "protein.cif" in msg and "other.pdb" in msg

    def test_present_esm_embeddings_pass(self, tmp_path):
        emb = tmp_path / "esm"
        emb.mkdir()
        (emb / "protein.pt").touch()
        _check_embeddings(["some/dir/protein.cif"], "esm", str(tmp_path))


@pytest.mark.integration
class TestEndToEnd:
    @pytest.mark.parametrize("selection", ["confidence", "density"])
    def test_pipeline_writes_predicted_structure(
        self, selection, pdb_4h0b, gvp_encoder, tmp_path
    ):
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
            geometry_cache=None,
            include_mates=False,
            method="euler",
            num_steps=2,
            water_ratio=1.0,
            selection=selection,
            # Permissive settings so waters survive and the write path runs.
            confidence_threshold=0.0 if selection == "confidence" else None,
            density_ratio=1.0 if selection == "density" else None,
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

    def test_geometry_cache_writes_and_reuses(self, pdb_4h0b, gvp_encoder, tmp_path):
        """--geometry_cache stores the flow input graph (<name>.pt) and candidate
        waters (candidates/<name>.pt); a second run reuses them, so its predicted
        waters are identical instead of freshly sampled."""
        device = torch.device("cpu")
        flow_model = FlowWaterGVP(
            encoder=gvp_encoder, hidden_dims=(64, 8), layers=1
        ).to(device)
        flow_model.eval()
        flow_matcher = FlowMatcher(model=flow_model, sampling_strategy="uniform_ball")
        conf_model = build_confidence_model(
            {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1},
            device,
        )
        conf_model.eval()  # deterministic scores, so reuse yields identical output
        cache = tmp_path / "geo_cache"

        def run(out_dir):
            args = SimpleNamespace(
                processed_dir=None,
                geometry_cache=str(cache),
                include_mates=False,
                method="euler",
                num_steps=2,
                water_ratio=1.0,
                selection="confidence",
                confidence_threshold=0.0,
                density_ratio=None,
                out_dir=str(out_dir),
                out_format=".pdb",
            )
            predict_structures(
                [pdb_4h0b], flow_matcher, conf_model, {"encoder_type": "gvp"}, args, device
            )

        run(tmp_path / "out1")
        # include_mates=False -> no suffix on the cached names.
        graph_pt = cache / "4h0b_final.pt"
        cand_pt = cache / "candidates" / "4h0b_final.pt"
        assert graph_pt.exists(), "flow-input graph not cached"
        assert cand_pt.exists(), "candidate waters not cached"
        assert "candidate_pos" in torch.load(cand_pt, weights_only=False)

        # Second run reuses the cached candidates -> identical predicted waters.
        run(tmp_path / "out2")
        r1 = np.loadtxt(tmp_path / "out1" / "4h0b_final_waters.txt").reshape(-1, 4)
        r2 = np.loadtxt(tmp_path / "out2" / "4h0b_final_waters.txt").reshape(-1, 4)
        assert r1.shape == r2.shape and np.allclose(r1, r2, atol=1e-4)

    def test_geometry_cache_separates_mates(self, pdb_4h0b, gvp_encoder, tmp_path):
        """mates and mates_off share one --geometry_cache dir without colliding:
        mates runs write <name>_mates.pt, mates_off runs write <name>.pt, so
        neither reuses the other's cached graph or candidates."""
        device = torch.device("cpu")
        flow_model = FlowWaterGVP(
            encoder=gvp_encoder, hidden_dims=(64, 8), layers=1
        ).to(device)
        flow_model.eval()
        flow_matcher = FlowMatcher(model=flow_model, sampling_strategy="uniform_ball")
        conf_model = build_confidence_model(
            {"encoder_type": "gvp", "hidden_s": 64, "hidden_v": 8, "flow_layers": 1},
            device,
        )
        conf_model.eval()
        cache = tmp_path / "geo_cache"

        def run(include_mates, out_dir):
            args = SimpleNamespace(
                processed_dir=None,
                geometry_cache=str(cache),
                include_mates=include_mates,
                method="euler",
                num_steps=2,
                water_ratio=1.0,
                selection="confidence",
                confidence_threshold=0.0,
                density_ratio=None,
                out_dir=str(out_dir),
                out_format=".pdb",
            )
            predict_structures(
                [pdb_4h0b], flow_matcher, conf_model, {"encoder_type": "gvp"}, args, device
            )

        run(False, tmp_path / "off")
        run(True, tmp_path / "on")
        assert (cache / "4h0b_final.pt").exists()
        assert (cache / "4h0b_final_mates.pt").exists()
        assert (cache / "candidates" / "4h0b_final.pt").exists()
        assert (cache / "candidates" / "4h0b_final_mates.pt").exists()
