"""Unit tests for src/confidence_dataset.py -- candidate join and target computation."""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.confidence import smootherstep_target
from src.confidence_dataset import _oxygen_features, ConfidenceDataset
from src.constants import EDGE_PP, ELEM_IDX, NUM_RBF
from src.dataset import element_onehot


class _FakeFlowDataset:
    """Minimal stand-in for ProteinWaterDataset: exposes `.entries` and yields
    flow HeteroData with GT waters at `data["water"].pos` and `data.pdb_id`."""

    def __init__(self, keys, gt_by_key=None, n_prot=6, n_gt=3, embedding_dim=None):
        self.entries = [{"cache_key": k} for k in keys]
        self._gt_by_key = gt_by_key or {}
        self._n_prot = n_prot
        self._n_gt = n_gt
        self._embedding_dim = embedding_dim

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        key = self.entries[idx]["cache_key"]
        data = HeteroData()
        data["protein"].x = F.one_hot(
            torch.randint(0, 16, (self._n_prot,)), num_classes=16
        ).float()
        data["protein"].pos = torch.randn(self._n_prot, 3)
        data["protein"].residue_index = torch.arange(self._n_prot, dtype=torch.long)
        data["protein"].is_ligand = torch.zeros(self._n_prot, dtype=torch.bool)
        data["protein"].num_nodes = self._n_prot
        if self._embedding_dim is not None:
            data["protein"].embedding = torch.randn(self._n_prot, self._embedding_dim)
            data["protein"].embedding_type = "esm"

        gt = self._gt_by_key.get(key)
        if gt is None:
            gt = torch.randn(self._n_gt, 3)
        data["water"].pos = gt
        data["water"].x = F.one_hot(
            torch.full((gt.size(0),), ELEM_IDX["O"]), num_classes=16
        ).float()
        data["water"].num_nodes = gt.size(0)

        data[EDGE_PP].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        data[EDGE_PP].edge_unit_vectors = torch.randn(3, 3)
        data[EDGE_PP].edge_rbf = torch.randn(3, NUM_RBF)
        data.pdb_id = key
        return data


def _write_candidate(path, cand):
    torch.save({"candidate_pos": cand}, path)


@pytest.mark.unit
class TestConfidenceDataset:
    def test_joins_candidates_and_computes_target(self, tmp_path):
        cand = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        gt = torch.tensor([[0.0, 0.0, 0.0]])
        flow = _FakeFlowDataset(["pdb_a"], gt_by_key={"pdb_a": gt}, embedding_dim=8)
        _write_candidate(tmp_path / "pdb_a.pt", cand)

        ds = ConfidenceDataset(flow, candidate_dir=tmp_path)
        assert len(ds) == 1
        sample = ds[0]

        # candidates became the scored water nodes
        assert sample["water"].num_nodes == 3
        assert torch.equal(sample["water"].pos, cand)
        assert (sample["water"].x[:, ELEM_IDX["O"]] == 1.0).all()  # oxygen one-hot

        # target on the fly: candidate on GT -> ~1; far candidate -> ~0
        tc = sample["water"].target_confidence
        assert tc.shape == (3,)
        assert tc[0].item() == pytest.approx(1.0, abs=1e-5)
        assert tc[2].item() < 0.05

        # protein graph + embedding + PP edges carried from the flow item
        assert sample["protein"].num_nodes == 6
        assert sample["protein"].embedding.shape == (6, 8)
        assert sample[EDGE_PP].edge_index.shape == (2, 3)
        assert sample.pdb_id == "pdb_a"

    def test_default_target_is_smootherstep(self, tmp_path):
        torch.manual_seed(1)
        cand = torch.randn(9, 3)
        gt = torch.randn(3, 3)
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": gt})
        _write_candidate(tmp_path / "k.pt", cand)
        ds = ConfidenceDataset(flow, candidate_dir=tmp_path, r_in=0.4, r_out=2.0)
        expected = smootherstep_target(cand, gt, r_in=0.4, r_out=2.0)
        assert torch.allclose(ds[0]["water"].target_confidence, expected, atol=1e-5)

    def test_empty_candidates(self, tmp_path):
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": torch.randn(2, 3)})
        _write_candidate(tmp_path / "k.pt", torch.empty(0, 3))
        ds = ConfidenceDataset(flow, candidate_dir=tmp_path)
        sample = ds[0]
        assert sample["water"].num_nodes == 0
        assert sample["water"].target_confidence.shape == (0,)

    def test_strict_missing_raises(self, tmp_path):
        flow = _FakeFlowDataset(["have", "gone"])
        _write_candidate(tmp_path / "have.pt", torch.randn(3, 3))
        with pytest.raises(FileNotFoundError):
            ConfidenceDataset(flow, candidate_dir=tmp_path, strict=True)

    def test_non_strict_filters_missing(self, tmp_path):
        flow = _FakeFlowDataset(["have", "gone"])
        _write_candidate(tmp_path / "have.pt", torch.randn(3, 3))
        ds = ConfidenceDataset(flow, candidate_dir=tmp_path, strict=False)
        assert len(ds) == 1
        assert ds[0].pdb_id == "have"

    def test_missing_candidate_dir_raises(self, tmp_path):
        flow = _FakeFlowDataset(["k"])
        with pytest.raises(FileNotFoundError):
            ConfidenceDataset(flow, candidate_dir=tmp_path / "missing")

    def test_invalid_parameters_raise(self, tmp_path):
        flow = _FakeFlowDataset(["k"])
        _write_candidate(tmp_path / "k.pt", torch.randn(2, 3))
        with pytest.raises(ValueError):
            ConfidenceDataset(flow, candidate_dir=tmp_path, r_in=1.5, r_out=0.5)
        with pytest.raises(ValueError):
            ConfidenceDataset(flow, candidate_dir=tmp_path, accept_radius=-1.0)
        with pytest.raises(ValueError):
            ConfidenceDataset(flow, candidate_dir=tmp_path, max_candidates=0)

    def test_no_candidates_at_all_raises(self, tmp_path):
        flow = _FakeFlowDataset(["k"])
        with pytest.raises(RuntimeError):
            ConfidenceDataset(flow, candidate_dir=tmp_path, strict=False)

    def test_flow_dataset_without_entries_raises(self, tmp_path):
        _write_candidate(tmp_path / "k.pt", torch.randn(2, 3))
        with pytest.raises(TypeError):
            ConfidenceDataset(object(), candidate_dir=tmp_path)

    def test_label_tracks_nearest_gt(self, tmp_path):
        # candidates at 0.5 A and 3 A from GT site 1; accept_radius default 1.0
        gt = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        cand = torch.tensor([[10.5, 0.0, 0.0], [13.0, 0.0, 0.0]])
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": gt})
        _write_candidate(tmp_path / "k.pt", cand)
        sample = ConfidenceDataset(flow, candidate_dir=tmp_path)[0]

        assert torch.equal(
            sample["water"].within_accept_radius, torch.tensor([1.0, 0.0])
        )

    def test_hard_label_replaces_the_soft_target(self, tmp_path):
        gt = torch.tensor([[0.0, 0.0, 0.0]])
        cand = torch.tensor([[0.8, 0.0, 0.0], [1.2, 0.0, 0.0]])
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": gt})
        _write_candidate(tmp_path / "k.pt", cand)

        soft = ConfidenceDataset(flow, candidate_dir=tmp_path)[0]
        hard = ConfidenceDataset(flow, candidate_dir=tmp_path, hard_label=True)[0]

        # soft target is strictly between the plateau and the floor at 0.8 A
        assert 0.0 < soft["water"].target_confidence[0].item() < 1.0
        assert torch.equal(
            hard["water"].target_confidence, hard["water"].within_accept_radius
        )
        assert torch.equal(hard["water"].target_confidence, torch.tensor([1.0, 0.0]))

    def test_accept_radius_widens_the_label(self, tmp_path):
        gt = torch.tensor([[0.0, 0.0, 0.0]])
        cand = torch.tensor([[1.5, 0.0, 0.0]])
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": gt})
        _write_candidate(tmp_path / "k.pt", cand)
        narrow = ConfidenceDataset(flow, candidate_dir=tmp_path)[0]
        wide = ConfidenceDataset(flow, candidate_dir=tmp_path, accept_radius=2.0)[0]
        assert narrow["water"].within_accept_radius.item() == 0.0
        assert wide["water"].within_accept_radius.item() == 1.0

    def test_max_candidates_subsamples_the_cloud(self, tmp_path):
        gt = torch.tensor([[0.0, 0.0, 0.0]])
        cand = torch.arange(20, dtype=torch.float32).reshape(20, 1).repeat(1, 3)
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": gt})
        _write_candidate(tmp_path / "k.pt", cand)

        ds = ConfidenceDataset(flow, candidate_dir=tmp_path, max_candidates=5)
        sample = ds[0]
        assert sample["water"].num_nodes == 5
        assert sample["water"].target_confidence.shape == (5,)
        # a subset of the candidate cloud, not a truncation
        rows = {tuple(r.tolist()) for r in sample["water"].pos}
        assert rows.issubset({tuple(r.tolist()) for r in cand})

        # A fresh draw each epoch: pin the RNG so the two reads are known to
        # subsample differently (an unseeded pair could collide, rarely).
        torch.manual_seed(0)
        first = ds[0]["water"].pos
        torch.manual_seed(1)
        second = ds[0]["water"].pos
        assert not torch.equal(first, second)

    def test_max_candidates_above_the_cloud_is_a_no_op(self, tmp_path):
        gt = torch.tensor([[0.0, 0.0, 0.0]])
        cand = torch.randn(4, 3)
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": gt})
        _write_candidate(tmp_path / "k.pt", cand)
        sample = ConfidenceDataset(flow, candidate_dir=tmp_path, max_candidates=10)[0]
        assert torch.equal(sample["water"].pos, cand)

    def test_no_gt_waters_yields_empty_targets(self, tmp_path):
        # A structure whose waters were all filtered out. Targets come back
        # empty while num_nodes stays at the candidate count -- such an item
        # cannot be collated, so the trainer must exclude these structures.
        flow = _FakeFlowDataset(["k"], gt_by_key={"k": torch.empty(0, 3)})
        _write_candidate(tmp_path / "k.pt", torch.randn(3, 3))
        sample = ConfidenceDataset(flow, candidate_dir=tmp_path)[0]
        assert sample["water"].num_nodes == 3
        assert sample["water"].target_confidence.shape == (0,)
        assert sample["water"].within_accept_radius.shape == (0,)

    def test_items_batch_via_pyg(self, tmp_path):
        """Different-sized candidate clouds must collate through
        Batch.from_data_list -- what a DataLoader does to train on many at once."""
        from torch_geometric.data import Batch

        gt = torch.tensor([[0.0, 0.0, 0.0]])
        flow = _FakeFlowDataset(
            ["a", "b"], gt_by_key={"a": gt, "b": gt}, embedding_dim=8
        )
        _write_candidate(tmp_path / "a.pt", torch.randn(3, 3))
        _write_candidate(tmp_path / "b.pt", torch.randn(5, 3))
        ds = ConfidenceDataset(flow, candidate_dir=tmp_path)

        batch = Batch.from_data_list([ds[0], ds[1]])

        assert batch.num_graphs == 2
        # candidate water nodes and their per-candidate targets concatenate (3 + 5)
        assert batch["water"].num_nodes == 8
        assert batch["water"].target_confidence.shape == (8,)
        assert batch["water"].within_accept_radius.shape == (8,)
        assert batch["water"].batch.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
        # protein graph and PP edges carried and offset per graph
        assert batch["protein"].num_nodes == 12
        assert batch[EDGE_PP].edge_index.shape[1] == 6


@pytest.mark.unit
class TestOxygenFeatures:
    def test_matches_element_onehot(self):
        assert torch.equal(_oxygen_features(4), element_onehot(["O"] * 4))

    def test_empty(self):
        feats = _oxygen_features(0)
        assert feats.shape == (0, 16)
        assert feats.dtype == torch.float32
