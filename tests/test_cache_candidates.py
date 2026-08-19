"""Unit tests for scripts/cache_candidates.py -- candidate cache generation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.cache_candidates as cc


class _FakeDataset:
    """Flow-dataset stand-in: `.entries` with cache keys and indexable graphs whose
    `pdb_id` matches, exactly as ProteinWaterDataset exposes them."""

    def __init__(self, keys):
        self.entries = [{"cache_key": k} for k in keys]
        self._graphs = [SimpleNamespace(pdb_id=k) for k in keys]

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx]


def _sampler(n_cand):
    """A `sample_batch` stand-in: a fixed-size candidate set per structure."""

    def sample_batch(graphs):
        return [
            {"pdb_id": g.pdb_id, "water_pred": np.zeros((n_cand, 3), dtype=np.float32)}
            for g in graphs
        ]

    return sample_batch


# Static run-info params; their exact values are opaque to _write_candidate_cache.
_STATIC = {"flow_run_dir": "run", "checkpoint": "best.pt", "epoch": 7}


@pytest.mark.unit
class TestWriteCandidateCache:
    def test_writes_thin_candidate_files(self, tmp_path):
        out = tmp_path / "cand"
        stats = cc._write_candidate_cache(
            _FakeDataset(["a_final", "b_final"]),
            _sampler(4),
            out,
            run_info=_STATIC,
            batch_size=1,
        )

        assert stats["n_written"] == 2
        assert (out / "a_final.pt").exists() and (out / "b_final.pt").exists()
        payload = torch.load(out / "a_final.pt", weights_only=True)
        assert payload["candidate_pos"].shape == (4, 3)
        assert payload["candidate_pos"].dtype == torch.float32
        # generation.json records run_info plus this run's counts
        info = json.loads((out / "generation.json").read_text())
        assert info["n_written"] == 2 and info["n_total"] == 2
        assert info["checkpoint"] == "best.pt"

    def test_skips_existing_without_overwrite(self, tmp_path):
        out = tmp_path / "cand"
        out.mkdir()
        torch.save({"candidate_pos": torch.ones(9, 3)}, out / "a_final.pt")

        stats = cc._write_candidate_cache(
            _FakeDataset(["a_final"]),
            _sampler(2),
            out,
            run_info=_STATIC,
        )

        assert stats["n_written"] == 0 and stats["n_skipped"] == 1
        untouched = torch.load(out / "a_final.pt", weights_only=True)
        assert untouched["candidate_pos"].shape == (9, 3)

    def test_overwrite_regenerates(self, tmp_path):
        out = tmp_path / "cand"
        out.mkdir()
        torch.save({"candidate_pos": torch.ones(9, 3)}, out / "a_final.pt")

        stats = cc._write_candidate_cache(
            _FakeDataset(["a_final"]),
            _sampler(2),
            out,
            run_info=_STATIC,
            overwrite=True,
        )

        assert stats["n_written"] == 1
        regenerated = torch.load(out / "a_final.pt", weights_only=True)
        assert regenerated["candidate_pos"].shape == (2, 3)


@pytest.mark.unit
class TestDefaultCandidateDir:
    def test_namespaced_by_sampling_inputs(self, tmp_path):
        out = cc.default_candidate_dir(
            tmp_path, "/runs/my_run", "best.pt", 3.0, 1, "euler", 100
        )

        assert out == tmp_path / "candidate_cache" / "my_run_best_euler100_r3_s1"

    def test_distinct_configs_never_collide(self, tmp_path):
        dirs = {
            cc.default_candidate_dir(tmp_path, run, ckpt, ratio, seed, method, steps)
            for run in ("/runs/a", "/runs/b")
            for ckpt in ("best.pt", "epoch_50.pt")
            for ratio in (2.0, 3.0)
            for seed in (0, 1)
            for method in ("euler", "rk4")
            for steps in (50, 100)
        }

        assert len(dirs) == 2 * 2 * 2 * 2 * 2 * 2
