"""
Tests for the DDP helpers and the single-writer cache prebuild.

These run single-process, so they cover the logic that decides *what* each rank
does -- sharding, the sum-then-divide reduction, sampler wiring, cache shard
disjointness -- not NCCL itself. The collective paths degrade to no-ops without a
launcher, which is exactly the property most of these assert.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.data import SequentialSampler

import scripts.train as train
from src.dataset import get_dataloader, ProteinWaterDataset
from src.distributed import (
    _ddp_world_size,
    all_reduce_means,
    ddp_barrier,
    ddp_is_active,
    ddp_rank_and_world,
    is_main_process,
    run_once_on_main,
    setup_distributed,
    teardown_distributed,
)
from src.flow import FlowMatcher


CPU = torch.device("cpu")


@pytest.fixture
def launcher_env(monkeypatch):
    """Factory that fakes a torchrun launch by setting WORLD_SIZE."""

    def _set(world_size):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))

    return _set


# ============== Launcher detection ==============


def test_world_size_defaults_to_one_without_launcher(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert _ddp_world_size() == 1
    assert ddp_is_active() is False


def test_single_process_launch_is_not_ddp(launcher_env):
    """torchrun --nproc_per_node=1 must take the plain single-GPU path."""
    launcher_env(1)
    assert ddp_is_active() is False


def test_world_size_above_one_activates_ddp(launcher_env):
    launcher_env(4)
    assert ddp_is_active() is True


def test_setup_distributed_is_inert_without_launcher(monkeypatch):
    """No process group is created, so a plain `python -m scripts.train` works."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert setup_distributed() == (0, 0, 1)


def test_rank_and_world_default_to_solo(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert ddp_rank_and_world() == (0, 1)


def test_barrier_and_teardown_are_noops_without_launcher(monkeypatch):
    """Both would raise on an uninitialized process group if they ran for real."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ddp_barrier()
    teardown_distributed()


def test_is_main_process_only_rank_zero():
    assert is_main_process(0) is True
    assert is_main_process(1) is False


# ============== Stride sharding ==============


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_stride_shard_covers_every_index_exactly_once(world_size):
    """
    The `i % world_size == rank` rule run_eval_sampling shards on.

    A structure evaluated twice would be double-counted in the all-reduced means;
    one evaluated zero times would be silently dropped.
    """
    n_structures = 17
    owners = [
        [i for i in range(n_structures) if i % world_size == rank]
        for rank in range(world_size)
    ]
    assert sorted(i for shard in owners for i in shard) == list(range(n_structures))


# ============== Gradient sync scheduling ==============


def _sync_steps(n_batches, accum_steps):
    return [
        step
        for step in range(n_batches)
        if train._needs_grad_sync(step, n_batches, accum_steps)
    ]


def test_every_step_syncs_without_accumulation():
    assert _sync_steps(5, 1) == [0, 1, 2, 3, 4]


def test_only_boundaries_sync_when_batches_divide_evenly():
    """No trailing window, so the intermediate micro-steps can skip the collective."""
    assert _sync_steps(8, 4) == [3, 7]


def test_trailing_partial_window_syncs_throughout():
    """
    10 batches at accum=4 leaves steps 8-9 in a window that still ends in a step().

    Letting those accumulate under no_sync would apply un-reduced gradients and
    drift the ranks apart for the rest of training.
    """
    assert _sync_steps(10, 4) == [3, 7, 8, 9]


@pytest.mark.parametrize("n_batches", [1, 2, 5, 7, 10, 33])
@pytest.mark.parametrize("accum_steps", [1, 2, 3, 4, 8])
def test_last_batch_of_an_epoch_always_syncs(n_batches, accum_steps):
    """The epoch's final optimizer.step() must never run on unsynced gradients."""
    assert train._needs_grad_sync(n_batches - 1, n_batches, accum_steps)


# ============== all_reduce_means ==============


def test_all_reduce_means_divides_sums_by_count():
    means, count = all_reduce_means({"a": 10.0, "b": 5.0}, 4, CPU)
    assert count == 4
    assert means == {"a": 2.5, "b": 1.25}


def test_all_reduce_means_reports_empty_instead_of_dividing_by_zero():
    """A rank with no items must still return, not raise -- callers branch on this."""
    assert all_reduce_means({"a": 0.0}, 0, CPU) == ({}, 0)


def test_all_reduce_means_matches_a_plain_mean():
    """Sum-then-divide reproduces the single-process average it replaced."""
    values = [0.31, 2.75, 1.5, 9.125, 0.0625]
    means, _ = all_reduce_means({"m": sum(values)}, len(values), CPU)
    assert means["m"] == pytest.approx(sum(values) / len(values), rel=1e-12)


def test_all_reduce_means_weights_by_item_count():
    """
    Reducing sums (not per-rank means) is what makes unequal shards correct.

    Two ranks holding 1 item at 10.0 and 3 items at 2.0 average to 4.0, not the
    6.0 that averaging their two means would give.
    """
    means, count = all_reduce_means({"m": 10.0 + 6.0}, 1 + 3, CPU)
    assert count == 4
    assert means["m"] == pytest.approx(4.0)


def test_all_reduce_means_preserves_key_order():
    """Keys and the reduced buffer are zipped positionally."""
    sums = {"z": 3.0, "a": 6.0, "m": 9.0}
    means, _ = all_reduce_means(sums, 3, CPU)
    assert list(means) == ["z", "a", "m"]
    assert means == {"z": 1.0, "a": 2.0, "m": 3.0}


# ============== Reading model config through the DDP wrapper ==============


def test_flow_matcher_reads_cutoff_through_a_wrapper():
    """
    DDP does not forward attribute lookups to the module it wraps.

    Reading `cutoff` off the wrapper would fall back to the 8.0 default and
    silently change the water prior's sampling radius under DDP only.
    """
    wrapped = SimpleNamespace(module=SimpleNamespace(cutoff=12.0))
    assert FlowMatcher(model=wrapped).graph_cutoff == 12.0


def test_flow_matcher_reads_cutoff_off_a_bare_model():
    assert FlowMatcher(model=SimpleNamespace(cutoff=12.0)).graph_cutoff == 12.0


# ============== run_once_on_main ==============


def test_run_once_on_main_runs_inline_without_launcher(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    calls = []
    store = run_once_on_main(lambda: calls.append("built"), key="k")
    assert calls == ["built"]
    assert store is None


# ============== get_dataloader sampler wiring ==============


@pytest.fixture
def pdb_list_file(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("6eey_final\n")
    return path


def _loader(pdb_list_file, tmp_path, pdb_base_dir, **kwargs):
    return get_dataloader(
        pdb_list_file=str(pdb_list_file),
        processed_dir=str(tmp_path / "processed"),
        base_pdb_dir=str(pdb_base_dir),
        num_workers=0,
        preprocess=False,
        **kwargs,
    )


def test_dataloader_has_no_sampler_by_default(pdb_list_file, tmp_path, pdb_base_dir):
    loader = _loader(pdb_list_file, tmp_path, pdb_base_dir, shuffle=True)
    assert not isinstance(loader.sampler, torch.utils.data.DistributedSampler)


def test_explicit_sampler_overrides_shuffle(pdb_list_file, tmp_path, pdb_base_dir):
    """
    DataLoader raises if both are set, so `shuffle` must yield to the sampler.

    Passing shuffle=True alongside a sampler is what a caller does by accident;
    it has to be tolerated rather than crash mid-run.
    """
    dataset = ProteinWaterDataset(
        pdb_list_file=str(pdb_list_file),
        processed_dir=str(tmp_path / "probe"),
        base_pdb_dir=str(pdb_base_dir),
        preprocess=False,
    )
    sampler = SequentialSampler(dataset)
    loader = _loader(
        pdb_list_file, tmp_path, pdb_base_dir, shuffle=True, sampler=sampler
    )
    assert loader.sampler is sampler


def test_distributed_flag_is_ignored_when_a_sampler_is_given(
    pdb_list_file, tmp_path, pdb_base_dir
):
    dataset = ProteinWaterDataset(
        pdb_list_file=str(pdb_list_file),
        processed_dir=str(tmp_path / "probe"),
        base_pdb_dir=str(pdb_base_dir),
        preprocess=False,
    )
    sampler = SequentialSampler(dataset)
    loader = _loader(
        pdb_list_file, tmp_path, pdb_base_dir, distributed=True, sampler=sampler
    )
    assert loader.sampler is sampler


def test_distributed_builds_a_distributed_sampler(
    pdb_list_file, tmp_path, pdb_base_dir, monkeypatch
):
    """
    distributed=True must produce a DistributedSampler with shuffle honored.

    DistributedSampler reads the process group, so stand in a world of one rather
    than initializing NCCL.
    """
    import torch.utils.data.distributed as dist_data

    monkeypatch.setattr(dist_data.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_data.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_data.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist_data.dist, "get_rank", lambda: 0)

    loader = _loader(
        pdb_list_file, tmp_path, pdb_base_dir, shuffle=True, distributed=True
    )
    assert isinstance(loader.sampler, torch.utils.data.DistributedSampler)
    assert loader.sampler.shuffle is True
    assert loader.sampler.drop_last is False


# ============== build_cache ==============


class _InlinePool:
    """Stand-in for a spawn Pool that runs starmap in-process."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap(self, fn, argsets):
        return [fn(*args) for args in argsets]


@pytest.fixture
def inline_pool(monkeypatch):
    monkeypatch.setattr(
        train.mp,
        "get_context",
        lambda method: SimpleNamespace(Pool=lambda n: _InlinePool()),
    )


@pytest.fixture
def cache_args(tmp_path, pdb_base_dir):
    """Real parsed args for a train+val pair with overlapping ids."""
    train_list = tmp_path / "train.txt"
    val_list = tmp_path / "val.txt"
    train_list.write_text("\n".join(f"e{i}_final" for i in range(6)) + "\n")
    val_list.write_text("e5_final\ne6_final\n")  # e5 overlaps the train list

    argv = [
        "train.py",
        "--train_list",
        str(train_list),
        "--val_list",
        str(val_list),
        "--processed_dir",
        str(tmp_path / "cache"),
        "--base_pdb_dir",
        str(pdb_base_dir),
    ]
    with mock.patch.object(sys, "argv", argv):
        return train.parse_args()


def _geometry_dir(args):
    """Where build_cache's probe will look for cached geometry."""
    dataset_kwargs, _, _ = train._build_dataset_config(args)
    probe = ProteinWaterDataset(
        pdb_list_file=args.val_list,
        processed_dir=args.processed_dir,
        preprocess=False,
        **dataset_kwargs,
    )
    return probe.geometry_dir


@pytest.fixture
def recorded_shards(monkeypatch):
    """
    Capture the cache keys each shard was handed, one list per worker call.

    Read eagerly: build_cache deletes its tmpdir before returning, so the shard
    files are gone by the time the test body runs.
    """
    recorded = []

    def _record(list_file, processed_dir, dataset_kwargs):
        recorded.append([line for line in Path(list_file).read_text().split() if line])

    monkeypatch.setattr(train, "_build_cache_shard", _record)
    return recorded


def test_build_cache_shards_are_disjoint_and_complete(
    cache_args, recorded_shards, inline_pool, monkeypatch
):
    """Every missing key is built exactly once; workers never race on a file."""
    monkeypatch.setattr(train.os, "cpu_count", lambda: 3)
    train.build_cache(cache_args)

    shards = recorded_shards
    assert len(shards) == 3
    built = [key for shard in shards for key in shard]
    assert sorted(built) == [f"e{i}_final" for i in range(7)]
    assert len(built) == len(set(built))


def test_build_cache_deduplicates_ids_across_train_and_val(
    cache_args, recorded_shards, inline_pool, monkeypatch
):
    """e5 appears in both lists but must only be built once."""
    monkeypatch.setattr(train.os, "cpu_count", lambda: 3)
    train.build_cache(cache_args)

    built = [key for shard in recorded_shards for key in shard]
    assert built.count("e5_final") == 1


def test_build_cache_skips_already_cached_entries(
    cache_args, recorded_shards, inline_pool, monkeypatch
):
    monkeypatch.setattr(train.os, "cpu_count", lambda: 2)
    geometry_dir = _geometry_dir(cache_args)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (geometry_dir / f"e{i}_final.pt").touch()

    train.build_cache(cache_args)

    built = [key for shard in recorded_shards for key in shard]
    assert sorted(built) == ["e5_final", "e6_final"]


def test_build_cache_is_a_noop_on_a_warm_cache(
    cache_args, recorded_shards, inline_pool
):
    """A warm cache must not spawn a pool at all -- this runs every job's startup."""
    geometry_dir = _geometry_dir(cache_args)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    for i in range(7):
        (geometry_dir / f"e{i}_final.pt").touch()

    train.build_cache(cache_args)
    assert recorded_shards == []


def test_build_cache_runs_single_shard_without_a_pool(
    cache_args, recorded_shards, monkeypatch
):
    """
    With one usable core the pool is skipped entirely.

    No inline_pool fixture here: reaching mp.get_context would spawn real workers.
    """
    monkeypatch.setattr(train.os, "cpu_count", lambda: 1)
    train.build_cache(cache_args)

    assert len(recorded_shards) == 1
    assert sorted(recorded_shards[0]) == [f"e{i}_final" for i in range(7)]


def test_build_cache_handles_empty_lists(cache_args, recorded_shards, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    cache_args.train_list = str(empty)
    cache_args.val_list = str(empty)

    train.build_cache(cache_args)
    assert recorded_shards == []
