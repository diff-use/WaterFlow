# distributed.py
"""
Distributed (DDP) helpers shared by the training entry points.

DDP is activated purely by the launcher: `torchrun --nproc_per_node=N` sets
WORLD_SIZE / RANK / LOCAL_RANK. Without those env vars every helper degrades to
single-GPU behavior, so scripts run unchanged under `python -m scripts.train`.
Nothing here reads CLI arguments, which keeps a recorded config.json identical
whether the run used one GPU or eight.
"""

import os
from collections.abc import Callable
from datetime import timedelta

import torch
import torch.distributed as dist


def _ddp_world_size() -> int:
    """World size from the launcher env, or 1 when not launched by torchrun."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_is_active() -> bool:
    """True when running under a multi-process launcher."""
    return _ddp_world_size() > 1


def ddp_rank_and_world() -> tuple[int, int]:
    """
    This rank's index and the world size, from the live process group.

    Returns (0, 1) when not distributed, so callers can shard work by
    `i % world_size == rank` without branching.
    """
    if not ddp_is_active():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def setup_distributed(store: dist.Store | None = None) -> tuple[int, int, int]:
    """
    Initialize the NCCL process group if launched under torchrun.

    Args:
        store: Optional rendezvous store. Passing the CPU-side TCPStore from
            `run_once_on_main` builds the NCCL group on top of it instead of
            re-rendezvousing via env://, so one store serves both phases.

    Returns:
        (rank, local_rank, world_size); (0, 0, 1) when not distributed.
    """
    if not ddp_is_active():
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # device_id binds this rank to its GPU eagerly and verifies the rank->GPU
    # mapping at init, instead of inferring it from rank order.
    if store is not None:
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(
            backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
        )
    return dist.get_rank(), local_rank, dist.get_world_size()


def is_main_process(rank: int) -> bool:
    """True on the single rank that owns disk IO and W&B logging."""
    return rank == 0


def ddp_barrier() -> None:
    """Block until every rank arrives. No-op when not distributed."""
    if ddp_is_active():
        dist.barrier()


def teardown_distributed() -> None:
    """Destroy the process group. No-op when not distributed."""
    if ddp_is_active():
        dist.destroy_process_group()


def run_once_on_main(work: Callable[[], None], key: str) -> dist.Store | None:
    """
    Run `work` on rank 0 only, holding every other rank until it finishes.
    Primarily for dataset processing on 0 rank to prevent race conditions.

    Coordination runs on a CPU-side TCPStore rather than a GPU collective, so a
    long single-writer job never counts against an NCCL timeout. Call this before
    `setup_distributed` and hand the store back to it.

    Args:
        work: Executed on rank 0 only. An exception propagates on rank 0 and
            leaves the others blocked -- a hang, not a half-written result.
        key: Store key signalling completion. Must be unique per work item.

    Returns:
        The store, for `setup_distributed(store=...)`; None when not distributed
        (in which case `work` simply runs inline).
    """
    if not ddp_is_active():
        work()
        return None

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    # Client of torchrun's agent store (is_master=False) -- the agent already owns
    # MASTER_PORT. The timeout covers a cold build, which can take hours.
    store = dist.TCPStore(host, port, world, False, timeout=timedelta(hours=24))

    if rank == 0:
        work()
        store.set(key, "1")
    else:
        store.wait([key])
    return store


def all_reduce_means(
    sums: dict[str, float], count: float, device: torch.device
) -> tuple[dict[str, float], int]:
    """
    Average per-item metric sums across ranks.

    Reduces the metric sums and the item count in one collective, then divides.
    Summing before dividing is what keeps the mean correct when ranks processed
    unequal numbers of items; float64 keeps it insensitive to the order NCCL
    combines ranks in.

    Call this on every rank, including ones where `count` is 0 -- a rank that
    skips the collective hangs the rest. That is why an empty result is reported
    through the return value rather than an early exit.

    Args:
        sums: Metric name -> sum of that metric over this rank's items.
        count: Number of items this rank contributed to `sums`.
        device: Buffer device. Must be this rank's CUDA device under NCCL.

    Returns:
        (means, total_count). `means` is empty when `total_count` is 0.
    """
    keys = list(sums)
    totals = torch.tensor(
        [float(sums[k]) for k in keys] + [float(count)],
        dtype=torch.float64,
        device=device,
    )
    if ddp_is_active():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_count = int(totals[-1].item())
    if total_count == 0:
        return {}, 0
    return {k: (totals[i] / totals[-1]).item() for i, k in enumerate(keys)}, total_count
