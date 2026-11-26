"""Process-based parallel map shared by the backtest and the scenario simulation."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Sequence[T], n_jobs: int | None) -> list[R]:
    """Map ``fn`` over items in worker processes (serially if n_jobs == 1).

    Used for embarrassingly parallel work (independent series or scenarios).
    Results come back in input order, so output is deterministic regardless of
    scheduling. ``fn`` must be a picklable top-level function.
    """
    workers = n_jobs or os.cpu_count() or 1
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunk))
