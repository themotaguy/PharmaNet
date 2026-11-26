"""A solver-independent network design solution and its cost.

Both the MILP and the greedy baseline return a ``Solution``, so they are
validated by the same checks and costed by the same function. That is what
makes the MILP-vs-baseline comparison fair: neither method grades itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pharmanet.optimization.instance import DRK, PDK, RK, NetworkInstance


@dataclass(frozen=True)
class SolveStats:
    """Solver diagnostics. Bounds are for minimisation: bound <= objective."""

    termination: str
    objective: float
    best_bound: float | None
    mip_gap: float | None
    wall_time_s: float
    n_variables: int
    n_binary: int
    n_constraints: int
    lp_bound: float | None = None  # LP relaxation of the model as solved
    lp_bound_weak: float | None = None  # LP relaxation without strong linking (C6)

    @staticmethod
    def _gap(obj: float, bound: float | None) -> float | None:
        return None if bound is None or obj == 0 else (obj - bound) / abs(obj)

    @property
    def integrality_gap(self) -> float | None:
        return self._gap(self.objective, self.lp_bound)

    @property
    def integrality_gap_weak(self) -> float | None:
        return self._gap(self.objective, self.lp_bound_weak)

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["integrality_gap"] = self.integrality_gap
        d["integrality_gap_weak"] = self.integrality_gap_weak
        return d


@dataclass(frozen=True)
class Solution:
    method: str  # "milp", "greedy", "greedy_dcs_optimal_flows", ...
    open_dcs: frozenset[str]
    x: dict[PDK, float]  # plant -> DC flow (non-zero entries only)
    z: dict[DRK, float]  # DC -> region flow
    u: dict[RK, float]  # unmet demand
    stats: SolveStats | None = None
    assignment: dict[str, str] | None = field(default=None)  # region -> DC if single-sourced


def cost_breakdown(inst: NetworkInstance, sol: Solution) -> dict[str, float]:
    """Recompute every objective term from the flows, independent of any solver."""
    fixed = sum(inst.fixed_cost[d] for d in sol.open_dcs)
    inbound = sum(inst.c_in[key] * v for key, v in sol.x.items())
    outbound = sum(inst.c_out[key] * v for key, v in sol.z.items())
    penalty = sum(inst.penalty[k] * v for (_, k), v in sol.u.items())
    return {
        "fixed": fixed,
        "inbound": inbound,
        "outbound": outbound,
        "penalty": penalty,
        "total": fixed + inbound + outbound + penalty,
    }


def service_level(inst: NetworkInstance, sol: Solution) -> float:
    """Share of design demand delivered (fill rate), in [0, 1]."""
    total = inst.total_demand()
    return 1.0 if total == 0 else 1.0 - sum(sol.u.values()) / total


def solution_frames(sol: Solution) -> dict[str, pd.DataFrame]:
    """Long tables for parquet output."""

    def frame(d: dict[Any, float], cols: list[str], val: str) -> pd.DataFrame:
        rows = [(*key, v) for key, v in sorted(d.items())]
        return pd.DataFrame(rows, columns=[*cols, val])

    return {
        "open_dcs": pd.DataFrame({"dc_id": sorted(sol.open_dcs)}),
        "flows_inbound": frame(sol.x, ["plant_id", "dc_id", "product_id"], "flow"),
        "flows_outbound": frame(sol.z, ["dc_id", "region_id", "product_id"], "flow"),
        "unmet": frame(sol.u, ["region_id", "product_id"], "unmet"),
    }
