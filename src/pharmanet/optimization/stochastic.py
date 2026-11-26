"""Two-stage stochastic network design (extensive form). See docs/formulation.md.

First stage: y_d (open DCs), shared by every scenario. Second stage: flows
x^s, z^s, u^s per scenario. Objective: fixed cost + probability-weighted
recourse cost. Solved as one MILP (the deterministic equivalent). With only
|D| binaries, branch-and-bound stays small even when the LP part is large.

Also provides the standard evaluation measures:
    RP  = optimal stochastic objective
    EEV = expected cost of a given (e.g. deterministic) design on the scenarios
    WS  = expected cost with perfect foresight (best design per scenario)
    VSS = EEV - RP >= 0,   EVPI = RP - WS >= 0
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from pharmanet.config import OptimizationConfig
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.network_design import (
    FLOW_TOL,
    ModelOptions,
    model_size,
    solve_model,
    solve_network_design,
)
from pharmanet.optimization.solution import Solution, cost_breakdown
from pharmanet.optimization.validate import validate_solution
from pharmanet.parallel import parallel_map

PROB_TOL = 1e-9


@dataclass(frozen=True)
class WeightedScenario:
    name: str
    probability: float
    instance: NetworkInstance


@dataclass(frozen=True)
class StochasticResult:
    open_dcs: frozenset[str]
    objective: float  # RP
    fixed_cost: float
    expected_recourse: float
    scenario_costs: pd.DataFrame  # name, probability, recourse cost, unmet, service level
    stats: dict[str, Any]


def _check_scenarios(scenarios: list[WeightedScenario]) -> NetworkInstance:
    if not scenarios:
        raise ValueError("need at least one scenario")
    total = sum(s.probability for s in scenarios)
    if abs(total - 1.0) > 1e-6 or any(s.probability < 0 for s in scenarios):
        raise ValueError(f"scenario probabilities must be >= 0 and sum to 1 (got {total})")
    base = scenarios[0].instance
    for s in scenarios[1:]:
        i = s.instance
        same = (i.plants, i.dcs, i.regions, i.products) == (
            base.plants, base.dcs, base.regions, base.products
        )  # fmt: skip
        if not same or i.fixed_cost != base.fixed_cost:
            raise ValueError(f"scenario {s.name} has different sets or fixed costs")
    return base


def build_extensive_form(
    scenarios: list[WeightedScenario], opts: ModelOptions
) -> pyo.ConcreteModel:
    """One MILP: shared y, per-scenario flows. Names mirror the deterministic model."""
    base = _check_scenarios(scenarios)
    inst = {i: s.instance for i, s in enumerate(scenarios)}
    prob = {i: s.probability for i, s in enumerate(scenarios)}

    m = pyo.ConcreteModel(name="stochastic_network_design")
    m.S = pyo.Set(initialize=list(inst), ordered=True)
    m.P = pyo.Set(initialize=base.plants, ordered=True)
    m.D = pyo.Set(initialize=base.dcs, ordered=True)
    m.R = pyo.Set(initialize=base.regions, ordered=True)
    m.K = pyo.Set(initialize=base.products, ordered=True)
    cold = [k for k in base.products if k in base.cold_products]

    m.y = pyo.Var(m.D, domain=pyo.Binary)
    m.x = pyo.Var(m.S, m.P, m.D, m.K, domain=pyo.NonNegativeReals)
    m.z = pyo.Var(m.S, m.D, m.R, m.K, domain=pyo.NonNegativeReals)
    m.u = pyo.Var(m.S, m.R, m.K, domain=pyo.NonNegativeReals)

    m.cost_fixed = pyo.Expression(expr=sum(base.fixed_cost[d] * m.y[d] for d in m.D))

    def recourse(m: pyo.ConcreteModel, s: int) -> Any:
        i = inst[s]
        return (
            sum(i.c_in[p, d, k] * m.x[s, p, d, k] for p in m.P for d in m.D for k in m.K)
            + sum(i.c_out[d, r, k] * m.z[s, d, r, k] for d in m.D for r in m.R for k in m.K)
            + sum(i.penalty[k] * m.u[s, r, k] for r in m.R for k in m.K)
        )

    m.recourse = pyo.Expression(m.S, rule=recourse)
    m.total_cost = pyo.Objective(
        expr=m.cost_fixed + sum(prob[s] * m.recourse[s] for s in m.S), sense=pyo.minimize
    )

    m.demand = pyo.Constraint(
        m.S, m.R, m.K,
        rule=lambda m, s, r, k: sum(m.z[s, d, r, k] for d in m.D) + m.u[s, r, k]
        == inst[s].dem[r, k],
    )  # fmt: skip
    m.flow_balance = pyo.Constraint(
        m.S, m.D, m.K,
        rule=lambda m, s, d, k: sum(m.x[s, p, d, k] for p in m.P)
        == sum(m.z[s, d, r, k] for r in m.R),
    )  # fmt: skip
    m.plant_capacity = pyo.Constraint(
        m.S, m.P, m.K,
        rule=lambda m, s, p, k: sum(m.x[s, p, d, k] for d in m.D)
        <= inst[s].cap_pk[p, k] * inst[s].qualified[p, k],
    )  # fmt: skip
    m.dc_throughput = pyo.Constraint(
        m.S, m.D,
        rule=lambda m, s, d: sum(m.z[s, d, r, k] for r in m.R for k in m.K)
        <= inst[s].cap_d[d] * m.y[d],
    )  # fmt: skip
    m.dc_cold = pyo.Constraint(
        m.S, m.D,
        rule=lambda m, s, d: sum(m.z[s, d, r, k] for r in m.R for k in cold)
        <= inst[s].cold_d[d] * m.y[d],
    )  # fmt: skip
    if opts.strong_linking:
        m.strong_link = pyo.Constraint(
            m.S, m.D, m.R, m.K,
            rule=lambda m, s, d, r, k: m.z[s, d, r, k] <= inst[s].dem[r, k] * m.y[d],
        )  # fmt: skip
    if opts.max_open_dcs is not None:
        m.max_open_dcs = pyo.Constraint(expr=sum(m.y[d] for d in m.D) <= opts.max_open_dcs)
    for s, i in inst.items():
        for p, d, k in i.blocked_in:
            m.x[s, p, d, k].fix(0.0)
        for d, r, k in i.blocked_out:
            m.z[s, d, r, k].fix(0.0)
    return m


def solve_stochastic(
    scenarios: list[WeightedScenario], cfg: OptimizationConfig
) -> StochasticResult:
    """Solve the extensive form; validate every scenario's flows independently.

    Single-sourcing and N-1 options are not supported here (they are design
    rules, orthogonal to the stochastic comparison); a ValueError says so.
    """
    if cfg.single_sourcing or cfg.n_minus_1_dc:
        raise ValueError("stochastic design supports neither single_sourcing nor n_minus_1_dc")
    opts = ModelOptions(strong_linking=cfg.strong_linking, max_open_dcs=cfg.max_open_dcs)
    t0 = time.perf_counter()
    m = build_extensive_form(scenarios, opts)
    build_s = time.perf_counter() - t0
    term, bound, wall = solve_model(m, cfg.solver)
    obj = float(pyo.value(m.total_cost))
    open_dcs = frozenset(d for d in m.D if (m.y[d].value or 0.0) > 0.5)

    rows = []
    expected = 0.0
    for s, sc in enumerate(scenarios):

        def nz(var: Any, s: int = s) -> dict[Any, float]:
            return {
                idx[1:]: float(v.value)
                for idx, v in var.items()
                if idx[0] == s and (v.value or 0.0) > FLOW_TOL
            }

        sol = Solution("stochastic", open_dcs, nz(m.x), nz(m.z), nz(m.u))
        costs = validate_solution(sc.instance, sol, max_open_dcs=cfg.max_open_dcs)
        rec = costs["total"] - costs["fixed"]
        expected += sc.probability * rec
        unmet = sum(sol.u.values())
        rows.append({"scenario": sc.name, "probability": sc.probability, "recourse_cost": rec,
                     "unmet": unmet,
                     "service_level": 1 - unmet / sc.instance.total_demand()})  # fmt: skip

    fixed = sum(scenarios[0].instance.fixed_cost[d] for d in open_dcs)
    if abs(fixed + expected - obj) > 1e-4 + 1e-6 * abs(obj):
        raise AssertionError(f"RP mismatch: solver {obj:.4f} vs recomputed {fixed + expected:.4f}")
    gap = None if bound is None or obj == 0 else max(0.0, (obj - bound) / abs(obj))
    stats = {"termination": term, "build_time_s": build_s, "solve_time_s": wall,
             "mip_gap": gap, "n_scenarios": len(scenarios), **model_size(m)}  # fmt: skip
    return StochasticResult(open_dcs, obj, fixed, expected, pd.DataFrame(rows), stats)


# ----------------------------------------------------------------------- evaluation
def _design_cost_one(
    args: tuple[WeightedScenario, frozenset[str], OptimizationConfig],
) -> dict[str, Any]:
    sc, dcs, cfg = args
    fixed = {d: int(d in dcs) for d in sc.instance.dcs}
    opts = ModelOptions(strong_linking=cfg.strong_linking, fixed_open=fixed)
    sol = solve_network_design(sc.instance, cfg, opts=opts, compute_lp_bounds=False)
    validate_solution(sc.instance, sol)
    costs = cost_breakdown(sc.instance, sol)
    unmet = sum(sol.u.values())
    return {"scenario": sc.name, "probability": sc.probability, "total_cost": costs["total"],
            "recourse_cost": costs["total"] - costs["fixed"], "unmet": unmet,
            "service_level": 1 - unmet / sc.instance.total_demand()}  # fmt: skip


def evaluate_design_on(
    scenarios: list[WeightedScenario],
    open_dcs: frozenset[str],
    cfg: OptimizationConfig,
    n_jobs: int | None,
) -> pd.DataFrame:
    """Per-scenario cost of a fixed design (flows re-optimized). EEV = sum p * total_cost."""
    _check_scenarios(scenarios)
    return pd.DataFrame(
        parallel_map(_design_cost_one, [(s, open_dcs, cfg) for s in scenarios], n_jobs)
    )


def _ws_one(args: tuple[WeightedScenario, OptimizationConfig]) -> dict[str, Any]:
    sc, cfg = args
    opts = ModelOptions(strong_linking=cfg.strong_linking, max_open_dcs=cfg.max_open_dcs)
    sol = solve_network_design(sc.instance, cfg, opts=opts, compute_lp_bounds=False)
    validate_solution(sc.instance, sol, max_open_dcs=cfg.max_open_dcs)
    return {"scenario": sc.name, "probability": sc.probability,
            "ws_cost": cost_breakdown(sc.instance, sol)["total"],
            "ws_open_dcs": ",".join(sorted(sol.open_dcs))}  # fmt: skip


def wait_and_see(
    scenarios: list[WeightedScenario], cfg: OptimizationConfig, n_jobs: int | None
) -> pd.DataFrame:
    """Best design per scenario as if it were known in advance. WS = sum p * ws_cost."""
    _check_scenarios(scenarios)
    return pd.DataFrame(parallel_map(_ws_one, [(s, cfg) for s in scenarios], n_jobs))


def expected(df: pd.DataFrame, col: str) -> float:
    return float((df["probability"] * df[col]).sum())
