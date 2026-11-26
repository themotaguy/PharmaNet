"""Evaluate a fixed network design under scenarios, and run sensitivity analysis.

DC decisions are strategic (a warehouse cannot be opened overnight), flows are
operational. So for every scenario we **fix y to the base design and
re-optimize only the flows**. With all binaries fixed the MILP is an LP, so
each scenario solves in ~0.1 s. Every scenario solution is validated.

Sensitivity analysis is different: there we *re-design* (full MILP) under
changed parameters, and also price the base design under those parameters.
The difference is the **regret** of not re-optimizing:

    regret = cost(base design | new params) - cost(best design | new params) >= 0
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from pharmanet.config import OptimizationConfig, SimulationConfig
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.network_design import ModelOptions, solve_network_design
from pharmanet.optimization.solution import Solution, cost_breakdown
from pharmanet.optimization.validate import validate_solution
from pharmanet.parallel import parallel_map
from pharmanet.simulation.scenarios import Scenario


def evaluate_design(
    inst: NetworkInstance, open_dcs: frozenset[str], cfg: OptimizationConfig
) -> Solution:
    """Optimal flows for a fixed set of open DCs (an LP), validated."""
    fixed = {d: int(d in open_dcs) for d in inst.dcs}
    # max_open_dcs is irrelevant once y is fixed; single-sourcing still applies.
    opts = replace(ModelOptions.from_config(cfg), fixed_open=fixed, max_open_dcs=None)
    sol = solve_network_design(inst, cfg, opts=opts, compute_lp_bounds=False, method="fixed")
    validate_solution(inst, sol, single_sourcing=cfg.single_sourcing)
    return sol


def _metrics(inst: NetworkInstance, sol: Solution) -> dict[str, Any]:
    costs = cost_breakdown(inst, sol)
    demand = inst.total_demand()
    unmet = sum(sol.u.values())
    cold_unmet = sum(v for (_, k), v in sol.u.items() if k in inst.cold_products)
    return {
        **{f"cost_{k}": v for k, v in costs.items()},
        "logistics_cost": costs["total"] - costs["penalty"],
        "logistics_cost_per_unit": (costs["total"] - costs["penalty"]) / (demand - unmet)
        if demand - unmet > 0
        else float("nan"),
        "demand": demand,
        "unmet": unmet,
        "cold_unmet": cold_unmet,
        "service_level": 1.0 - unmet / demand if demand else 1.0,
        "regions_affected": len({r for (r, _), v in sol.u.items() if v > 1e-6}),
    }


def _evaluate_one(
    args: tuple[Scenario, frozenset[str], OptimizationConfig],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sc, open_dcs, cfg = args
    sol = evaluate_design(sc.instance, open_dcs, cfg)
    row = {"scenario_id": sc.scenario_id, "kind": sc.kind, "target": sc.target}
    row.update(_metrics(sc.instance, sol))
    unmet = [
        {"scenario_id": sc.scenario_id, "region_id": r, "product_id": k, "unmet": v,
         "demand": sc.instance.dem[(r, k)]}
        for (r, k), v in sorted(sol.u.items())
        if v > 1e-6
    ]  # fmt: skip
    return row, unmet


def evaluate_scenarios(
    scenarios: list[Scenario],
    open_dcs: frozenset[str],
    cfg: OptimizationConfig,
    n_jobs: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every scenario; returns (one row per scenario, unmet detail rows)."""
    results = parallel_map(_evaluate_one, [(s, open_dcs, cfg) for s in scenarios], n_jobs)
    rows = pd.DataFrame([r for r, _ in results])
    detail = pd.DataFrame(
        [u for _, us in results for u in us],
        columns=["scenario_id", "region_id", "product_id", "unmet", "demand"],
    )
    return rows, detail


def summarize_monte_carlo(rows: pd.DataFrame, base_cost: float) -> dict[str, Any]:
    """Distribution of cost and service across demand scenarios."""
    c, s = rows["cost_total"], rows["service_level"]
    q = [0.05, 0.5, 0.95]
    return {
        "n_scenarios": len(rows),
        "base_cost_at_forecast": base_cost,
        "cost_mean": float(c.mean()),
        "cost_p5_p50_p95": [float(v) for v in c.quantile(q)],
        "logistics_cost_p5_p50_p95": [float(v) for v in rows["logistics_cost"].quantile(q)],
        "service_level_mean": float(s.mean()),
        "service_level_p5": float(s.quantile(0.05)),
        "service_level_min": float(s.min()),
        "prob_any_shortfall": float((rows["unmet"] > 1e-6).mean()),
        "prob_cold_shortfall": float((rows["cold_unmet"] > 1e-6).mean()),
    }


def summarize_disruptions(rows: pd.DataFrame, base_cost_per_unit: float) -> list[dict[str, Any]]:
    """Disruptions ranked by unmet demand.

    The headline impact is **unmet units / service level**. Total cost includes
    the unmet-demand penalty (an artificial device that would dominate any
    percentage), and total logistics cost *falls* when less is shipped, so the
    cost impact is reported as the change in logistics cost **per delivered
    unit** versus the undisrupted design at forecast demand.
    """
    change = 100 * (rows["logistics_cost_per_unit"] - base_cost_per_unit) / base_cost_per_unit
    df = rows.assign(cost_per_unit_change_pct=change).sort_values(
        ["unmet", "cost_per_unit_change_pct"], ascending=False
    )
    cols = ["scenario_id", "kind", "target", "unmet", "service_level", "cold_unmet",
            "regions_affected", "cost_per_unit_change_pct"]  # fmt: skip
    return [{str(k): v for k, v in rec.items()} for rec in df[cols].to_dict(orient="records")]


# ------------------------------------------------------------------- sensitivity
def _sensitivity_one(
    args: tuple[str, float, NetworkInstance, frozenset[str], OptimizationConfig],
) -> dict[str, Any]:
    param, mult, inst, base_dcs, cfg = args
    redesign = solve_network_design(inst, cfg, compute_lp_bounds=False, method="redesign")
    validate_solution(inst, redesign, single_sourcing=cfg.single_sourcing,
                      max_open_dcs=cfg.max_open_dcs)  # fmt: skip
    kept = evaluate_design(inst, base_dcs, cfg)
    c_new = cost_breakdown(inst, redesign)["total"]
    c_kept = cost_breakdown(inst, kept)["total"]
    demand = inst.total_demand()
    return {
        "parameter": param,
        "multiplier": mult,
        "redesign_open_dcs": ",".join(sorted(redesign.open_dcs)),
        "redesign_n_open": len(redesign.open_dcs),
        "design_changed": redesign.open_dcs != base_dcs,
        "redesign_cost": c_new,
        "base_design_cost": c_kept,
        "regret": c_kept - c_new,
        "regret_pct": 100 * (c_kept - c_new) / c_new if c_new else float("nan"),
        "redesign_service_level": 1 - sum(redesign.u.values()) / demand,
        "base_design_service_level": 1 - sum(kept.u.values()) / demand,
    }


def run_sensitivity(
    base: NetworkInstance,
    base_dcs: frozenset[str],
    ocfg: OptimizationConfig,
    scfg: SimulationConfig,
) -> pd.DataFrame:
    """One-at-a-time sensitivity: fixed costs, transport costs, demand level."""
    sens = scfg.sensitivity
    jobs: list[tuple[str, float, NetworkInstance, frozenset[str], OptimizationConfig]] = []
    for param, mults, modify in [
        ("fixed_cost", sens.fixed_cost_multipliers, base.scale_fixed_costs),
        ("transport_cost", sens.transport_cost_multipliers, base.scale_transport),
        ("demand", sens.demand_multipliers, base.scale_demand),
    ]:
        jobs.extend((param, m, modify(m), base_dcs, ocfg) for m in mults)
    return pd.DataFrame(parallel_map(_sensitivity_one, jobs, scfg.n_jobs))
