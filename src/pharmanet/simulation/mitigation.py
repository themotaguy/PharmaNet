"""Resilience mitigations and their price.

Each mitigation is evaluated the same way: what it costs at design demand
(the "price of resilience") and what it does to demand at risk across every
single-node failure (each plant and each open DC fully down, at forecast demand).

* ``base``: the cost-optimal design from Phase 4.
* ``n_minus_1``: re-solve the MILP with C8/C9, so open throughput and cold
  capacity survive the loss of any one DC.
* ``dual_qualification``: give every single-source product a backup plant
  (the other plant with the cheapest inbound lanes to the open DCs) with
  ``backup_capacity_share`` x the product's demand. In pharma this is a tech
  transfer plus regulatory approval; its one-off cost is **not** in the model,
  so the reported premium covers network costs only.
* ``n_minus_1+dual_qualification``: both.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from pharmanet.config import OptimizationConfig, SimulationConfig
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.network_design import ModelOptions, solve_network_design
from pharmanet.optimization.solution import cost_breakdown
from pharmanet.optimization.validate import validate_solution
from pharmanet.simulation.evaluate import evaluate_design, evaluate_scenarios
from pharmanet.simulation.scenarios import disruption_scenarios


def single_source_products(inst: NetworkInstance) -> dict[str, str]:
    """Products with exactly one qualified plant -> that plant."""
    out = {}
    for k in inst.products:
        plants = [p for p in inst.plants if inst.qualified[(p, k)] and inst.cap_pk[(p, k)] > 0]
        if len(plants) == 1:
            out[k] = plants[0]
    return out


def dual_qualify(
    design_inst: NetworkInstance,
    forecast_inst: NetworkInstance,
    open_dcs: frozenset[str],
    share: float,
) -> tuple[NetworkInstance, NetworkInstance, list[dict[str, Any]]]:
    """Add a backup plant for each single-source product, in both demand bases."""
    actions = []
    di, fi = design_inst, forecast_inst
    for k, current in sorted(single_source_products(design_inst).items()):
        candidates = [p for p in design_inst.plants if p != current]
        if not candidates:
            continue
        backup = min(
            candidates,
            key=lambda p: (sum(design_inst.c_in[(p, d, k)] for d in open_dcs), p),
        )
        cap = share * sum(design_inst.dem[(r, k)] for r in design_inst.regions)
        di, fi = di.qualify(backup, k, cap), fi.qualify(backup, k, cap)
        actions.append({"product_id": k, "sole_plant": current, "backup_plant": backup,
                        "backup_capacity": cap})  # fmt: skip
    return di, fi, actions


def _node_failures(
    forecast_inst: NetworkInstance,
    open_dcs: frozenset[str],
    ocfg: OptimizationConfig,
    scfg: SimulationConfig,
) -> pd.DataFrame:
    base = evaluate_design(forecast_inst, open_dcs, ocfg)
    only_nodes = scfg.model_copy(
        update={
            "disruptions": scfg.disruptions.model_copy(update={"cold_lane_failure": False}),
            "outage_capacity_loss": 1.0,
        }
    )
    rows, _ = evaluate_scenarios(
        disruption_scenarios(forecast_inst, base, only_nodes), open_dcs, ocfg, scfg.n_jobs
    )
    return rows


def evaluate_mitigation(
    name: str,
    design_inst: NetworkInstance,
    forecast_inst: NetworkInstance,
    open_dcs: frozenset[str],
    ocfg: OptimizationConfig,
    scfg: SimulationConfig,
) -> dict[str, Any]:
    design_sol = evaluate_design(design_inst, open_dcs, ocfg)
    rows = _node_failures(forecast_inst, open_dcs, ocfg, scfg)
    worst = rows.sort_values("unmet", ascending=False).iloc[0]
    total = forecast_inst.total_demand()
    return {
        "mitigation": name,
        "open_dcs": ",".join(sorted(open_dcs)),
        "n_open": len(open_dcs),
        "design_cost": cost_breakdown(design_inst, design_sol)["total"],
        "open_capacity_headroom_pct": 100
        * (sum(design_inst.cap_d[d] for d in open_dcs) / design_inst.total_demand() - 1),
        "worst_failure": str(worst["target"]),
        "worst_failure_unmet": float(worst["unmet"]),
        "worst_failure_demand_at_risk_pct": 100 * float(worst["unmet"]) / total,
        "failures_with_shortfall": int((rows["unmet"] > 1e-6).sum()),
        "n_failures": len(rows),
        "mean_demand_at_risk_pct": 100 * float(rows["unmet"].mean()) / total,
    }


def run_mitigations(
    design_inst: NetworkInstance,
    forecast_inst: NetworkInstance,
    base_dcs: frozenset[str],
    ocfg: OptimizationConfig,
    scfg: SimulationConfig,
    backup_capacity_share: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate base and the three mitigations; returns (table, dual-qualification actions)."""

    def resilient(inst: NetworkInstance) -> frozenset[str]:
        opts = replace(ModelOptions.from_config(ocfg), n_minus_1_dc=True)
        sol = solve_network_design(inst, ocfg, opts=opts, compute_lp_bounds=False)
        validate_solution(inst, sol, max_open_dcs=ocfg.max_open_dcs)
        return sol.open_dcs

    dq_design, dq_forecast, actions = dual_qualify(
        design_inst, forecast_inst, base_dcs, backup_capacity_share
    )
    n1_dcs = resilient(design_inst)
    both_dcs = resilient(dq_design)
    rows = [
        evaluate_mitigation("base", design_inst, forecast_inst, base_dcs, ocfg, scfg),
        evaluate_mitigation("n_minus_1", design_inst, forecast_inst, n1_dcs, ocfg, scfg),
        evaluate_mitigation("dual_qualification", dq_design, dq_forecast, base_dcs, ocfg, scfg),
        evaluate_mitigation(
            "n_minus_1+dual_qualification", dq_design, dq_forecast, both_dcs, ocfg, scfg
        ),
    ]
    df = pd.DataFrame(rows)
    base_cost = float(df["design_cost"].iloc[0])
    base_worst = float(df["worst_failure_unmet"].iloc[0])
    df["cost_premium_pct"] = 100 * (df["design_cost"] - base_cost) / base_cost
    reduction = 100 * (base_worst - df["worst_failure_unmet"]) / base_worst if base_worst else 0.0
    df["worst_failure_reduction_pct"] = reduction
    # Avoid "-0.00" from floating-point noise in an unchanged design.
    for col in ("cost_premium_pct", "worst_failure_reduction_pct"):
        df.loc[df[col].abs() < 1e-9, col] = 0.0
    return df, actions
