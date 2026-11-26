"""Greedy heuristic baseline, used to measure what the MILP is worth.

This is the kind of rule a planner might apply without an optimizer:

1. **Choose DCs**: open DCs in order of increasing fixed cost until open
   capacity covers total demand *and* open cold capacity covers cold demand.
2. **Route**: take (region, product) demands **cold first** (cold capacity is the
   scarce resource, so a sensible planner allocates it before ambient volume
   can fill the cold-capable DCs), then largest-first; send each to the
   nearest open DC that can hold it (cold storage for cold products) and has
   room, sourcing from the nearest qualified plant with capacity left. Split
   over the next-nearest options when capacity runs out; anything left is unmet.

"Nearest" = cheapest lane: lane costs are proportional to distance in the data.

The heuristic is myopic in two ways the MILP is not: it ignores geography when
choosing DCs, and it routes one demand at a time (early choices can use up
capacity that later demands needed more). ``run_benchmarks`` separates the two
by also solving "greedy DCs + optimal flows".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from pharmanet.config import OptimizationConfig
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.network_design import ModelOptions, solve_network_design
from pharmanet.optimization.solution import Solution, cost_breakdown

TOL = 1e-9


def greedy_open_dcs(inst: NetworkInstance) -> list[str]:
    """Cheapest-fixed-cost-first until total and cold capacity cover demand."""
    total = inst.total_demand()
    cold_total = sum(v for (_, k), v in inst.dem.items() if k in inst.cold_products)
    opened: list[str] = []
    cap = cold = 0.0
    for d in sorted(inst.dcs, key=lambda d: (inst.fixed_cost[d], d)):
        if cap >= total and cold >= cold_total:
            break
        opened.append(d)
        cap += inst.cap_d[d]
        cold += inst.cold_d[d]
    return opened


def greedy_solution(inst: NetworkInstance, open_dcs: list[str] | None = None) -> Solution:
    """Full greedy heuristic (DC choice + routing). Deterministic: ties broken by ID."""
    opened = open_dcs if open_dcs is not None else greedy_open_dcs(inst)
    rem_dc = {d: inst.cap_d[d] for d in opened}
    rem_cold = {d: inst.cold_d[d] for d in opened}
    rem_plant = {pk: inst.cap_pk[pk] * inst.qualified[pk] for pk in inst.cap_pk}

    x: dict[tuple[str, str, str], float] = defaultdict(float)
    z: dict[tuple[str, str, str], float] = defaultdict(float)
    u: dict[tuple[str, str], float] = {}

    def priority(kv: tuple[tuple[str, str], float]) -> tuple[bool, float, tuple[str, str]]:
        (_, k), demand = kv
        return (k not in inst.cold_products, -demand, kv[0])  # cold first, then largest

    for (r, k), demand in sorted(inst.dem.items(), key=priority):
        need = demand
        cold = k in inst.cold_products
        candidates = [d for d in opened if not cold or inst.cold_d[d] > 0]
        for d in sorted(candidates, key=lambda d: (inst.c_out[d, r, k], d)):
            room = min(rem_dc[d], rem_cold[d]) if cold else rem_dc[d]
            if need <= TOL:
                break
            if room <= TOL:
                continue
            plants = sorted(
                (p for p in inst.plants if rem_plant[p, k] > TOL),
                key=lambda p: (inst.c_in[p, d, k], p),
            )
            for p in plants:
                q = min(need, room, rem_plant[p, k])
                x[p, d, k] += q
                z[d, r, k] += q
                rem_plant[p, k] -= q
                rem_dc[d] -= q
                if cold:
                    rem_cold[d] -= q
                need -= q
                room -= q
                if need <= TOL or room <= TOL:
                    break
        if need > TOL:
            u[(r, k)] = need

    return Solution(
        method="greedy",
        open_dcs=frozenset(opened),
        x={key: v for key, v in x.items() if v > TOL},
        z={key: v for key, v in z.items() if v > TOL},
        u=u,
    )


def run_benchmarks(
    inst: NetworkInstance, cfg: OptimizationConfig, milp: Solution
) -> tuple[Solution, Solution, dict[str, float]]:
    """Greedy vs MILP, with the saving split into DC *choice* vs *routing*.

    * greedy:                    greedy DCs + greedy routing
    * greedy_dcs_optimal_flows:  greedy DCs, flows re-optimized by LP (y fixed)
    * milp:                      optimal DCs + optimal flows

    routing saving  = greedy - greedy_dcs_optimal_flows
    location saving = greedy_dcs_optimal_flows - milp
    All costs come from ``cost_breakdown`` (same function for every method).
    """
    greedy = greedy_solution(inst)
    fixed = {d: int(d in greedy.open_dcs) for d in inst.dcs}
    opts = replace(ModelOptions.from_config(cfg), fixed_open=fixed, max_open_dcs=None)
    greedy_of = solve_network_design(
        inst, cfg, opts=opts, compute_lp_bounds=False, method="greedy_dcs_optimal_flows"
    )
    b_g, b_gof, b_m = (cost_breakdown(inst, s) for s in (greedy, greedy_of, milp))
    c_g, c_gof, c_m = b_g["total"], b_gof["total"], b_m["total"]
    pct = 100.0 / c_g if c_g else float("nan")

    def logistics(b: dict[str, float]) -> float:
        return b["total"] - b["penalty"]

    # Penalty-free comparison: real logistics cost per unit actually delivered.
    # Guards against a headline inflated by the (artificial) unmet-demand penalty.
    def per_unit(sol: Solution, b: dict[str, float]) -> float:
        delivered = inst.total_demand() - sum(sol.u.values())
        return logistics(b) / delivered if delivered else float("nan")

    cpu_g, cpu_m = per_unit(greedy, b_g), per_unit(milp, b_m)
    return (
        greedy,
        greedy_of,
        {
            "greedy_total_cost": c_g,
            "greedy_dcs_optimal_flows_total_cost": c_gof,
            "milp_total_cost": c_m,
            "savings_vs_greedy_pct": (c_g - c_m) * pct,
            "savings_from_routing_pct": (c_g - c_gof) * pct,
            "savings_from_dc_choice_pct": (c_gof - c_m) * pct,
            "greedy_penalty_cost": b_g["penalty"],
            "milp_penalty_cost": b_m["penalty"],
            "greedy_logistics_cost_per_unit": cpu_g,
            "milp_logistics_cost_per_unit": cpu_m,
            "logistics_cost_per_unit_savings_pct": 100 * (cpu_g - cpu_m) / cpu_g,
        },
    )
