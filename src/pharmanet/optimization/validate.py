"""Independent validation of a network design solution.

These checks re-derive every constraint directly from the flows, without
trusting the solver, the model, or the heuristic that produced them. If any
check fails we raise: a wrong solution must never be reported as a result.

Tolerances are relative to the scale of the quantity (solvers work in floating
point and accept small violations, ~1e-6 relative, by design).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.solution import Solution, cost_breakdown

ABS_TOL = 1e-4
REL_TOL = 1e-6


class SolutionValidationError(AssertionError):
    """A solution violates a constraint of the network design problem."""


def _close(a: float, b: float, scale: float) -> bool:
    return abs(a - b) <= ABS_TOL + REL_TOL * max(1.0, abs(scale))


def _le(a: float, b: float) -> bool:
    return a <= b + ABS_TOL + REL_TOL * max(1.0, abs(b))


def validate_solution(
    inst: NetworkInstance,
    sol: Solution,
    *,
    single_sourcing: bool = False,
    max_open_dcs: int | None = None,
) -> dict[str, float]:
    """Check flow conservation, capacities, qualification, closed DCs, and the objective.

    Returns the independently computed cost breakdown on success; raises
    ``SolutionValidationError`` listing *all* violations otherwise.
    """
    errors: list[str] = []
    sets = {"P": set(inst.plants), "D": set(inst.dcs), "R": set(inst.regions)}
    K = set(inst.products)

    # Non-negativity and known indices
    checks: list[tuple[str, dict[Any, float], str]] = [
        ("x", sol.x, "PDK"),
        ("z", sol.z, "DRK"),
        ("u", sol.u, "RK"),
    ]
    for name, flows, kinds in checks:
        for key, v in flows.items():
            if v < -ABS_TOL:
                errors.append(f"{name}{key} is negative ({v})")
            for kind, part in zip(kinds, key, strict=True):
                if (part not in K) if kind == "K" else (part not in sets[kind]):
                    errors.append(f"{name}{key} references unknown index {part}")
    if not sol.open_dcs <= sets["D"]:
        errors.append(f"open DCs not in instance: {sorted(sol.open_dcs - sets['D'])}")

    inflow: dict[tuple[str, str], float] = defaultdict(float)  # (d, k)
    plant_out: dict[tuple[str, str], float] = defaultdict(float)  # (p, k)
    for (p, d, k), v in sol.x.items():
        inflow[(d, k)] += v
        plant_out[(p, k)] += v
    outflow: dict[tuple[str, str], float] = defaultdict(float)  # (d, k)
    delivered: dict[tuple[str, str], float] = defaultdict(float)  # (r, k)
    dc_total: dict[str, float] = defaultdict(float)
    dc_cold: dict[str, float] = defaultdict(float)
    regions_served_by: dict[str, set[str]] = defaultdict(set)
    for (d, r, k), v in sol.z.items():
        outflow[(d, k)] += v
        delivered[(r, k)] += v
        dc_total[d] += v
        if k in inst.cold_products:
            dc_cold[d] += v
        regions_served_by[r].add(d)

    # C1 demand: delivered + unmet == demand (unmet must be explicit, never silent)
    for (r, k), dem in inst.dem.items():
        got = delivered[(r, k)] + sol.u.get((r, k), 0.0)
        if not _close(got, dem, dem):
            errors.append(f"demand ({r},{k}): delivered+unmet={got:.4f} != {dem:.4f}")

    # C2 flow conservation at every DC and product
    for d in inst.dcs:
        for k in inst.products:
            a, b = inflow[(d, k)], outflow[(d, k)]
            if not _close(a, b, max(a, b)):
                errors.append(f"flow balance ({d},{k}): in={a:.4f} out={b:.4f}")

    # C3 plant capacity and qualification
    for (p, k), v in plant_out.items():
        if v > ABS_TOL and not inst.qualified[(p, k)]:
            errors.append(f"plant {p} ships {k} without qualification ({v:.4f})")
        if not _le(v, inst.cap_pk[(p, k)]):
            errors.append(f"plant capacity ({p},{k}): {v:.4f} > {inst.cap_pk[(p, k)]:.4f}")

    # C4/C5 DC capacity; no flow through closed DCs
    for d in inst.dcs:
        is_open = d in sol.open_dcs
        through = dc_total[d] + sum(inflow[(d, k)] for k in inst.products)
        if not is_open and through > ABS_TOL:
            errors.append(f"closed DC {d} carries flow ({through:.4f})")
        if not _le(dc_total[d], inst.cap_d[d]):
            errors.append(f"DC throughput {d}: {dc_total[d]:.4f} > {inst.cap_d[d]:.4f}")
        if not _le(dc_cold[d], inst.cold_d[d]):
            errors.append(f"DC cold capacity {d}: {dc_cold[d]:.4f} > {inst.cold_d[d]:.4f}")

    # Optional structural rules
    if single_sourcing:
        for r, served in regions_served_by.items():
            if len(served) > 1:
                errors.append(f"single-sourcing: region {r} served by {sorted(served)}")
    if max_open_dcs is not None and len(sol.open_dcs) > max_open_dcs:
        errors.append(f"{len(sol.open_dcs)} DCs open > max_open_dcs={max_open_dcs}")

    # Objective: independently recomputed cost must match what the solver reported
    costs = cost_breakdown(inst, sol)
    if sol.stats is not None and not _close(sol.stats.objective, costs["total"], costs["total"]):
        errors.append(
            f"objective mismatch: solver={sol.stats.objective:.4f} recomputed={costs['total']:.4f}"
        )

    if errors:
        raise SolutionValidationError(
            f"Solution '{sol.method}' failed validation ({len(errors)} issues):\n  - "
            + "\n  - ".join(errors[:50])
        )
    return costs
