"""Network design MILP (Pyomo). Matches docs/formulation.md one-to-one.

Constraint names in code = names in the doc:
    demand (C1), flow_balance (C2), plant_capacity (C3), dc_throughput (C4),
    dc_cold (C5), strong_link (C6), max_open_dcs (C7),
    single_source_one_dc (S1), single_source_open (S2), single_source_link (S3).

The solver is chosen by config (HiGHS by default). Gurobi/CBC are wired up
through the same interface but are untested here (not installed).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from pharmanet.config import OptimizationConfig, SolverConfig
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.solution import Solution, SolveStats

FLOW_TOL = 1e-6  # flows below this are treated as zero when extracting a solution

_SOLVER_NAMES = {"highs": "appsi_highs", "gurobi": "gurobi", "cbc": "cbc"}
_OK = {TerminationCondition.optimal, TerminationCondition.maxTimeLimit}


class SolverError(RuntimeError):
    """Raised when the solver returns no usable solution."""


@dataclass(frozen=True)
class ModelOptions:
    """Structural switches for the model (from config, plus programmatic overrides)."""

    strong_linking: bool = True
    single_sourcing: bool = False
    max_open_dcs: int | None = None
    # Fix y_d to given 0/1 values. Used to evaluate a fixed design (Phase 5),
    # the "greedy DCs + optimal flows" benchmark, and property tests.
    fixed_open: Mapping[str, int] | None = None

    @classmethod
    def from_config(cls, cfg: OptimizationConfig) -> ModelOptions:
        return cls(
            strong_linking=cfg.strong_linking,
            single_sourcing=cfg.single_sourcing,
            max_open_dcs=cfg.max_open_dcs,
        )


# ------------------------------------------------------------------------- model
def build_model(inst: NetworkInstance, opts: ModelOptions) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name="network_design")

    # Sets
    m.P = pyo.Set(initialize=inst.plants, ordered=True)
    m.D = pyo.Set(initialize=inst.dcs, ordered=True)
    m.R = pyo.Set(initialize=inst.regions, ordered=True)
    m.K = pyo.Set(initialize=inst.products, ordered=True)
    m.Kc = pyo.Set(within=m.K, initialize=[k for k in inst.products if k in inst.cold_products])

    # Variables
    m.y = pyo.Var(m.D, domain=pyo.Binary)
    m.x = pyo.Var(m.P, m.D, m.K, domain=pyo.NonNegativeReals)
    m.z = pyo.Var(m.D, m.R, m.K, domain=pyo.NonNegativeReals)
    m.u = pyo.Var(m.R, m.K, domain=pyo.NonNegativeReals)

    # Objective, kept as named terms so the breakdown can be read off the model
    m.cost_fixed = pyo.Expression(expr=sum(inst.fixed_cost[d] * m.y[d] for d in m.D))
    m.cost_inbound = pyo.Expression(
        expr=sum(inst.c_in[p, d, k] * m.x[p, d, k] for p in m.P for d in m.D for k in m.K)
    )
    m.cost_outbound = pyo.Expression(
        expr=sum(inst.c_out[d, r, k] * m.z[d, r, k] for d in m.D for r in m.R for k in m.K)
    )
    m.cost_penalty = pyo.Expression(expr=sum(inst.penalty[k] * m.u[r, k] for r in m.R for k in m.K))
    m.total_cost = pyo.Objective(
        expr=m.cost_fixed + m.cost_inbound + m.cost_outbound + m.cost_penalty,
        sense=pyo.minimize,
    )

    # (C1) demand: delivered + unmet = design demand
    m.demand = pyo.Constraint(
        m.R, m.K, rule=lambda m, r, k: sum(m.z[d, r, k] for d in m.D) + m.u[r, k] == inst.dem[r, k]
    )
    # (C2) flow_balance: DC outflow = DC inflow, per product
    m.flow_balance = pyo.Constraint(
        m.D,
        m.K,
        rule=lambda m, d, k: sum(m.x[p, d, k] for p in m.P) == sum(m.z[d, r, k] for r in m.R),
    )
    # (C3) plant_capacity: only qualified plants, within capacity
    m.plant_capacity = pyo.Constraint(
        m.P,
        m.K,
        rule=lambda m, p, k: (
            sum(m.x[p, d, k] for d in m.D) <= inst.cap_pk[p, k] * inst.qualified[p, k]
        ),
    )
    # (C4) dc_throughput: total flow within capacity, zero if closed
    m.dc_throughput = pyo.Constraint(
        m.D, rule=lambda m, d: sum(m.z[d, r, k] for r in m.R for k in m.K) <= inst.cap_d[d] * m.y[d]
    )
    # (C5) dc_cold: cold flow within cold capacity, zero if closed
    m.dc_cold = pyo.Constraint(
        m.D,
        rule=lambda m, d: sum(m.z[d, r, k] for r in m.R for k in m.Kc) <= inst.cold_d[d] * m.y[d],
    )

    if opts.single_sourcing:
        m.a = pyo.Var(m.D, m.R, domain=pyo.Binary)
        # (S1) each region assigned to exactly one DC
        m.single_source_one_dc = pyo.Constraint(
            m.R, rule=lambda m, r: sum(m.a[d, r] for d in m.D) == 1
        )
        # (S2) only to an open DC
        m.single_source_open = pyo.Constraint(m.D, m.R, rule=lambda m, d, r: m.a[d, r] <= m.y[d])
        # (S3) flows only along the assignment (also implies C6 via S2)
        m.single_source_link = pyo.Constraint(
            m.D, m.R, m.K, rule=lambda m, d, r, k: m.z[d, r, k] <= inst.dem[r, k] * m.a[d, r]
        )
    elif opts.strong_linking:
        # (C6) strong_link: disaggregated linking, tightens the LP relaxation
        m.strong_link = pyo.Constraint(
            m.D, m.R, m.K, rule=lambda m, d, r, k: m.z[d, r, k] <= inst.dem[r, k] * m.y[d]
        )

    if opts.max_open_dcs is not None:
        # (C7) max_open_dcs
        m.max_open_dcs = pyo.Constraint(expr=sum(m.y[d] for d in m.D) <= opts.max_open_dcs)

    if opts.fixed_open is not None:
        for d, v in opts.fixed_open.items():
            m.y[d].fix(int(v))
    # Disrupted lanes carry nothing (Phase 5 scenarios).
    for p, d, k in inst.blocked_in:
        m.x[p, d, k].fix(0.0)
    for d, r, k in inst.blocked_out:
        m.z[d, r, k].fix(0.0)
    return m


def model_size(m: pyo.ConcreteModel) -> dict[str, int]:
    variables = list(m.component_data_objects(pyo.Var))
    return {
        "n_variables": len(variables),
        "n_binary": sum(1 for v in variables if v.is_binary()),
        "n_constraints": sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True)),
    }


# ------------------------------------------------------------------------ solver
def make_solver(cfg: SolverConfig) -> Any:
    """Instantiate the configured solver and translate common options to its dialect."""
    solver = pyo.SolverFactory(_SOLVER_NAMES[cfg.name])
    if not solver.available(exception_flag=False):
        raise SolverError(f"Solver '{cfg.name}' is not available in this environment")
    if cfg.name == "highs":
        solver.config.time_limit = cfg.time_limit_s
        solver.config.mip_gap = cfg.mip_gap
        if cfg.threads:
            solver.highs_options["threads"] = cfg.threads
    elif cfg.name == "gurobi":
        solver.options.update({"TimeLimit": cfg.time_limit_s, "MIPGap": cfg.mip_gap})
        if cfg.threads:
            solver.options["Threads"] = cfg.threads
    else:  # cbc
        solver.options.update({"seconds": cfg.time_limit_s, "ratioGap": cfg.mip_gap})
        if cfg.threads:
            solver.options["threads"] = cfg.threads
    return solver


def _solve(m: pyo.ConcreteModel, cfg: SolverConfig) -> tuple[str, float | None, float]:
    """Solve in place. Returns (termination, best bound, wall seconds)."""
    solver = make_solver(cfg)
    t0 = time.perf_counter()
    res = solver.solve(m, load_solutions=False)
    wall = time.perf_counter() - t0
    term = res.solver.termination_condition
    if term not in _OK or len(res.solution) == 0:
        raise SolverError(f"Solver returned no usable solution (termination={term})")
    m.solutions.load_from(res)
    lb = res.problem.lower_bound
    bound = float(lb) if lb is not None and abs(float(lb)) != float("inf") else None
    return str(term), bound, wall


def lp_relaxation_bound(inst: NetworkInstance, opts: ModelOptions, cfg: SolverConfig) -> float:
    """Optimal value with integrality dropped (y in [0,1]): a lower bound on the MILP."""
    m = build_model(inst, opts)
    pyo.TransformationFactory("core.relax_integer_vars").apply_to(m)
    _solve(m, cfg)
    return float(pyo.value(m.total_cost))


# ---------------------------------------------------------------------- top level
def extract_solution(m: pyo.ConcreteModel, method: str, stats: SolveStats | None) -> Solution:
    def nz(var: Any) -> dict[Any, float]:
        return {idx: float(v.value) for idx, v in var.items() if (v.value or 0.0) > FLOW_TOL}

    assignment = None
    if hasattr(m, "a"):
        assignment = {r: d for (d, r), v in m.a.items() if (v.value or 0.0) > 0.5}
    return Solution(
        method=method,
        open_dcs=frozenset(d for d in m.D if (m.y[d].value or 0.0) > 0.5),
        x=nz(m.x),
        z=nz(m.z),
        u=nz(m.u),
        stats=stats,
        assignment=assignment,
    )


def solve_network_design(
    inst: NetworkInstance,
    cfg: OptimizationConfig,
    *,
    opts: ModelOptions | None = None,
    compute_lp_bounds: bool = True,
    method: str = "milp",
) -> Solution:
    """Build, solve, and package the MILP, with LP relaxation bounds for diagnostics.

    ``lp_bound`` is the relaxation of the model exactly as solved;
    ``lp_bound_weak`` drops the strong-linking constraints (C6) to show how much
    they tighten the relaxation (only defined without single-sourcing).
    """
    opts = opts or ModelOptions.from_config(cfg)
    m = build_model(inst, opts)
    term, bound, wall = _solve(m, cfg.solver)
    obj = float(pyo.value(m.total_cost))

    lp = lp_weak = None
    if compute_lp_bounds:
        lp = lp_relaxation_bound(inst, opts, cfg.solver)
        if not opts.single_sourcing:
            lp_weak = lp_relaxation_bound(inst, replace(opts, strong_linking=False), cfg.solver)

    gap = None if bound is None or obj == 0 else max(0.0, (obj - bound) / abs(obj))
    stats = SolveStats(
        termination=term,
        objective=obj,
        best_bound=bound,
        mip_gap=gap,
        wall_time_s=wall,
        lp_bound=lp,
        lp_bound_weak=lp_weak,
        **model_size(m),
    )
    return extract_solution(m, method, stats)
