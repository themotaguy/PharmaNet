"""Optimization tests: hand-verified regression instance, validator, and properties.

The regression instance (2 plants, 3 DCs, 3 regions, 2 products) is small enough
to solve on paper. Its optimum is derived in docs/formulation.md ("Regression
instance"); the tests assert the exact objective and open DCs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pharmanet.config import OptimizationConfig, load_config
from pharmanet.data.generate import annual_demand, generate
from pharmanet.optimization.baseline import greedy_open_dcs, greedy_solution, run_benchmarks
from pharmanet.optimization.instance import NetworkInstance, build_instance
from pharmanet.optimization.network_design import ModelOptions, solve_network_design
from pharmanet.optimization.solution import Solution, cost_breakdown
from pharmanet.optimization.validate import SolutionValidationError, validate_solution

# ------------------------------------------------------------- regression instance
P, D, R, K = ("P1", "P2"), ("D1", "D2", "D3"), ("R1", "R2", "R3"), ("A", "C")
IN_COST = {("P1", "D1"): 1, ("P1", "D2"): 2, ("P1", "D3"): 3,
           ("P2", "D1"): 3, ("P2", "D2"): 2, ("P2", "D3"): 1}  # fmt: skip
OUT_COST = {("D1", "R1"): 1, ("D1", "R2"): 3, ("D1", "R3"): 5,
            ("D2", "R1"): 3, ("D2", "R2"): 1, ("D2", "R3"): 3,
            ("D3", "R1"): 5, ("D3", "R2"): 3, ("D3", "R3"): 1}  # fmt: skip
COLD_PREMIUM = 2.0


def regression_instance() -> NetworkInstance:
    """A = ambient, C = cold. Only P1 may make C; D3 has no cold storage."""
    prem = {"A": 1.0, "C": COLD_PREMIUM}
    return NetworkInstance(
        plants=P,
        dcs=D,
        regions=R,
        products=K,
        cold_products=frozenset({"C"}),
        fixed_cost={"D1": 500.0, "D2": 350.0, "D3": 400.0},
        c_in={(p, d, k): IN_COST[p, d] * prem[k] for p in P for d in D for k in K},
        c_out={(d, r, k): OUT_COST[d, r] * prem[k] for d in D for r in R for k in K},
        dem={(r, k): 100.0 if k == "A" else 50.0 for r in R for k in K},
        cap_pk={("P1", "A"): 1000.0, ("P1", "C"): 1000.0, ("P2", "A"): 1000.0, ("P2", "C"): 0.0},
        qualified={("P1", "A"): 1, ("P1", "C"): 1, ("P2", "A"): 1, ("P2", "C"): 0},
        cap_d={d: 1000.0 for d in D},
        cold_d={"D1": 1000.0, "D2": 1000.0, "D3": 0.0},
        penalty={"A": 100.0, "C": 100.0},
    )


@pytest.fixture
def ocfg(small_config_path: Path) -> OptimizationConfig:
    return load_config(small_config_path).optimization


def _solve(
    inst: NetworkInstance, ocfg: OptimizationConfig, **opts: object
) -> tuple[Solution, dict[str, float]]:
    sol = solve_network_design(inst, ocfg, opts=replace(ModelOptions(), **opts))  # type: ignore[arg-type]
    return sol, validate_solution(inst, sol)


def test_regression_optimum(ocfg: OptimizationConfig) -> None:
    sol, costs = _solve(regression_instance(), ocfg)
    assert sol.open_dcs == {"D1", "D2"}
    assert costs["total"] == pytest.approx(2850.0)
    assert costs == pytest.approx(
        {"fixed": 850.0, "inbound": 1000.0, "outbound": 1000.0, "penalty": 0.0, "total": 2850.0}
    )
    assert not sol.u


@pytest.mark.parametrize(
    ("open_set", "expected"),
    [
        ({"D1"}, 2900.0),
        ({"D2"}, 2950.0),
        ({"D3"}, 16600.0),  # no cold DC: all 150 cold units unmet at 100 each
        ({"D1", "D3"}, 2900.0),
        ({"D2", "D3"}, 3050.0),
        ({"D1", "D2", "D3"}, 2950.0),
    ],
)
def test_regression_every_other_design_costs_more(
    ocfg: OptimizationConfig, open_set: set[str], expected: float
) -> None:
    """The enumeration table from the docs, checked design by design."""
    fixed = {d: int(d in open_set) for d in D}
    _, costs = _solve(regression_instance(), ocfg, fixed_open=fixed)
    assert costs["total"] == pytest.approx(expected)
    assert expected > 2850.0


def test_regression_unmet_is_explicit(ocfg: OptimizationConfig) -> None:
    sol, _ = _solve(regression_instance(), ocfg, fixed_open={"D1": 0, "D2": 0, "D3": 1})
    assert sol.u == pytest.approx({("R1", "C"): 50.0, ("R2", "C"): 50.0, ("R3", "C"): 50.0})


def test_regression_max_one_dc(ocfg: OptimizationConfig) -> None:
    sol, costs = _solve(regression_instance(), ocfg, max_open_dcs=1)
    assert sol.open_dcs == {"D1"}
    assert costs["total"] == pytest.approx(2900.0)


def test_regression_single_sourcing(ocfg: OptimizationConfig) -> None:
    sol, costs = _solve(regression_instance(), ocfg, single_sourcing=True)
    validate_solution(regression_instance(), sol, single_sourcing=True)
    assert costs["total"] == pytest.approx(2850.0)
    assert sol.assignment == {"R1": "D1", "R2": "D2", "R3": "D2"}


def test_regression_greedy(ocfg: OptimizationConfig) -> None:
    inst = regression_instance()
    assert greedy_open_dcs(inst) == ["D2"]  # cheapest fixed cost, enough capacity alone
    g = greedy_solution(inst)
    assert cost_breakdown(inst, validate_and_return(inst, g))["total"] == pytest.approx(2950.0)
    milp, _ = _solve(inst, ocfg)
    _, _, bench = run_benchmarks(inst, ocfg, milp)
    assert bench["savings_vs_greedy_pct"] == pytest.approx(100 * 100 / 2950)
    assert bench["savings_from_routing_pct"] == pytest.approx(0.0)


def validate_and_return(inst: NetworkInstance, sol: Solution) -> Solution:
    validate_solution(inst, sol)
    return sol


def test_lp_bounds_are_valid_and_strong_link_tightens(ocfg: OptimizationConfig) -> None:
    sol, costs = _solve(regression_instance(), ocfg)
    st = sol.stats
    assert st is not None and st.lp_bound is not None and st.lp_bound_weak is not None
    assert st.lp_bound_weak <= st.lp_bound + 1e-6 <= costs["total"] + 1e-6
    assert st.n_binary == 3
    assert st.termination == "optimal"


# ---------------------------------------------------------------------- validator
def _tamper(sol: Solution, **changes: object) -> Solution:
    return replace(sol, **changes)  # type: ignore[arg-type]


@pytest.fixture
def good(ocfg: OptimizationConfig) -> Solution:
    return _solve(regression_instance(), ocfg)[0]


def test_validator_flags_flow_through_closed_dc(good: Solution) -> None:
    bad = _tamper(good, open_dcs=frozenset({"D1"}))
    with pytest.raises(SolutionValidationError, match="closed DC D2"):
        validate_solution(regression_instance(), bad)


def test_validator_flags_unqualified_plant(good: Solution) -> None:
    x = dict(good.x)
    moved = x.pop(("P1", "D2", "C"))
    x[("P2", "D2", "C")] = moved
    with pytest.raises(SolutionValidationError, match="without qualification"):
        validate_solution(regression_instance(), _tamper(good, x=x))


def test_validator_flags_imbalance_and_silent_shortfall(good: Solution) -> None:
    z = dict(good.z)
    z[("D1", "R1", "A")] -= 10
    with pytest.raises(SolutionValidationError) as exc:
        validate_solution(regression_instance(), _tamper(good, z=z))
    assert "flow balance (D1,A)" in str(exc.value) and "demand (R1,A)" in str(exc.value)


def test_validator_flags_objective_mismatch(good: Solution) -> None:
    assert good.stats is not None
    bad = _tamper(good, stats=replace(good.stats, objective=good.stats.objective + 5))
    with pytest.raises(SolutionValidationError, match="objective mismatch"):
        validate_solution(regression_instance(), bad)


def test_validator_flags_single_sourcing_violation(good: Solution) -> None:
    z = dict(good.z)
    x = dict(good.x)
    # Serve R1's cold demand from D2 instead of D1 (still balanced and feasible).
    q = z.pop(("D1", "R1", "C"))
    z[("D2", "R1", "C")] = q
    x[("P1", "D1", "C")] -= q
    x[("P1", "D2", "C")] += q
    x = {k: v for k, v in x.items() if v > 0}
    with pytest.raises(SolutionValidationError, match="single-sourcing"):
        validate_solution(regression_instance(), _tamper(good, x=x, z=z), single_sourcing=True)


# ------------------------------------------------------------- property tests
@pytest.fixture(scope="module")
def small_inst() -> NetworkInstance:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "small_test.yaml")
    ds = generate(cfg)
    dd = annual_demand(ds.demand, ["region_id", "product_id"]).rename("design_demand")
    return build_instance(ds, dd.reset_index())


@pytest.mark.parametrize("factor", [1.05, 1.2, 1.5, 3.0])
def test_more_demand_never_costs_less(
    small_inst: NetworkInstance, ocfg: OptimizationConfig, factor: float
) -> None:
    _, base = _solve(small_inst, ocfg)
    _, more = _solve(small_inst.scale_demand(factor), ocfg)
    assert more["total"] >= base["total"] - 1e-6


def test_removing_a_dc_never_costs_less(
    small_inst: NetworkInstance, ocfg: OptimizationConfig
) -> None:
    _, base = _solve(small_inst, ocfg)
    for d in small_inst.dcs:
        # Forbid DC d (y_d = 0), leave the others free.
        sol = solve_network_design(small_inst, ocfg, opts=ModelOptions(fixed_open={d: 0}))
        assert validate_solution(small_inst, sol)["total"] >= base["total"] - 1e-6


def test_milp_never_worse_than_greedy(
    small_inst: NetworkInstance, ocfg: OptimizationConfig
) -> None:
    for factor in (0.8, 1.0, 1.5):
        inst = small_inst.scale_demand(factor)
        _, m = _solve(inst, ocfg)
        g = greedy_solution(inst)
        assert m["total"] <= validate_solution(inst, g)["total"] + 1e-6


def test_strong_link_does_not_change_the_optimum(
    small_inst: NetworkInstance, ocfg: OptimizationConfig
) -> None:
    _, strong = _solve(small_inst, ocfg, strong_linking=True)
    _, weak = _solve(small_inst, ocfg, strong_linking=False)
    assert strong["total"] == pytest.approx(weak["total"], rel=1e-6)
