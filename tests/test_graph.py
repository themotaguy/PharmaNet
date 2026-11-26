"""Graph tests: construction, proportional flow tracing, dominator-based SPOFs,
the structural <= simulated <= exposure cross-check, and mitigations.

Most tests use the regression instance with one tie broken (P2->D2 for A made
slightly dearer) so the optimal flows are unique and every number is checkable
by hand: all A and C then come from P1; P2 is idle but qualified for A.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pharmanet.config import OptimizationConfig, SimulationConfig, load_config
from pharmanet.graph.build import SOURCE, capability_graph, flow_graph
from pharmanet.graph.risk import cross_check, exposure_detail, node_risk, structural_spofs
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.network_design import ModelOptions, solve_network_design
from pharmanet.optimization.solution import Solution, cost_breakdown
from pharmanet.optimization.validate import validate_solution
from pharmanet.simulation.evaluate import evaluate_design, evaluate_scenarios
from pharmanet.simulation.mitigation import dual_qualify, run_mitigations, single_source_products
from pharmanet.simulation.scenarios import disruption_scenarios
from regression_instance import regression_instance

DESIGN = frozenset({"D1", "D2"})


def tie_broken() -> NetworkInstance:
    inst = regression_instance()
    return replace(inst, c_in={**inst.c_in, ("P2", "D2", "A"): 2.5})


@pytest.fixture
def cfgs(small_config_path: Path) -> tuple[OptimizationConfig, SimulationConfig]:
    cfg = load_config(small_config_path)
    return cfg.optimization, cfg.simulation


@pytest.fixture
def base(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> Solution:
    return evaluate_design(tie_broken(), DESIGN, cfgs[0])


# ----------------------------------------------------------------- construction
def test_flow_graph_matches_solution(base: Solution) -> None:
    g = flow_graph(tie_broken(), base)
    assert "D3" not in g  # closed DCs are not part of the network
    assert g.nodes["P1"]["kind"] == "plant" and g.nodes["R1"]["kind"] == "region"
    assert not list(g.out_edges("P2"))  # idle plant: in the graph, no flow
    assert g["P1"]["D1"]["flow"] == pytest.approx(150.0)  # A 100 + C 50 for R1
    assert g["P1"]["D2"]["flow"] == pytest.approx(300.0)
    assert g["D2"]["R3"]["cold_flow"] == pytest.approx(50.0)
    total_in = sum(e["flow"] for _, _, e in g.in_edges(["R1", "R2", "R3"], data=True))
    assert total_in == pytest.approx(450.0)


def test_capability_graph_respects_qualification_and_cold() -> None:
    inst = tie_broken()
    gc = capability_graph(inst, frozenset({"D1", "D2", "D3"}), "C")
    assert set(gc.successors(SOURCE)) == {"P1"}  # only P1 is qualified for C
    assert set(gc.successors("P1")) == {"D1", "D2"}  # D3 has no cold storage
    ga = capability_graph(inst, DESIGN, "A")
    assert set(ga.successors(SOURCE)) == {"P1", "P2"}


def test_capability_graph_drops_blocked_lanes() -> None:
    inst = tie_broken().block_outbound(frozenset({("D2", "R3", "C")}))
    g = capability_graph(inst, DESIGN, "C")
    assert not g.has_edge("D2", "R3") and g.has_edge("D1", "R3")


# --------------------------------------------------------------- flow tracing
def test_proportional_flow_tracing_by_hand() -> None:
    """D1 receives A from P1 (60) and P2 (40) and ships 50 each to R1 and R2.
    P1 makes up 60% of D1's A, so it is exposed for 30 units in each region."""
    inst = tie_broken()
    sol = Solution(
        method="hand",
        open_dcs=frozenset({"D1"}),
        x={("P1", "D1", "A"): 60.0, ("P2", "D1", "A"): 40.0},
        z={("D1", "R1", "A"): 50.0, ("D1", "R2", "A"): 50.0},
        u={},
    )
    e = exposure_detail(inst, sol).set_index(["node", "region_id"])["exposed"]
    assert e[("P1", "R1")] == pytest.approx(30.0)
    assert e[("P2", "R2")] == pytest.approx(20.0)
    assert e[("D1", "R1")] == pytest.approx(50.0)


# ------------------------------------------------------------------- dominators
def test_structural_spofs_by_hand() -> None:
    """Only P1 can make C, so P1 is on every C path; A has P2 as an alternative.
    With two cold-capable DCs open, no DC is a single point of failure."""
    spofs = structural_spofs(tie_broken(), DESIGN)
    assert set(zip(spofs["node"], spofs["product_id"], strict=True)) == {("P1", "C")}
    assert sorted(spofs["region_id"]) == ["R1", "R2", "R3"]


def test_single_open_dc_dominates_everything() -> None:
    spofs = structural_spofs(tie_broken(), frozenset({"D2"}))
    d2 = spofs[spofs["node"] == "D2"]
    assert len(d2) == 6  # every (region, product)
    assert d2["demand"].sum() == pytest.approx(450.0)


def test_unreachable_demand_is_reported() -> None:
    # D3 alone cannot hold cold product: C has no path even before any failure.
    spofs = structural_spofs(tie_broken(), frozenset({"D3"}))
    unreach = spofs[spofs["node"] == "UNREACHABLE"]
    assert set(unreach["product_id"]) == {"C"} and unreach["demand"].sum() == pytest.approx(150.0)


# -------------------------------------------------------------- cross-check
def test_bounds_bracket_simulation(
    cfgs: tuple[OptimizationConfig, SimulationConfig], base: Solution
) -> None:
    """P1 down: structural 150 (all C) = simulated 150 <= exposure 450 (A reroutes to P2).
    D1 down: structural 0 = simulated 0 <= exposure 150 (D2 absorbs it)."""
    inst = tie_broken()
    risk = node_risk(inst, base, exposure_detail(inst, base), structural_spofs(inst, DESIGN))
    rows, _ = evaluate_scenarios(disruption_scenarios(inst, base, cfgs[1]), DESIGN, cfgs[0], 1)
    check = cross_check(risk, rows).set_index("node")
    assert check["lower_bound_ok"].all() and check["upper_bound_ok"].all()
    p1 = check.loc["P1"]
    assert (p1["structural_loss"], p1["simulated_loss"], p1["exposure"]) == pytest.approx(
        (150.0, 150.0, 450.0)
    )
    d1 = check.loc["D1"]
    assert (d1["structural_loss"], d1["simulated_loss"], d1["exposure"]) == pytest.approx(
        (0.0, 0.0, 150.0)
    )


# -------------------------------------------------------------------- mitigation
def test_dual_qualification_removes_the_plant_spof(
    cfgs: tuple[OptimizationConfig, SimulationConfig],
) -> None:
    inst = tie_broken()
    assert single_source_products(inst) == {"C": "P1"}
    _, fi, actions = dual_qualify(inst, inst, DESIGN, share=1.0)
    assert actions == [
        {"product_id": "C", "sole_plant": "P1", "backup_plant": "P2", "backup_capacity": 150.0}
    ]
    assert structural_spofs(fi, DESIGN).empty
    sol = evaluate_design(fi.plant_outage("P1"), DESIGN, cfgs[0])
    assert not sol.u  # P2 now covers both products


def _tight(inst: NetworkInstance) -> NetworkInstance:
    """DC capacity 400 each: two DCs cover 450 units, but not after losing one."""
    return replace(inst, cap_d={d: 400.0 for d in inst.dcs},
                   cold_d={"D1": 400.0, "D2": 400.0, "D3": 0.0})  # fmt: skip


def test_n_minus_1_opens_a_spare_dc(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> None:
    inst = _tight(regression_instance())
    plain = solve_network_design(inst, cfgs[0])
    assert plain.open_dcs == {"D1", "D2"}
    res = solve_network_design(inst, cfgs[0], opts=ModelOptions(n_minus_1_dc=True))
    costs = validate_solution(inst, res)
    assert res.open_dcs == {"D1", "D2", "D3"}
    assert costs["total"] == pytest.approx(2950.0)  # matches the enumeration table
    for d in res.open_dcs:  # survives the loss of any one DC
        lost = evaluate_design(inst.dc_outage(d), res.open_dcs, cfgs[0])
        assert not lost.u


def test_mitigation_table(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> None:
    inst = _tight(tie_broken())
    df, _ = run_mitigations(inst, inst, DESIGN, cfgs[0], cfgs[1], 1.0)
    df = df.set_index("mitigation")
    assert df.loc["base", "cost_premium_pct"] == 0.0
    assert df.loc["n_minus_1", "design_cost"] >= df.loc["base", "design_cost"] - 1e-6
    both = df.loc["n_minus_1+dual_qualification"]
    assert both["failures_with_shortfall"] == 0  # no single plant or DC failure hurts
    assert both["worst_failure_reduction_pct"] == pytest.approx(100.0)
    assert df.loc["base", "failures_with_shortfall"] > 0
    assert cost_breakdown(inst, evaluate_design(inst, DESIGN, cfgs[0]))["total"] == pytest.approx(
        df.loc["base", "design_cost"]
    )


def test_graph_stage_end_to_end(small_config_path: Path) -> None:
    from pharmanet.data.generate import generate
    from pharmanet.pipeline import (
        analyze_graph,
        forecast_and_safety_stock,
        optimize_network,
        simulate_design,
    )

    cfg = load_config(small_config_path)
    ds = generate(cfg)
    fc = forecast_and_safety_stock(cfg, ds)
    opt = optimize_network(cfg, ds, fc.forecasts, fc.selection, fc.safety_stock)
    dd = opt.design_demand.merge(
        fc.safety_stock[["region_id", "product_id", "sigma_monthly"]],
        on=["region_id", "product_id"],
    )
    sim = simulate_design(cfg, ds, dd, opt.milp.open_dcs)
    out = analyze_graph(cfg, ds, opt.design_demand, opt.milp.open_dcs, sim.disruptions)
    assert out.summary["cross_check"]["lower_bound_holds"]
    assert out.summary["cross_check"]["upper_bound_holds"]
    assert set(out.mitigations["mitigation"]) == {
        "base", "n_minus_1", "dual_qualification", "n_minus_1+dual_qualification"
    }  # fmt: skip
    assert isinstance(out.node_risk, pd.DataFrame) and len(out.node_risk) > 0
