"""Two-stage stochastic design: hand-checked example (docs/formulation.md),
reduction to the deterministic model, and RP/EEV/WS ordering properties."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pharmanet.config import OptimizationConfig, load_config
from pharmanet.optimization.stochastic import (
    WeightedScenario,
    evaluate_design_on,
    expected,
    solve_stochastic,
    wait_and_see,
)
from pharmanet.simulation.scenarios import simulation_rng, stochastic_scenarios
from regression_instance import regression_instance

DET = frozenset({"D1", "D2"})  # deterministic optimum of the regression instance


@pytest.fixture
def ocfg(small_config_path: Path) -> OptimizationConfig:
    return load_config(small_config_path).optimization


def _a_b(q: float) -> list[WeightedScenario]:
    """Scenario A: base. Scenario B (probability q): D1 down."""
    inst = regression_instance()
    return [WeightedScenario("A", 1 - q, inst), WeightedScenario("B", q, inst.dc_outage("D1"))]


def test_single_scenario_equals_deterministic(ocfg: OptimizationConfig) -> None:
    res = solve_stochastic([WeightedScenario("only", 1.0, regression_instance())], ocfg)
    assert res.open_dcs == DET
    assert res.objective == pytest.approx(2850.0)


def test_hand_example_switches_design(ocfg: OptimizationConfig) -> None:
    """cost({D1,D2}) = 2850 + 600q, cost({D2}) = 2950: switch above q = 1/6."""
    low = solve_stochastic(_a_b(0.1), ocfg)
    assert low.open_dcs == DET and low.objective == pytest.approx(2910.0)
    high = solve_stochastic(_a_b(0.5), ocfg)
    assert high.open_dcs == {"D2"} and high.objective == pytest.approx(2950.0)


def test_hand_example_vss_and_evpi(ocfg: OptimizationConfig) -> None:
    """q = 0.5: RP 2950, EEV 3150 -> VSS 200; WS 0.5*2850 + 0.5*2950 = 2900 -> EVPI 50."""
    sc = _a_b(0.5)
    rp = solve_stochastic(sc, ocfg).objective
    eev = expected(evaluate_design_on(sc, DET, ocfg, n_jobs=1), "total_cost")
    ws = expected(wait_and_see(sc, ocfg, n_jobs=1), "ws_cost")
    assert (rp, eev, ws) == pytest.approx((2950.0, 3150.0, 2900.0))
    assert eev - rp == pytest.approx(200.0) and rp - ws == pytest.approx(50.0)


def test_scenario_costs_are_validated_and_consistent(ocfg: OptimizationConfig) -> None:
    res = solve_stochastic(_a_b(0.5), ocfg)
    sc = res.scenario_costs.set_index("scenario")
    assert sc.loc["A", "recourse_cost"] + 350 == pytest.approx(2950.0)  # {D2} fixed = 350
    assert res.fixed_cost + res.expected_recourse == pytest.approx(res.objective)
    assert res.stats["n_binary"] == 3


def test_probabilities_must_sum_to_one(ocfg: OptimizationConfig) -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        solve_stochastic([WeightedScenario("x", 0.7, regression_instance())], ocfg)


def test_ordering_ws_le_rp_le_eev_on_sampled_scenarios(
    ocfg: OptimizationConfig, small_config_path: Path
) -> None:
    """Holds for any scenario set: perfect foresight <= here-and-now <= fixed design."""
    inst = regression_instance()
    fc = pd.DataFrame(
        {
            "region_id": [r for r in inst.regions for _ in inst.products],
            "product_id": list(inst.products) * len(inst.regions),
            "annual_forecast": [inst.dem[(r, k)] for r in inst.regions for k in inst.products],
            "sigma_monthly": 8.0,
        }
    )
    sc = stochastic_scenarios(inst, fc, 6, 0.2, 0.5, 1.0, simulation_rng(1), "t")
    assert len(sc) == 6 + 2 + 3 and sum(s.probability for s in sc) == pytest.approx(1.0)
    rp = solve_stochastic(sc, ocfg).objective
    eev = expected(evaluate_design_on(sc, DET, ocfg, 1), "total_cost")
    ws = expected(wait_and_see(sc, ocfg, 1), "ws_cost")
    assert ws <= rp + 1e-6 <= eev + 2e-6


def test_stochastic_stage_end_to_end(small_config_path: Path) -> None:
    from pharmanet.data.generate import generate
    from pharmanet.pipeline import forecast_and_safety_stock, optimize_network, stochastic_design

    cfg = load_config(small_config_path)
    ds = generate(cfg)
    fc = forecast_and_safety_stock(cfg, ds)
    opt = optimize_network(cfg, ds, fc.forecasts, fc.selection, fc.safety_stock)
    dd = opt.design_demand.merge(
        fc.safety_stock[["region_id", "product_id", "sigma_monthly"]],
        on=["region_id", "product_id"],
    )
    summary, tables = stochastic_design(cfg, ds, dd, opt.milp.open_dcs)
    ins = summary["in_sample"]
    assert ins["VSS"] >= -1e-6 and ins["EVPI"] >= -1e-6
    assert set(tables) >= {"sp_in_sample", "deterministic_in_sample", "sp_out_of_sample"}
