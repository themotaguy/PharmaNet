"""Simulation tests: hand-checked disruptions, demand sampling statistics,
fixed-design evaluation, and sensitivity properties."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmanet.config import OptimizationConfig, SimulationConfig, load_config
from pharmanet.optimization.network_design import solve_network_design
from pharmanet.optimization.solution import cost_breakdown
from pharmanet.optimization.validate import SolutionValidationError, validate_solution
from pharmanet.simulation.evaluate import evaluate_design, evaluate_scenarios, run_sensitivity
from pharmanet.simulation.scenarios import disruption_scenarios, sample_demand, simulation_rng
from regression_instance import regression_instance

DESIGN = frozenset({"D1", "D2"})  # optimum of the regression instance


@pytest.fixture
def cfgs(small_config_path: Path) -> tuple[OptimizationConfig, SimulationConfig]:
    cfg = load_config(small_config_path)
    return cfg.optimization, cfg.simulation


# ------------------------------------------------- hand-checked disruptions (D1, D2 open)
def test_fixed_design_reproduces_milp_cost(
    cfgs: tuple[OptimizationConfig, SimulationConfig],
) -> None:
    sol = evaluate_design(regression_instance(), DESIGN, cfgs[0])
    assert cost_breakdown(regression_instance(), sol)["total"] == pytest.approx(2850.0)


def test_plant_outage_by_hand(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> None:
    """P1 down: nobody else is qualified for C, so all 150 cold units are unmet.

    A must come from P2: via D1 costs 3+out, via D2 costs 2+out.
    R1: min(3+1, 2+3)=4, R2: min(3+3, 2+1)=3, R3: min(3+5, 2+3)=5 -> 100*(4+3+5)=1200.
    Total = fixed 850 + 1200 + penalty 150*100 = 17050.
    """
    inst = regression_instance().plant_outage("P1")
    sol = evaluate_design(inst, DESIGN, cfgs[0])
    assert sum(sol.u.values()) == pytest.approx(150.0)
    assert set(sol.u) == {("R1", "C"), ("R2", "C"), ("R3", "C")}
    assert cost_breakdown(inst, sol)["total"] == pytest.approx(17050.0)


def test_dc_outage_by_hand(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> None:
    """D1 down: everything goes through D2 (= the {D2} design's flows) but both
    fixed costs are still paid: 850 + 1300 + 1300 = 3450, nothing unmet."""
    inst = regression_instance().dc_outage("D1")
    sol = evaluate_design(inst, DESIGN, cfgs[0])
    assert not sol.u
    assert cost_breakdown(inst, sol)["total"] == pytest.approx(3450.0)


def test_cold_lane_failure_by_hand(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> None:
    """Cold lane D2->R3 fails: R3's 50 cold units reroute via D1 at 12 instead of 10."""
    inst = regression_instance().block_outbound(frozenset({("D2", "R3", "C")}))
    sol = evaluate_design(inst, DESIGN, cfgs[0])
    assert sol.z.get(("D2", "R3", "C"), 0.0) == 0.0
    assert cost_breakdown(inst, sol)["total"] == pytest.approx(2950.0)


def test_partial_plant_outage_scales_capacity() -> None:
    inst = regression_instance().plant_outage("P1", capacity_loss=0.25)
    assert inst.cap_pk[("P1", "C")] == pytest.approx(750.0)
    assert inst.cap_pk[("P2", "A")] == pytest.approx(1000.0)  # other plants untouched


def test_disruption_list_targets_the_design(
    cfgs: tuple[OptimizationConfig, SimulationConfig],
) -> None:
    inst = regression_instance()
    base = evaluate_design(inst, DESIGN, cfgs[0])
    scen = disruption_scenarios(inst, base, cfgs[1])
    kinds = pd.Series([s.kind for s in scen]).value_counts().to_dict()
    # 2 plants; 2 open DCs (closed D3 is not a disruption); cold lanes actually used:
    # inbound P1->D1, P1->D2; outbound D1->R1, D2->R2, D2->R3.
    assert kinds == {"plant_outage": 2, "dc_outage": 2, "cold_lane_failure": 5}


def test_validator_rejects_flow_on_blocked_lane(
    cfgs: tuple[OptimizationConfig, SimulationConfig],
) -> None:
    base = evaluate_design(regression_instance(), DESIGN, cfgs[0])
    blocked = regression_instance().block_outbound(frozenset({("D2", "R3", "C")}))
    with pytest.raises(SolutionValidationError, match="blocked outbound lane"):
        validate_solution(blocked, base)


def test_evaluate_scenarios_parallel_matches_serial(
    cfgs: tuple[OptimizationConfig, SimulationConfig],
) -> None:
    inst = regression_instance()
    scen = disruption_scenarios(inst, evaluate_design(inst, DESIGN, cfgs[0]), cfgs[1])
    serial, _ = evaluate_scenarios(scen, DESIGN, cfgs[0], n_jobs=1)
    parallel, _ = evaluate_scenarios(scen, DESIGN, cfgs[0], n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


# -------------------------------------------------------------- demand sampling
def _forecast(sigma: float = 30.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_id": ["R1", "R2", "R1", "R2"],
            "product_id": ["A", "A", "B", "B"],
            "annual_forecast": [1200.0, 600.0, 2400.0, 300.0],
            "sigma_monthly": [sigma, sigma / 2, sigma * 2, sigma / 4],
        }
    )


def test_sampling_is_reproducible() -> None:
    _, a = sample_demand(_forecast(), 50, 0.5, simulation_rng(1))
    _, b = sample_demand(_forecast(), 50, 0.5, simulation_rng(1))
    np.testing.assert_array_equal(a, b)


def test_zero_error_gives_the_forecast() -> None:
    keys, draws = sample_demand(_forecast(sigma=0.0), 5, 0.5, simulation_rng(0))
    np.testing.assert_allclose(draws, np.tile([1200.0, 2400.0, 600.0, 300.0], (5, 1)))
    assert keys == [("R1", "A"), ("R1", "B"), ("R2", "A"), ("R2", "B")]


def test_draws_are_unbiased_with_the_right_spread() -> None:
    f = _forecast()
    _, draws = sample_demand(f, 40_000, 0.5, simulation_rng(3))
    mean = np.array([1200.0, 2400.0, 600.0, 300.0])
    sd = np.array([30.0, 60.0, 15.0, 7.5]) * np.sqrt(12)
    np.testing.assert_allclose(draws.mean(axis=0), mean, rtol=0.01)
    np.testing.assert_allclose(draws.std(axis=0), sd, rtol=0.03)
    assert (draws > 0).all()


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_common_shock_correlates_regions_of_same_product(rho: float) -> None:
    _, draws = sample_demand(_forecast(), 40_000, rho, simulation_rng(4))
    logs = np.log(draws)
    same_product = np.corrcoef(logs[:, 0], logs[:, 2])[0, 1]  # (R1,A) vs (R2,A)
    diff_product = np.corrcoef(logs[:, 0], logs[:, 1])[0, 1]  # (R1,A) vs (R1,B)
    assert same_product == pytest.approx(rho, abs=0.02)
    assert diff_product == pytest.approx(0.0, abs=0.02)


# ------------------------------------------------------------------ sensitivity
def test_sensitivity_regret_is_never_negative(
    cfgs: tuple[OptimizationConfig, SimulationConfig],
) -> None:
    ocfg, scfg = cfgs
    sens = run_sensitivity(regression_instance(), DESIGN, ocfg, scfg)
    assert len(sens) == 9
    assert (sens["regret"] >= -1e-6).all()
    at_one = sens[sens["multiplier"] == 1.0]
    assert (~at_one["design_changed"]).all() and (at_one["regret"].abs() < 1e-6).all()


def test_cheaper_dcs_open_more_dcs(cfgs: tuple[OptimizationConfig, SimulationConfig]) -> None:
    """With fixed costs quartered, D3's 300 transport saving beats its 100 fixed cost."""
    inst = regression_instance().scale_fixed_costs(0.25)
    sol = solve_network_design(inst, cfgs[0])
    assert sol.open_dcs == {"D1", "D2", "D3"}
