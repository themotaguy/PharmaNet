"""Tests for the safety stock formula and fastest-path lead times."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmanet.config import load_config
from pharmanet.data.generate import Dataset, generate
from pharmanet.inventory.safety_stock import (
    compute_safety_stock,
    fastest_path_lead_times,
    safety_stock,
    z_value,
)


def test_z_values() -> None:
    assert z_value(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert z_value(0.5) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        z_value(1.0)


@pytest.mark.parametrize(
    ("sigma", "mean", "lt", "lt_sd", "expected_over_z"),
    [
        (10.0, 100.0, 1.0, 0.0, 10.0),  # demand risk only: sigma * sqrt(L)
        (10.0, 100.0, 0.25, 0.0, 5.0),  # a quarter-month lead time halves it
        (0.0, 100.0, 1.0, 0.1, 10.0),  # lead-time risk only: d_bar * sigma_L
        (3.0, 40.0, 1.0, 0.1, 5.0),  # both: sqrt(9 + 16) = 5
    ],
)
def test_safety_stock_hand_values(
    sigma: float, mean: float, lt: float, lt_sd: float, expected_over_z: float
) -> None:
    z = 2.0
    ss = safety_stock(np.array([sigma]), np.array([mean]), np.array([lt]), np.array([lt_sd]), z)
    assert ss[0] == pytest.approx(z * expected_over_z)


def test_safety_stock_rejects_negative_inputs() -> None:
    one = np.array([1.0])
    with pytest.raises(ValueError, match="sigma_d"):
        safety_stock(-one, one, one, one, 1.0)


@pytest.fixture(scope="module")
def base_ds() -> Dataset:
    return generate(load_config(Path(__file__).parents[1] / "configs" / "base.yaml"))


def test_fastest_path_matches_brute_force(base_ds: Dataset) -> None:
    lt = fastest_path_lead_times(base_ds).set_index(["region_id", "product_id"])
    assert len(lt) == len(base_ds.regions) * len(base_ds.products)

    lin = base_ds.lanes_inbound.set_index(["plant_id", "dc_id"])["lead_time_mean_days"]
    lout = base_ds.lanes_outbound.set_index(["dc_id", "region_id"])["lead_time_mean_days"]
    pp = base_ds.plant_products
    cold_dcs = set(base_ds.dcs.loc[base_ds.dcs["has_cold"], "dc_id"])
    for _, prod in base_ds.products.iterrows():
        k = prod["product_id"]
        plants = pp.loc[pp["qualified"] & (pp["product_id"] == k), "plant_id"]
        dcs = cold_dcs if prod["is_cold"] else set(base_ds.dcs["dc_id"])
        for r in base_ds.regions["region_id"]:
            best = min(lin[(p, d)] + lout[(d, r)] for p in plants for d in dcs)
            assert lt.loc[(r, k), "lt_mean_days"] == pytest.approx(best)
            assert lt.loc[(r, k), "dc_id"] in dcs  # cold products only via cold DCs


def _toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = {"region_id": ["R1"], "product_id": ["P1"]}
    months = pd.date_range("2026-01-01", periods=12, freq="MS")
    fc = pd.DataFrame({"region_id": "R1", "product_id": "P1", "month": months, "forecast": 100.0})
    sel = pd.DataFrame({**key, "best_model": ["ets"], "rmse": [20.0]})
    lt = pd.DataFrame(
        {
            **key,
            "plant_id": ["PL1"],
            "dc_id": ["D1"],
            "lt_mean_days": [365.25 / 12],
            "lt_std_days": [0.0],
        }
    )
    return fc, sel, lt


def test_compute_safety_stock_toy() -> None:
    fc, sel, lt = _toy_inputs()
    out = compute_safety_stock(fc, sel, lt, 0.95).iloc[0]
    assert out["annual_forecast"] == pytest.approx(1200.0)
    assert out["safety_stock"] == pytest.approx(z_value(0.95) * 20.0)  # L = 1 month
    assert out["design_demand"] == pytest.approx(1200.0 + out["safety_stock"])


def test_higher_service_level_means_more_stock() -> None:
    fc, sel, lt = _toy_inputs()
    levels = [0.8, 0.9, 0.95, 0.99]
    ss = [compute_safety_stock(fc, sel, lt, s)["safety_stock"].item() for s in levels]
    assert ss == sorted(ss) and len(set(ss)) == len(ss)


def test_missing_supply_path_fails_loudly() -> None:
    fc, sel, lt = _toy_inputs()
    with pytest.raises(ValueError, match="No feasible supply path"):
        compute_safety_stock(fc, sel, lt.iloc[0:0], 0.95)


def test_flow_weighted_lead_time_single_and_split(base_ds: Dataset) -> None:
    from pharmanet.inventory.safety_stock import flow_weighted_lead_times

    lin = base_ds.lanes_inbound.set_index(["plant_id", "dc_id"])
    lout = base_ds.lanes_outbound.set_index(["dc_id", "region_id"])
    fb = fastest_path_lead_times(base_ds)
    k = "SKU02"
    # One path: lead time = inbound leg + outbound leg.
    one = flow_weighted_lead_times(
        base_ds, {("PL01", "DC01", k): 10.0}, {("DC01", "RG01", k): 10.0}, fb
    ).set_index(["region_id", "product_id"])
    leg = (
        lin.loc[("PL01", "DC01"), "lead_time_mean_days"]
        + lout.loc[("DC01", "RG01"), "lead_time_mean_days"]
    )
    assert one.loc[("RG01", k), "lt_mean_days"] == pytest.approx(leg)
    # 50/50 split over two DCs: mean is the average of the two paths.
    split = flow_weighted_lead_times(
        base_ds,
        {("PL01", "DC01", k): 5.0, ("PL01", "DC02", k): 5.0},
        {("DC01", "RG01", k): 5.0, ("DC02", "RG01", k): 5.0},
        fb,
    ).set_index(["region_id", "product_id"])
    leg2 = (
        lin.loc[("PL01", "DC02"), "lead_time_mean_days"]
        + lout.loc[("DC02", "RG01"), "lead_time_mean_days"]
    )
    assert split.loc[("RG01", k), "lt_mean_days"] == pytest.approx((leg + leg2) / 2)
    # Untouched pairs keep the fastest-path fallback.
    other = fb.set_index(["region_id", "product_id"]).loc[("RG02", k), "lt_mean_days"]
    assert split.loc[("RG02", k), "lt_mean_days"] == pytest.approx(other)
