"""Tests for the synthetic data generator.

Two kinds of tests: (1) structural invariants on the default instances, and
(2) "switch off the randomness" tests where noise/trend/seasonality are set to
fixed values so the output can be checked against a closed-form expectation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pharmanet.config import PharmaNetConfig, load_config
from pharmanet.data.generate import (
    Dataset,
    DataValidationError,
    annual_demand,
    generate,
    validate_dataset,
)


def _cfg(raw: dict[str, Any], **overrides: Any) -> PharmaNetConfig:
    """Build a config from a raw dict with dotted-path overrides, e.g. run__seed=1."""
    data = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return PharmaNetConfig.model_validate(data)


@pytest.fixture(scope="module")
def base_ds() -> Dataset:
    return generate(load_config(Path(__file__).parents[1] / "configs" / "base.yaml"))


# ------------------------------------------------------------------ reproducibility
def test_same_seed_same_data(small_config_path: Path) -> None:
    cfg = load_config(small_config_path)
    a, b = generate(cfg), generate(cfg)
    for name, table in a.tables().items():
        pd.testing.assert_frame_equal(table, b.tables()[name], obj=name)


def test_different_seed_different_data(small_config_dict: dict[str, Any]) -> None:
    a = generate(_cfg(small_config_dict, run__seed=1))
    b = generate(_cfg(small_config_dict, run__seed=2))
    assert not a.demand["demand"].equals(b.demand["demand"])


# ------------------------------------------------------------------------ structure
def test_shapes_match_config(base_ds: Dataset) -> None:
    assert len(base_ds.plants) == 4
    assert len(base_ds.dcs) == 8
    assert len(base_ds.regions) == 20
    assert len(base_ds.products) == 12
    assert len(base_ds.demand) == 20 * 12 * 36
    assert len(base_ds.lanes_inbound) == 4 * 8
    assert len(base_ds.lanes_outbound) == 8 * 20


def test_cold_fraction_is_exact(base_ds: Dataset) -> None:
    assert base_ds.products["is_cold"].sum() == round(0.30 * 12)


def test_demand_nonnegative_integers(base_ds: Dataset) -> None:
    assert base_ds.demand["demand"].dtype == np.int64
    assert (base_ds.demand["demand"] >= 0).all()


def test_months_are_consecutive_month_starts(base_ds: Dataset) -> None:
    months = pd.DatetimeIndex(sorted(base_ds.demand["month"].unique()))
    expected = pd.date_range("2023-01-01", periods=36, freq="MS")
    assert months.equals(expected)


def test_intermittent_series_count(base_ds: Dataset) -> None:
    has_zero = base_ds.demand.groupby(["region_id", "product_id"])["demand"].apply(
        lambda s: (s == 0).any()
    )
    # 10% of 240 series are intermittent; regular series (base >= 50) never hit zero.
    assert has_zero.sum() == round(0.10 * 240)


def test_penalty_exceeds_any_path(base_ds: Dataset) -> None:
    li, lo = base_ds.lanes_inbound, base_ds.lanes_outbound
    worst_cold = li["cold_cost_per_unit"].max() + lo["cold_cost_per_unit"].max()
    assert (base_ds.products["unmet_penalty"] > worst_cold).all()


def test_aggregate_capacity_covers_demand(base_ds: Dataset) -> None:
    by_product = annual_demand(base_ds.demand, ["product_id"])
    plant_cap = base_ds.plant_products.groupby("product_id")["capacity"].sum()
    assert (plant_cap.reindex(by_product.index) >= 1.2 * by_product - 1).all()
    assert base_ds.dcs["throughput_capacity"].sum() >= by_product.sum()


# ------------------------------------------------------------- feasibility guarantees
def test_every_product_gets_a_plant_even_with_low_qualification(
    base_config_dict: dict[str, Any],
) -> None:
    ds = generate(_cfg(base_config_dict, generation__plants__qualification_prob=0.01))
    per_product = ds.plant_products.groupby("product_id")["qualified"].sum()
    assert (per_product >= 1).all()


def test_at_least_one_cold_dc_when_cold_products_exist(base_config_dict: dict[str, Any]) -> None:
    # no_cold_fraction=1.0 would remove cold storage everywhere; the generator must keep one.
    # Cold capacity is then too small in aggregate, so validation is expected to flag it.
    cfg = _cfg(base_config_dict, generation__dcs__no_cold_fraction=1.0)
    with pytest.raises(DataValidationError, match="cold capacity"):
        generate(cfg)


# ----------------------------------------------------- closed-form (noise-free) checks
def _noise_free(raw: dict[str, Any], **extra: Any) -> PharmaNetConfig:
    fixed = {
        "generation__demand__base_level": {"low": 100.0, "high": 100.0},
        "generation__demand__noise_cv": {"low": 0.0, "high": 0.0},
        "generation__demand__annual_trend": {"low": 0.0, "high": 0.0},
        "generation__demand__seasonal_amplitude": {"low": 0.0, "high": 0.0},
    }
    fixed.update(extra)
    return _cfg(raw, **fixed)


def test_seasonality_bounds(small_config_dict: dict[str, Any]) -> None:
    amp = {"low": 0.2, "high": 0.2}
    ds = generate(_noise_free(small_config_dict, generation__demand__seasonal_amplitude=amp))
    d = ds.demand["demand"]
    assert d.min() >= 80 and d.max() <= 120
    assert d.max() - d.min() >= 35  # the wave actually spans most of +-20%
    # Over whole years the sine averages out to the base level.
    assert abs(d.mean() - 100) < 1


def test_trend_compounds_annually(small_config_dict: dict[str, Any]) -> None:
    trend = {"low": 0.10, "high": 0.10}
    ds = generate(_noise_free(small_config_dict, generation__demand__annual_trend=trend))
    s = ds.demand.groupby("month")["demand"].mean()
    assert s.iloc[24] / s.iloc[0] == pytest.approx(1.21, abs=0.01)


def test_flat_when_everything_off(small_config_dict: dict[str, Any]) -> None:
    ds = generate(_noise_free(small_config_dict))
    assert (ds.demand["demand"] == 100).all()


# ------------------------------------------------------------------------ validation
def test_validation_passes_on_generated(base_ds: Dataset) -> None:
    validate_dataset(base_ds)


def _replace(ds: Dataset, **tables: pd.DataFrame) -> Dataset:
    return Dataset(**{**ds.tables(), **tables})


def test_validation_catches_unqualified_product(base_ds: Dataset) -> None:
    pp = base_ds.plant_products.copy()
    mask = pp["product_id"] == "SKU01"
    pp.loc[mask, ["qualified", "capacity"]] = [False, 0.0]
    with pytest.raises(DataValidationError, match="no qualified plant"):
        validate_dataset(_replace(base_ds, plant_products=pp))


def test_validation_catches_negative_demand(base_ds: Dataset) -> None:
    demand = base_ds.demand.copy()
    demand.loc[0, "demand"] = -5
    with pytest.raises(DataValidationError, match="negative values"):
        validate_dataset(_replace(base_ds, demand=demand))


def test_validation_catches_cold_on_non_cold_dc(base_ds: Dataset) -> None:
    dcs = base_ds.dcs.copy()
    idx = dcs.index[~dcs["has_cold"]][0]
    dcs.loc[idx, "cold_capacity"] = 10.0
    with pytest.raises(DataValidationError, match="without cold storage"):
        validate_dataset(_replace(base_ds, dcs=dcs))


def test_validation_reports_all_errors_at_once(base_ds: Dataset) -> None:
    demand = base_ds.demand.copy()
    demand.loc[0, "demand"] = -5
    dcs = base_ds.dcs.copy()
    dcs.loc[0, "fixed_cost"] = 0.0
    with pytest.raises(DataValidationError) as exc:
        validate_dataset(_replace(base_ds, demand=demand, dcs=dcs))
    assert "negative values" in str(exc.value) and "fixed cost" in str(exc.value)
