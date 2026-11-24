"""Tests for forecasting models, metrics, backtest mechanics, and model selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmanet.config import ForecastConfig, load_config
from pharmanet.forecasting import models as models_mod
from pharmanet.forecasting.backtest import (
    bias,
    holdout_selection_summary,
    mape,
    parallel_map,
    rmse,
    rolling_origin,
    select_models,
    wape,
)
from pharmanet.forecasting.models import forecast, seasonal_naive


@pytest.fixture
def fcfg(small_config_path: Path) -> ForecastConfig:
    return load_config(small_config_path).forecasting


def _seasonal_series(n: int = 36, base: float = 100.0, amp: float = 0.2, growth: float = 0.0):
    t = np.arange(n)
    return base * (1 + growth) ** (t / 12) * (1 + amp * np.sin(2 * np.pi * t / 12))


# ------------------------------------------------------------------------- models
def test_seasonal_naive_repeats_last_season(fcfg: ForecastConfig) -> None:
    y = np.arange(24, dtype=float)
    np.testing.assert_array_equal(seasonal_naive(y, 3, fcfg), [12, 13, 14])
    # Beyond one season, the last observed season repeats.
    assert seasonal_naive(y, 14, fcfg)[12] == 12


def test_seasonal_naive_needs_one_season(fcfg: ForecastConfig) -> None:
    with pytest.raises(ValueError, match="needs >= 12"):
        seasonal_naive(np.ones(11), 1, fcfg)


@pytest.mark.parametrize("name", ["ets", "sarima"])
def test_models_learn_clean_seasonality(name: str, fcfg: ForecastConfig) -> None:
    y = _seasonal_series(48)
    truth = _seasonal_series(60)[48:]
    res = forecast(name, y[:48], 12, fcfg)  # type: ignore[arg-type]
    assert not res.fell_back
    assert wape(truth, res.yhat) < 0.05


def test_forecasts_are_clipped_at_zero(fcfg: ForecastConfig) -> None:
    y = np.linspace(500, 5, 36)  # steep decline: unclipped trend would go negative
    res = forecast("ets", y, 12, fcfg)
    assert (res.yhat >= 0).all()


def test_failed_model_falls_back_to_naive(
    fcfg: ForecastConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(history: np.ndarray, horizon: int, cfg: ForecastConfig) -> np.ndarray:
        raise ValueError("did not converge")

    monkeypatch.setitem(models_mod.MODELS, "ets", broken)
    y = _seasonal_series(24)
    res = forecast("ets", y, 3, fcfg)
    assert res.fell_back
    np.testing.assert_allclose(res.yhat, seasonal_naive(y, 3, fcfg))


def test_non_finite_output_counts_as_failure(
    fcfg: ForecastConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(models_mod.MODELS, "ets", lambda h, n, c: np.full(n, np.nan))
    assert forecast("ets", _seasonal_series(24), 2, fcfg).fell_back


# ------------------------------------------------------------------------ metrics
def test_metrics_hand_values() -> None:
    a = np.array([100.0, 0.0, 50.0])
    f = np.array([90.0, 10.0, 60.0])
    assert wape(a, f) == pytest.approx(30 / 150)
    assert mape(a, f) == pytest.approx((0.10 + 0.20) / 2)  # zero-actual month excluded
    assert bias(a, f) == pytest.approx(10 / 150)  # over-forecast -> positive
    assert rmse(a, f) == pytest.approx(10.0)


def test_metric_edge_cases() -> None:
    z = np.zeros(3)
    assert wape(z, z) == 0.0
    assert wape(z, np.ones(3)) == float("inf")
    assert np.isnan(mape(z, np.ones(3)))


# ----------------------------------------------------------------------- backtest
def test_rolling_origin_never_sees_the_future(
    fcfg: ForecastConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Data-leakage guard: at origin t the model may only see y[0:t]."""
    seen: list[int] = []

    def spy(history: np.ndarray, horizon: int, cfg: ForecastConfig) -> np.ndarray:
        seen.append(len(history))
        assert history[-1] == len(history) - 1  # last visible value is y[t-1]
        return np.zeros(horizon)

    monkeypatch.setitem(models_mod.MODELS, "ets", spy)
    y = np.arange(36, dtype=float)
    actual, _, _ = rolling_origin(y, "ets", fcfg)
    assert seen == list(range(24, 36))
    np.testing.assert_array_equal(actual, y[24:])


def test_naive_is_perfect_on_pure_seasonality(fcfg: ForecastConfig) -> None:
    y = _seasonal_series(36)
    actual, fc, _ = rolling_origin(y, "seasonal_naive", fcfg)
    assert wape(actual, fc) == pytest.approx(0.0, abs=1e-12)


def test_parallel_map_matches_serial() -> None:
    items = list(range(-10, 10))
    assert parallel_map(abs, items, 2) == parallel_map(abs, items, 1)


# ---------------------------------------------------------------------- selection
def _metrics(
    rows: list[tuple[str, float]], region: str = "R", volume: float = 100.0
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "region_id": region,
                "product_id": "P",
                "model": m,
                "wape": w,
                "mape": w,
                "bias": 0.0,
                "rmse": 1.0,
                "abs_error_sum": w * volume,
                "actual_sum": volume,
            }
            for m, w in rows
        ]
    )


def test_selection_picks_lowest_wape(fcfg: ForecastConfig) -> None:
    sel = select_models(_metrics([("seasonal_naive", 0.3), ("ets", 0.1), ("sarima", 0.2)]), fcfg)
    assert sel["best_model"].item() == "ets"
    assert sel["naive_wape"].item() == 0.3


def test_selection_tie_goes_to_simpler_model(fcfg: ForecastConfig) -> None:
    sel = select_models(_metrics([("sarima", 0.2), ("ets", 0.2), ("seasonal_naive", 0.2)]), fcfg)
    assert sel["best_model"].item() == "seasonal_naive"


def test_global_strategy_uses_pooled_error(fcfg: ForecastConfig) -> None:
    """SARIMA wins two small series narrowly; ETS wins one huge series by a lot.

    Per-series picks SARIMA twice. Global pools by volume, so ETS (much better on
    most of the units) is chosen for every series.
    """
    metrics = pd.concat(
        [
            _metrics([("seasonal_naive", 0.5), ("ets", 0.21), ("sarima", 0.20)], "R1", 10),
            _metrics([("seasonal_naive", 0.5), ("ets", 0.21), ("sarima", 0.20)], "R2", 10),
            _metrics([("seasonal_naive", 0.5), ("ets", 0.10), ("sarima", 0.40)], "R3", 1000),
        ]
    )
    per = select_models(metrics, fcfg)
    assert per["best_model"].tolist() == ["sarima", "sarima", "ets"]
    glob = select_models(metrics, fcfg.model_copy(update={"selection_strategy": "global"}))
    assert set(glob["best_model"]) == {"ets"}
    assert glob["wape"].tolist() == [0.21, 0.21, 0.10]  # each series keeps its own ETS score


def test_global_tie_goes_to_simpler_model(fcfg: ForecastConfig) -> None:
    metrics = _metrics([("sarima", 0.2), ("ets", 0.2), ("seasonal_naive", 0.3)])
    glob = select_models(metrics, fcfg.model_copy(update={"selection_strategy": "global"}))
    assert glob["best_model"].item() == "ets"


def test_holdout_selection_uses_only_early_months(fcfg: ForecastConfig) -> None:
    """ETS is perfect early and terrible late; naive is the opposite.

    Selection on early months picks ETS, so the honest holdout score must show
    selection doing *worse* than naive. If the late months leaked into the
    choice, it would pick naive and look perfect.
    """
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    rows = []
    for i, m in enumerate(months):
        late = i >= 6
        for model, err in [("ets", 50.0 if late else 0.0), ("seasonal_naive", 0.0 if late else 50)]:
            rows.append(
                {
                    "region_id": "R",
                    "product_id": "P",
                    "model": model,
                    "month": m,
                    "actual": 100.0,
                    "forecast": 100.0 + err,
                    "series_fallbacks": 0,
                }
            )
    out = holdout_selection_summary(pd.DataFrame(rows), fcfg)
    assert out["holdout_wape_naive"] == 0.0
    assert out["holdout_wape_per_series"] == pytest.approx(0.5)
    assert out["holdout_wape_global"] == pytest.approx(0.5)
    assert out["holdout_global_model"] == "ets"
