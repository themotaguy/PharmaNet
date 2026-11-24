"""Forecasting models behind one common signature.

Every model is a function ``(history, horizon, cfg) -> forecast`` operating on a
plain NumPy array of monthly values. Keeping them interchangeable is what makes
the backtest generic: it loops over model *names* and never cares which model
it is running.

Statistical models can fail to fit on short or degenerate series (e.g. an
all-zero history). ``forecast`` catches that, falls back to seasonal naive,
and *reports* the fallback rather than hiding it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

# Imported at module level on purpose: statsmodels installs
# ``warnings.simplefilter("always", ...)`` rules on first import. Importing
# inside a ``catch_warnings`` block would put those rules *ahead of* our
# "ignore" filter and let fit warnings leak.
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pharmanet.config import ForecastConfig, ForecastModelName

ModelFn = Callable[[np.ndarray, int, ForecastConfig], np.ndarray]


def seasonal_naive(history: np.ndarray, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    """Forecast each future month as the value from the same month one season earlier.

    ``y_hat[T+h] = y[T + h - m * ceil(h/m)]``: for h <= m that is simply
    "same month last year"; beyond one season the last observed season repeats.
    """
    m = cfg.seasonal_period
    if len(history) < m:
        raise ValueError(f"seasonal_naive needs >= {m} observations, got {len(history)}")
    last_season = history[-m:]
    return np.array([last_season[h % m] for h in range(horizon)], dtype=float)


def ets(history: np.ndarray, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    """Holt-Winters exponential smoothing (level + optionally damped trend + seasonality)."""
    e = cfg.ets
    model = ExponentialSmoothing(
        history.astype(float),
        trend=e.trend,
        damped_trend=e.damped_trend if e.trend else False,
        seasonal=e.seasonal,
        seasonal_periods=cfg.seasonal_period if e.seasonal else None,
        initialization_method="estimated",
    )
    return np.asarray(model.fit().forecast(horizon), dtype=float)


def sarima(history: np.ndarray, horizon: int, cfg: ForecastConfig) -> np.ndarray:
    """Seasonal ARIMA via statsmodels' state-space SARIMAX (no exogenous regressors)."""
    s = cfg.sarima
    model = SARIMAX(
        history.astype(float),
        order=s.order,
        seasonal_order=(*s.seasonal_order, cfg.seasonal_period),
        trend=s.trend,
    )
    return np.asarray(model.fit(disp=False).forecast(horizon), dtype=float)


MODELS: dict[ForecastModelName, ModelFn] = {
    "seasonal_naive": seasonal_naive,
    "ets": ets,
    "sarima": sarima,
}


class ForecastResult(NamedTuple):
    yhat: np.ndarray  # forecast values
    fell_back: bool  # True if the requested model failed and seasonal naive was used


def forecast(
    name: ForecastModelName, history: np.ndarray, horizon: int, cfg: ForecastConfig
) -> ForecastResult:
    """Run one model safely: suppress fit warnings, clip at zero, fall back on failure.

    Demand cannot be negative, so forecasts are clipped at 0. Non-finite output
    (NaN/inf from a diverged fit) counts as a failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence chatter on short series
            values = MODELS[name](history, horizon, cfg)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} produced non-finite forecasts")
        fell_back = False
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, IndexError):
        if name == "seasonal_naive":
            raise
        values = seasonal_naive(history, horizon, cfg)
        fell_back = True
    return ForecastResult(np.clip(values, 0.0, None), fell_back)
