"""Rolling-origin backtesting, error metrics, and per-series model selection.

Rolling origin (a.k.a. time-series cross-validation): for each of the last
``backtest_months`` months t, fit on months [0, t) and forecast month t one
step ahead. Every test point is genuinely out-of-sample, and refitting at each
origin mimics how a planner re-forecasts every month.

We evaluate one-step-ahead because that is the error safety stock must absorb:
replenishment lead times in this network are days, i.e. shorter than a month.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from pharmanet.config import ForecastConfig, ForecastModelName
from pharmanet.forecasting.models import MODELS, forecast

SERIES_KEY = ["region_id", "product_id"]

T = TypeVar("T")
R = TypeVar("R")


# ------------------------------------------------------------------------ metrics
def wape(actual: np.ndarray, fc: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|A - F| / sum A.

    Robust to zero-demand months (unlike MAPE) and weights big-volume months
    more, which matches how cost and service actually accrue. If total actual
    demand is zero, WAPE is 0 for a perfect forecast and +inf otherwise.
    """
    denom = float(np.sum(actual))
    err = float(np.sum(np.abs(actual - fc)))
    if denom == 0:
        return 0.0 if err == 0 else float("inf")
    return err / denom


def mape(actual: np.ndarray, fc: np.ndarray) -> float:
    """Mean absolute percentage error over months with non-zero actuals (NaN if none).

    Zero-actual months are excluded because |A-F|/A is undefined there; this is
    exactly why MAPE is not the primary metric for intermittent demand.
    """
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - fc[mask]) / actual[mask]))


def bias(actual: np.ndarray, fc: np.ndarray) -> float:
    """Signed error relative to volume: sum(F - A) / sum A. Positive = over-forecasting."""
    denom = float(np.sum(actual))
    return float(np.sum(fc - actual)) / denom if denom else float("nan")


def rmse(actual: np.ndarray, fc: np.ndarray) -> float:
    """Root mean squared error, in units. Used as sigma_d for safety stock.

    RMSE (not the std of errors) is used deliberately: it includes bias, so a
    systematically off model gets more protection, not less.
    """
    return float(np.sqrt(np.mean((actual - fc) ** 2)))


def improvement_pct(baseline: float, new: float) -> float:
    """Percent reduction of an error metric vs a baseline (NaN if the baseline is 0)."""
    return float("nan") if baseline == 0 else 100 * (baseline - new) / baseline


# ------------------------------------------------------------------- parallelism
def parallel_map(fn: Callable[[T], R], items: Sequence[T], n_jobs: int | None) -> list[R]:
    """Map over items in worker processes (or serially if n_jobs == 1).

    Each series is independent, so the backtest is embarrassingly parallel.
    Results come back in input order, so output is deterministic regardless of
    scheduling.
    """
    workers = n_jobs or os.cpu_count() or 1
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunk))


def iter_series(demand: pd.DataFrame) -> Iterable[tuple[str, str, np.ndarray]]:
    """Yield (region_id, product_id, monthly values in time order)."""
    ordered = demand.sort_values([*SERIES_KEY, "month"])
    for (region, product), g in ordered.groupby(SERIES_KEY, sort=True):
        yield str(region), str(product), g["demand"].to_numpy(dtype=float)


# ---------------------------------------------------------------------- backtest
def rolling_origin(
    y: np.ndarray, name: ForecastModelName, cfg: ForecastConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    """One-step-ahead forecasts for each of the last ``backtest_months`` points.

    Returns (actuals, forecasts, number of origins where the model fell back).
    """
    n = cfg.backtest_months
    start = len(y) - n
    fcs = np.empty(n)
    fallbacks = 0
    for i, origin in enumerate(range(start, len(y))):
        res = forecast(name, y[:origin], 1, cfg)
        fcs[i] = res.yhat[0]
        fallbacks += int(res.fell_back)
    return y[start:], fcs, fallbacks


def _backtest_one(args: tuple[str, str, np.ndarray, ForecastConfig]) -> list[dict[str, Any]]:
    region, product, y, cfg = args
    rows: list[dict[str, Any]] = []
    for name in cfg.models:
        actual, fc, fallbacks = rolling_origin(y, name, cfg)
        for step, (a, f) in enumerate(zip(actual, fc, strict=True)):
            rows.append(
                {
                    "region_id": region,
                    "product_id": product,
                    "model": name,
                    "step": step,
                    "actual": a,
                    "forecast": f,
                    "series_fallbacks": fallbacks,
                }
            )
    return rows


def backtest_all(demand: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Backtest every configured model on every (region, product) series.

    Output: one row per (series, model, backtest step) with actual and forecast.
    ``month`` is attached so errors can be inspected over time.
    """
    jobs = [(r, p, y, cfg) for r, p, y in iter_series(demand)]
    rows = [row for chunk in parallel_map(_backtest_one, jobs, cfg.n_jobs) for row in chunk]
    errors = pd.DataFrame(rows)
    months = np.sort(demand["month"].unique())[-cfg.backtest_months :]
    errors["month"] = months[errors["step"].to_numpy()]
    return errors.drop(columns="step")


# --------------------------------------------------------------------- selection
def series_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    """WAPE, MAPE, bias, RMSE and fallback count per (series, model)."""

    def agg(g: pd.DataFrame) -> pd.Series:
        a, f = g["actual"].to_numpy(), g["forecast"].to_numpy()
        return pd.Series(
            {
                "wape": wape(a, f),
                "mape": mape(a, f),
                "bias": bias(a, f),
                "rmse": rmse(a, f),
                "fallbacks": int(g["series_fallbacks"].iloc[0]),
                # Sums kept so pooled (portfolio) metrics can be rebuilt exactly.
                "abs_error_sum": float(np.sum(np.abs(a - f))),
                "actual_sum": float(np.sum(a)),
            }
        )

    return (
        errors.groupby([*SERIES_KEY, "model"], sort=True)[
            ["actual", "forecast", "series_fallbacks"]
        ]
        .apply(agg)
        .reset_index()
    )


def select_models(metrics: pd.DataFrame, cfg: ForecastConfig) -> pd.DataFrame:
    """Choose a model for every series; ties go to the simpler model.

    * ``per_series``: lowest ``selection_metric`` for each (region, product).
    * ``global``: the single model with the lowest *pooled* error across all
      series (sum|A-F| / sum A for WAPE; mean MAPE otherwise) is used everywhere.
      Fewer choices means less room to fit noise, which matters when each
      series has only a few backtest points.

    Simplicity order follows ``MODELS`` (seasonal naive < ETS < SARIMA): if a
    complex model cannot beat a simple one, we should not pay for its complexity.
    """
    key = cfg.selection_metric
    rank = {name: i for i, name in enumerate(MODELS)}
    m = metrics.assign(_complexity=metrics["model"].map(rank), _score=metrics[key].fillna(np.inf))

    if cfg.selection_strategy == "global":
        g = m.groupby("model")
        pooled = (
            g["abs_error_sum"].sum() / g["actual_sum"].sum() if key == "wape" else g[key].mean()
        )
        ranked = (
            pooled.to_frame("score").assign(c=pooled.index.map(rank)).sort_values(["score", "c"])
        )
        m = m[m["model"] == ranked.index[0]]

    best = (
        m.sort_values([*SERIES_KEY, "_score", "_complexity"])
        .groupby(SERIES_KEY, sort=True)
        .head(1)
        .rename(columns={"model": "best_model"})
    )
    naive = metrics.loc[metrics["model"] == "seasonal_naive", [*SERIES_KEY, "wape"]].rename(
        columns={"wape": "naive_wape"}
    )
    out = best.merge(naive, on=SERIES_KEY, how="left")
    return out[[*SERIES_KEY, "best_model", "wape", "mape", "bias", "rmse", "naive_wape"]]


def portfolio_summary(errors: pd.DataFrame, selection: pd.DataFrame) -> dict[str, Any]:
    """Headline accuracy numbers across all series (volume-weighted).

    Portfolio WAPE = sum of all absolute errors / sum of all actuals, so large
    series count more. "selected" uses each series' chosen model. Because the
    same backtest window is used to *choose* and to *score*, the selected
    figure is slightly optimistic; the single-model rows have no selection step
    and are unbiased comparisons.
    """
    per_model = {
        str(name): wape(g["actual"].to_numpy(), g["forecast"].to_numpy())
        for name, g in errors.groupby("model")
    }
    chosen = errors.merge(selection[[*SERIES_KEY, "best_model"]], on=SERIES_KEY).query(
        "model == best_model"
    )
    a, f = chosen["actual"].to_numpy(), chosen["forecast"].to_numpy()
    selected = wape(a, f)
    naive = per_model["seasonal_naive"]
    others = {k: v for k, v in per_model.items() if k != "seasonal_naive"}
    best_name = min(others, key=lambda k: others[k]) if others else "seasonal_naive"
    best_single = per_model[best_name]
    return {
        "portfolio_wape_by_model": {k: round(v, 4) for k, v in per_model.items()},
        "portfolio_wape_selected": round(selected, 4),
        "portfolio_bias_selected": round(bias(a, f), 4),
        "wape_improvement_vs_naive_pct": round(improvement_pct(naive, selected), 2),
        "best_single_model": best_name,
        "best_single_improvement_vs_naive_pct": round(improvement_pct(naive, best_single), 2),
        "models_selected": selection["best_model"].value_counts().to_dict(),
        "series_with_fallbacks": int(
            errors.groupby([*SERIES_KEY, "model"])["series_fallbacks"].first().gt(0).sum()
        ),
    }


def holdout_selection_summary(errors: pd.DataFrame, cfg: ForecastConfig) -> dict[str, Any]:
    """Unbiased estimate of what model selection is worth, for *both* strategies.

    Choose models on the earlier backtest months only, then score those choices
    (and seasonal naive) on the final ``selection_holdout_months`` months, which
    the choice never saw. Reusing the backtest errors costs no extra model fits.
    ``holdout_improvement_vs_naive_pct`` refers to the configured strategy.
    """
    months = np.sort(errors["month"].unique())
    cutoff = months[-cfg.selection_holdout_months]
    early, late = errors[errors["month"] < cutoff], errors[errors["month"] >= cutoff]
    early_metrics = series_metrics(early)

    naive = late[late["model"] == "seasonal_naive"]
    naive_w = wape(naive["actual"].to_numpy(), naive["forecast"].to_numpy())
    out: dict[str, Any] = {
        "holdout_months": cfg.selection_holdout_months,
        "holdout_wape_naive": round(naive_w, 4),
    }
    for strategy in ("per_series", "global"):
        scfg = cfg.model_copy(update={"selection_strategy": strategy})
        choice = select_models(early_metrics, scfg)[[*SERIES_KEY, "best_model"]]
        chosen = late.merge(choice, on=SERIES_KEY).query("model == best_model")
        w = wape(chosen["actual"].to_numpy(), chosen["forecast"].to_numpy())
        out[f"holdout_wape_{strategy}"] = round(w, 4)
        out[f"holdout_improvement_{strategy}_pct"] = round(improvement_pct(naive_w, w), 2)
        if strategy == "global":
            out["holdout_global_model"] = str(choice["best_model"].iloc[0])
    out["holdout_improvement_vs_naive_pct"] = out[
        f"holdout_improvement_{cfg.selection_strategy}_pct"
    ]
    return out
