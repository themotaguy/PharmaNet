"""Final forward forecast: refit each series' selected model on *all* history.

The backtest chooses the model; this step produces the numbers planning uses.
Refitting on the full history (rather than reusing a backtest fit) uses the
most recent data, which is what a planner would do on the day of the plan.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pharmanet.config import ForecastConfig, ForecastModelName
from pharmanet.forecasting.backtest import SERIES_KEY, iter_series, parallel_map
from pharmanet.forecasting.models import forecast


def _forecast_one(
    args: tuple[str, str, np.ndarray, ForecastModelName, ForecastConfig],
) -> dict[str, Any]:
    region, product, y, name, cfg = args
    res = forecast(name, y, cfg.horizon, cfg)
    return {
        "region_id": region,
        "product_id": product,
        "model": name,
        "fell_back": res.fell_back,
        "values": res.yhat,
    }


def final_forecasts(
    demand: pd.DataFrame, selection: pd.DataFrame, cfg: ForecastConfig
) -> pd.DataFrame:
    """One row per (region, product, future month) with the selected model's forecast."""
    best = selection.set_index(SERIES_KEY)["best_model"]
    jobs = [(r, p, y, best.loc[(r, p)], cfg) for r, p, y in iter_series(demand)]
    results = parallel_map(_forecast_one, jobs, cfg.n_jobs)

    last = pd.Timestamp(demand["month"].max())
    future = pd.date_range(last + pd.offsets.MonthBegin(1), periods=cfg.horizon, freq="MS")
    frames = [
        pd.DataFrame(
            {
                "region_id": r["region_id"],
                "product_id": r["product_id"],
                "month": future,
                "forecast": r["values"],
                "model": r["model"],
                "fell_back": r["fell_back"],
            }
        )
        for r in results
    ]
    return pd.concat(frames, ignore_index=True)
