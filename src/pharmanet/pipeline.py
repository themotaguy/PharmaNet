"""End-to-end orchestration. Each stage is a function the CLI (or a Databricks
job) can call; stages communicate only through files in the run directory.

Run directory layout::

    outputs/<run_id>/
        config.yaml      # exact config used, for provenance
        data/*.parquet   # Phase 2: synthetic inputs
        forecast/*.parquet  # Phase 3: backtest, selection, forecasts, safety stock
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pharmanet.config import PharmaNetConfig
from pharmanet.data.generate import Dataset, generate
from pharmanet.data.io import load_dataset, save_dataset
from pharmanet.forecasting.backtest import (
    backtest_all,
    holdout_selection_summary,
    portfolio_summary,
    select_models,
    series_metrics,
)
from pharmanet.forecasting.forecast import final_forecasts
from pharmanet.inventory.safety_stock import compute_safety_stock, fastest_path_lead_times


def run_dir(cfg: PharmaNetConfig, run_id: str | None = None) -> Path:
    """Output directory for a run; defaults to the config's run name."""
    return cfg.run.output_dir / (run_id or cfg.run.name)


def write_config_snapshot(cfg: PharmaNetConfig, directory: Path) -> Path:
    """Save the exact config next to the outputs, so every result is traceable."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return path


def run_generate(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[Dataset, Path]:
    out = run_dir(cfg, run_id)
    write_config_snapshot(cfg, out)
    ds = generate(cfg)
    data_dir = out / "data"
    save_dataset(ds, data_dir)
    return ds, data_dir


@dataclass(frozen=True)
class ForecastOutputs:
    errors: pd.DataFrame
    metrics: pd.DataFrame
    selection: pd.DataFrame
    forecasts: pd.DataFrame
    safety_stock: pd.DataFrame
    summary: dict[str, Any]


def forecast_and_safety_stock(cfg: PharmaNetConfig, ds: Dataset) -> ForecastOutputs:
    """Phase 3 as a pure function of (config, data): no file I/O, easy to test."""
    fc = cfg.forecasting
    errors = backtest_all(ds.demand, fc)
    metrics = series_metrics(errors)
    selection = select_models(metrics, fc)
    forecasts = final_forecasts(ds.demand, selection, fc)
    ss = compute_safety_stock(
        forecasts, selection, fastest_path_lead_times(ds), cfg.inventory.service_level
    )
    summary = portfolio_summary(errors, selection)
    summary["selection_strategy"] = fc.selection_strategy
    summary.update(holdout_selection_summary(errors, fc))
    summary["service_level"] = cfg.inventory.service_level
    summary["total_annual_forecast"] = round(float(ss["annual_forecast"].sum()))
    summary["total_safety_stock"] = round(float(ss["safety_stock"].sum()))
    summary["safety_stock_pct_of_annual"] = round(
        100 * float(ss["safety_stock"].sum() / ss["annual_forecast"].sum()), 2
    )
    return ForecastOutputs(errors, metrics, selection, forecasts, ss, summary)


def run_forecast(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[ForecastOutputs, Path]:
    """Load this run's data, run Phase 3, and write outputs to ``<run>/forecast/``."""
    out = run_dir(cfg, run_id)
    data_dir = out / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found; run `pharmanet generate` first")
    result = forecast_and_safety_stock(cfg, load_dataset(data_dir))
    fdir = out / "forecast"
    fdir.mkdir(parents=True, exist_ok=True)
    result.errors.to_parquet(fdir / "backtest_errors.parquet", index=False)
    result.metrics.to_parquet(fdir / "metrics.parquet", index=False)
    result.selection.to_parquet(fdir / "selection.parquet", index=False)
    result.forecasts.to_parquet(fdir / "forecast.parquet", index=False)
    result.safety_stock.to_parquet(fdir / "safety_stock.parquet", index=False)
    (fdir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    return result, fdir
