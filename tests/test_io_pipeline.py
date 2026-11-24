"""Parquet round-trip and run-directory provenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pharmanet.config import load_config
from pharmanet.data.generate import DataValidationError, generate
from pharmanet.data.io import load_dataset, save_dataset
from pharmanet.pipeline import run_generate


def test_parquet_round_trip(tmp_path: Path, small_config_path: Path) -> None:
    ds = generate(load_config(small_config_path))
    save_dataset(ds, tmp_path)
    loaded = load_dataset(tmp_path)
    for name, table in ds.tables().items():
        pd.testing.assert_frame_equal(table, loaded.tables()[name], obj=name)


def test_load_revalidates(tmp_path: Path, small_config_path: Path) -> None:
    ds = generate(load_config(small_config_path))
    save_dataset(ds, tmp_path)
    demand = ds.demand.copy()
    demand.loc[0, "demand"] = -1
    demand.to_parquet(tmp_path / "demand.parquet", index=False)
    with pytest.raises(DataValidationError):
        load_dataset(tmp_path)


def test_run_generate_writes_config_snapshot(
    tmp_path: Path, small_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config(small_config_path)
    _, data_dir = run_generate(cfg, run_id="t1")
    assert data_dir == Path("outputs/t1/data")
    assert (data_dir / "demand.parquet").exists()
    # The snapshot must reload to an identical config: results are traceable.
    assert load_config(Path("outputs/t1/config.yaml")) == cfg


def test_forecast_stage_end_to_end(small_config_path: Path) -> None:
    from pharmanet.pipeline import forecast_and_safety_stock

    cfg = load_config(small_config_path)
    out = forecast_and_safety_stock(cfg, generate(cfg))
    n_series = 3 * 2
    assert len(out.selection) == n_series
    assert len(out.forecasts) == n_series * cfg.forecasting.horizon
    assert len(out.errors) == n_series * len(cfg.forecasting.models) * 12
    assert (out.safety_stock["safety_stock"] >= 0).all()
    assert (out.safety_stock["design_demand"] >= out.safety_stock["annual_forecast"]).all()
    assert "holdout_improvement_vs_naive_pct" in out.summary


def test_run_forecast_requires_data(
    tmp_path: Path, small_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pharmanet.pipeline import run_forecast

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="pharmanet generate"):
        run_forecast(load_config(small_config_path), run_id="missing")
