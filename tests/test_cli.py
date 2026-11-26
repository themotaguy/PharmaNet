"""Smoke tests for the CLI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pharmanet import __version__
from pharmanet.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_check_config_ok(small_config_path: Path) -> None:
    result = runner.invoke(app, ["check-config", "-c", str(small_config_path)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_check_config_invalid(tmp_path: Path, small_config_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(small_config_path.read_text().replace("service_level: 0.95", "service_level: 2"))
    result = runner.invoke(app, ["check-config", "-c", str(bad)])
    assert result.exit_code == 1


def test_generate_command(
    tmp_path: Path, small_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["generate", "-c", str(small_config_path), "--run-id", "cli"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs/cli/data/products.parquet").exists()
    assert "annual_demand_units" in result.output


def test_forecast_command(
    tmp_path: Path, small_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["generate", "-c", str(small_config_path), "--run-id", "f"])
    result = runner.invoke(app, ["forecast", "-c", str(small_config_path), "--run-id", "f"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs/f/forecast/safety_stock.parquet").exists()


def test_optimize_command(
    tmp_path: Path, small_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for cmd in ("generate", "forecast", "optimize"):
        result = runner.invoke(app, [cmd, "-c", str(small_config_path), "--run-id", "o"])
        assert result.exit_code == 0, result.output
    assert (tmp_path / "outputs/o/optimize/summary.json").exists()
    assert "savings_vs_greedy_pct" in result.output


def test_optimize_requires_forecast(
    tmp_path: Path, small_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["generate", "-c", str(small_config_path), "--run-id", "o2"])
    result = runner.invoke(app, ["optimize", "-c", str(small_config_path), "--run-id", "o2"])
    assert result.exit_code == 1
