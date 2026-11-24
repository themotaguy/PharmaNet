"""Smoke tests for the CLI wiring."""

from __future__ import annotations

from pathlib import Path

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
