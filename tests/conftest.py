"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture
def base_config_path() -> Path:
    return CONFIG_DIR / "base.yaml"


@pytest.fixture
def small_config_path() -> Path:
    return CONFIG_DIR / "small_test.yaml"


@pytest.fixture
def base_config_dict(base_config_path: Path) -> dict[str, Any]:
    """Raw YAML as a dict, so tests can mutate it to build invalid configs."""
    with base_config_path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


@pytest.fixture
def small_config_dict(small_config_path: Path) -> dict[str, Any]:
    with small_config_path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data
