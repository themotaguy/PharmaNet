"""Tests for config loading and validation.

The goal is to prove that bad configs fail loudly at load time rather than
producing silently wrong downstream results.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pharmanet.config import PharmaNetConfig, Range, load_config


@pytest.mark.parametrize("name", ["base_config_path", "small_config_path"])
def test_shipped_configs_load(name: str, request: pytest.FixtureRequest) -> None:
    cfg = load_config(request.getfixturevalue(name))
    assert isinstance(cfg, PharmaNetConfig)


def test_base_matches_spec(base_config_path: Path) -> None:
    """Default size from CLAUDE.md: 4 plants, 8 DCs, 20 regions, 12 products, 36 months."""
    n = load_config(base_config_path).network
    assert (n.n_plants, n.n_dcs, n.n_regions, n.n_products, n.n_months) == (4, 8, 20, 12, 36)


def test_small_matches_regression_size(small_config_path: Path) -> None:
    n = load_config(small_config_path).network
    assert (n.n_plants, n.n_dcs, n.n_regions, n.n_products) == (2, 3, 3, 2)


def test_config_is_frozen(base_config_path: Path) -> None:
    cfg = load_config(base_config_path)
    with pytest.raises(ValidationError):
        cfg.run.seed = 1  # type: ignore[misc]


def test_range_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError, match="must be <= high"):
        Range(low=2.0, high=1.0)


def _mutate(d: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(d)
    node = out
    *parents, leaf = path.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value
    return out


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("network.n_plants", 0),
        ("generation.products.cold_fraction", 1.5),
        ("inventory.service_level", 1.0),
        ("inventory.service_level", 0.3),
        ("costs.unmet_penalty_multiplier", 0.5),
        ("costs.cold_chain_premium", 0.9),
        ("optimization.solver.name", "cplex"),
        ("optimization.max_open_dcs", 99),
        ("network.n_months", 24),
        ("forecasting.models", ["ets", "sarima"]),
    ],
)
def test_invalid_values_rejected(base_config_dict: dict[str, Any], path: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        PharmaNetConfig.model_validate(_mutate(base_config_dict, path, value))


def test_unknown_key_rejected(base_config_dict: dict[str, Any]) -> None:
    """Typos in YAML keys must not be silently ignored."""
    bad = _mutate(base_config_dict, "network.n_plant", 4)
    with pytest.raises(ValidationError, match="n_plant"):
        PharmaNetConfig.model_validate(bad)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_rejected(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)
