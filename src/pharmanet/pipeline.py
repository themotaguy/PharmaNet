"""End-to-end orchestration. Each stage is a function the CLI (or a Databricks
job) can call; stages communicate only through files in the run directory.

Run directory layout::

    outputs/<run_id>/
        config.yaml      # exact config used, for provenance
        data/*.parquet   # Phase 2: synthetic inputs
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pharmanet.config import PharmaNetConfig
from pharmanet.data.generate import Dataset, generate
from pharmanet.data.io import save_dataset


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
