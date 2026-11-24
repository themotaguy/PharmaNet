"""Read/write a Dataset as one parquet file per table.

Parquet keeps column types (bools stay bools, timestamps stay timestamps),
which CSV does not, and it is the native format for Spark/Delta in Phase 8.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd

from pharmanet.data.generate import Dataset, validate_dataset


def save_dataset(ds: Dataset, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in ds.tables().items():
        path = directory / f"{name}.parquet"
        table.to_parquet(path, index=False)
        paths.append(path)
    return paths


def load_dataset(directory: Path) -> Dataset:
    """Load and re-validate, so downstream phases never consume a corrupted dataset."""
    tables = {f.name: pd.read_parquet(directory / f"{f.name}.parquet") for f in fields(Dataset)}
    ds = Dataset(**tables)
    validate_dataset(ds)
    return ds
