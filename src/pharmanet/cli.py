"""Command-line interface.

The CLI is deliberately thin: it parses arguments, loads config, and calls
functions in the library. No business logic lives here, so everything the CLI
does is also testable (and callable from Databricks jobs) without the CLI.
Commands for each pipeline stage are added in the phase that implements them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from pharmanet import __version__
from pharmanet.config import load_config
from pharmanet.data.generate import summarize
from pharmanet.pipeline import run_generate

app = typer.Typer(
    name="pharmanet",
    help="Pharma supply network optimization and resilience (synthetic data).",
    no_args_is_help=True,
)

ConfigOpt = Annotated[
    Path,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="Path to YAML config."),
]


@app.callback()
def main() -> None:
    """PharmaNet CLI."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("check-config")
def check_config(config: ConfigOpt = Path("configs/base.yaml")) -> None:
    """Validate a config file and print a short summary."""
    try:
        cfg = load_config(config)
    except ValidationError as exc:
        typer.echo(f"Invalid config {config}:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc
    n = cfg.network
    typer.echo(
        f"OK: {config} (run={cfg.run.name}, seed={cfg.run.seed}) | "
        f"{n.n_plants} plants, {n.n_dcs} DCs, {n.n_regions} regions, "
        f"{n.n_products} products, {n.n_months} months | solver={cfg.optimization.solver.name}"
    )


RunIdOpt = Annotated[
    str | None, typer.Option("--run-id", help="Output folder name (default: run.name).")
]


@app.command()
def generate(config: ConfigOpt = Path("configs/base.yaml"), run_id: RunIdOpt = None) -> None:
    """Generate the synthetic dataset and write it to outputs/<run_id>/data/."""
    cfg = load_config(config)
    ds, data_dir = run_generate(cfg, run_id)
    typer.echo(f"Wrote {len(ds.tables())} tables to {data_dir}")
    for key, value in summarize(ds).items():
        typer.echo(f"  {key:<32} {value}")


if __name__ == "__main__":
    app()
