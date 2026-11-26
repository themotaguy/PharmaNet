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
from pharmanet.pipeline import (
    benchmark_across_seeds,
    run_dir,
    run_forecast,
    run_generate,
    run_optimize,
)

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


@app.command()
def forecast(config: ConfigOpt = Path("configs/base.yaml"), run_id: RunIdOpt = None) -> None:
    """Backtest models, select per series, forecast, and compute safety stock."""
    cfg = load_config(config)
    try:
        result, out_dir = run_forecast(cfg, run_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote forecast outputs to {out_dir}")
    for key, value in result.summary.items():
        typer.echo(f"  {key:<38} {value}")


def _echo_nested(d: dict[str, object], indent: int = 2) -> None:
    for key, value in d.items():
        if isinstance(value, dict):
            typer.echo(f"{' ' * indent}{key}:")
            _echo_nested(value, indent + 2)
        elif isinstance(value, float):
            typer.echo(f"{' ' * indent}{key:<40} {value:,.4f}")
        else:
            typer.echo(f"{' ' * indent}{key:<40} {value}")


@app.command()
def optimize(config: ConfigOpt = Path("configs/base.yaml"), run_id: RunIdOpt = None) -> None:
    """Solve the network design MILP, refine safety stock, and benchmark vs greedy."""
    cfg = load_config(config)
    try:
        result, out_dir = run_optimize(cfg, run_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote optimization outputs to {out_dir}")
    summary = dict(result.summary)
    refinement = summary.pop("safety_stock_refinement")
    _echo_nested(summary)
    typer.echo("  safety_stock_refinement:")
    for row in refinement:
        typer.echo(f"    {row}")


@app.command()
def benchmark(
    config: ConfigOpt = Path("configs/base.yaml"),
    n_seeds: Annotated[int, typer.Option("--n-seeds", min=1)] = 10,
    run_id: RunIdOpt = None,
) -> None:
    """MILP vs greedy across many generated networks (seeds 1..N)."""
    cfg = load_config(config)
    df = benchmark_across_seeds(cfg, list(range(1, n_seeds + 1)))
    out = run_dir(cfg, run_id) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "milp_vs_greedy.parquet", index=False)
    typer.echo(df.round(3).to_string(index=False))
    s = df["savings_vs_greedy_pct"]
    typer.echo(
        f"\nSavings vs greedy over {len(df)} instances: mean {s.mean():.2f}%, "
        f"median {s.median():.2f}%, range {s.min():.2f}%-{s.max():.2f}%"
    )
    typer.echo(f"Wrote {out / 'milp_vs_greedy.parquet'}")


if __name__ == "__main__":
    app()
