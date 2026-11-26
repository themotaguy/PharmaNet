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
    run_all,
    run_benchmark,
    run_dir,
    run_forecast,
    run_generate,
    run_graph,
    run_neo4j_export,
    run_optimize,
    run_simulate,
    run_stochastic,
)
from pharmanet.report import write_report

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
def simulate(config: ConfigOpt = Path("configs/base.yaml"), run_id: RunIdOpt = None) -> None:
    """Stress-test the chosen design: Monte Carlo demand, disruptions, sensitivity."""
    cfg = load_config(config)
    try:
        result, out_dir = run_simulate(cfg, run_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote simulation outputs to {out_dir}")
    s = dict(result.summary)
    worst = s["disruptions"].pop("worst")
    _echo_nested(s)
    typer.echo("  worst disruptions (by unmet demand):")
    for w in worst:
        typer.echo(
            f"    {w['scenario_id']:<22} unmet={w['unmet']:>10,.0f}  "
            f"service={w['service_level']:.2%}  cost/unit delivered "
            f"{w['cost_per_unit_change_pct']:+.2f}%"
        )
    typer.echo("  sensitivity:")
    cols = [
        "parameter",
        "multiplier",
        "redesign_open_dcs",
        "design_changed",
        "regret_pct",
        "base_design_service_level",
    ]
    typer.echo(result.sensitivity[cols].round(3).to_string(index=False))


@app.command()
def graph(config: ConfigOpt = Path("configs/base.yaml"), run_id: RunIdOpt = None) -> None:
    """Graph risk: exposure, single points of failure, cross-check, mitigations."""
    cfg = load_config(config)
    try:
        result, out_dir = run_graph(cfg, run_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote graph outputs to {out_dir}")
    s = result.summary
    typer.echo(f"  structural single points of failure: {s['structural_spofs']}")
    typer.echo(f"  cross-check: {s['cross_check']}")
    typer.echo("  node risk (units; structural <= simulated <= exposure):")
    cols = ["node", "kind", "structural_loss", "simulated_loss", "exposure", "betweenness"]
    typer.echo(result.cross_check[cols].round(3).to_string(index=False))
    typer.echo("  mitigations:")
    mcols = ["mitigation", "open_dcs", "cost_premium_pct", "worst_failure",
             "worst_failure_demand_at_risk_pct", "failures_with_shortfall",
             "worst_failure_reduction_pct"]  # fmt: skip
    typer.echo(result.mitigations[mcols].round(2).to_string(index=False))
    for a in result.dual_qualification_actions:
        typer.echo(f"    dual-qualify {a['product_id']}: {a['sole_plant']} + {a['backup_plant']}")


@app.command("neo4j-export")
def neo4j_export(
    config: ConfigOpt = Path("configs/base.yaml"),
    run_id: RunIdOpt = None,
    plant: Annotated[str | None, typer.Option(help="Plant for the failure-impact query.")] = None,
) -> None:
    """Write a Neo4j bundle: CSVs, import.cypher, queries.cypher, expected answers."""
    cfg = load_config(config)
    try:
        answers, out_dir = run_neo4j_export(cfg, run_id, plant)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote Neo4j bundle to {out_dir} (see README.md there)")
    for name, df in answers.items():
        typer.echo(f"\n  {name}: {len(df)} rows")
        typer.echo(df.head(5).round(3).to_string(index=False))


@app.command()
def stochastic(config: ConfigOpt = Path("configs/base.yaml"), run_id: RunIdOpt = None) -> None:
    """Two-stage stochastic design vs the deterministic one (VSS, EVPI, out-of-sample)."""
    cfg = load_config(config)
    try:
        summary, out_dir = run_stochastic(cfg, run_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote stochastic outputs to {out_dir}")
    _echo_nested(summary)


@app.command()
def benchmark(
    config: ConfigOpt = Path("configs/base.yaml"),
    n_seeds: Annotated[int | None, typer.Option("--n-seeds", min=1)] = None,
    run_id: RunIdOpt = None,
) -> None:
    """MILP vs greedy across many generated networks (seeds 1..N; default from config)."""
    cfg = load_config(config)
    n = n_seeds or cfg.report.benchmark_seeds or 10
    df, path = run_benchmark(cfg, n, run_id)
    typer.echo(df.round(3).to_string(index=False))
    s = df["savings_vs_greedy_pct"]
    typer.echo(
        f"\nSavings vs greedy over {len(df)} instances: mean {s.mean():.2f}%, "
        f"median {s.median():.2f}%, range {s.min():.2f}%-{s.max():.2f}%"
    )
    typer.echo(f"Wrote {path}")


PublishOpt = Annotated[
    bool,
    typer.Option("--publish/--no-publish", help="Also copy results.md to docs/results.md."),
]
DOCS_RESULTS = Path("docs/results.md")


@app.command("run-all")
def run_all_cmd(
    config: ConfigOpt = Path("configs/base.yaml"),
    run_id: RunIdOpt = None,
    publish: PublishOpt = False,
) -> None:
    """Run every stage end to end, then write manifest.json and results.md."""
    cfg = load_config(config)
    out, manifest = run_all(cfg, run_id, str(config), DOCS_RESULTS if publish else None)
    for stage, secs in manifest["stage_seconds"].items():
        typer.echo(f"  {stage:<10} {secs:6.1f} s")
    typer.echo(
        f"Report: {out / 'results.md'}" + (f" (published to {DOCS_RESULTS})" if publish else "")
    )


@app.command()
def report(
    config: ConfigOpt = Path("configs/base.yaml"),
    run_id: RunIdOpt = None,
    publish: PublishOpt = False,
) -> None:
    """Regenerate results.md from an existing run's outputs (no re-computation)."""
    cfg = load_config(config)
    try:
        path = write_report(run_dir(cfg, run_id), DOCS_RESULTS if publish else None)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {path}" + (f" and {DOCS_RESULTS}" if publish else ""))


@app.command()
def view(
    output_dir: Annotated[Path, typer.Option(help="Folder containing run directories.")] = Path(
        "outputs"
    ),
) -> None:
    """Open the Streamlit results viewer (needs the `viewer` extra)."""
    import importlib.util
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        typer.echo("Streamlit is not installed. Run: uv sync --extra viewer", err=True)
        raise typer.Exit(code=1)
    app_path = Path(__file__).parent / "viewer" / "app.py"
    env = {**os.environ, "PHARMANET_OUTPUT_DIR": str(output_dir.resolve())}
    raise typer.Exit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)], env=env)
    )


if __name__ == "__main__":
    app()
