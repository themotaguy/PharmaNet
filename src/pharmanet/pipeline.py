"""End-to-end orchestration. Each stage is a function the CLI (or a Databricks
job) can call; stages communicate only through files in the run directory.

Run directory layout::

    outputs/<run_id>/
        config.yaml      # exact config used, for provenance
        data/*.parquet   # Phase 2: synthetic inputs
        forecast/*.parquet  # Phase 3: backtest, selection, forecasts, safety stock
        optimize/*.parquet  # Phase 4: MILP + baseline designs and flows, summary.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pharmanet.config import PharmaNetConfig
from pharmanet.data.generate import Dataset, annual_demand, generate
from pharmanet.data.io import load_dataset, save_dataset
from pharmanet.forecasting.backtest import (
    backtest_all,
    holdout_selection_summary,
    portfolio_summary,
    select_models,
    series_metrics,
)
from pharmanet.forecasting.forecast import final_forecasts
from pharmanet.inventory.safety_stock import (
    compute_safety_stock,
    fastest_path_lead_times,
    flow_weighted_lead_times,
)
from pharmanet.optimization.baseline import run_benchmarks
from pharmanet.optimization.instance import NetworkInstance, build_instance
from pharmanet.optimization.network_design import solve_network_design
from pharmanet.optimization.solution import (
    Solution,
    cost_breakdown,
    service_level,
    solution_frames,
)
from pharmanet.optimization.validate import validate_solution


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


@dataclass(frozen=True)
class ForecastOutputs:
    errors: pd.DataFrame
    metrics: pd.DataFrame
    selection: pd.DataFrame
    forecasts: pd.DataFrame
    safety_stock: pd.DataFrame
    summary: dict[str, Any]


def forecast_and_safety_stock(cfg: PharmaNetConfig, ds: Dataset) -> ForecastOutputs:
    """Phase 3 as a pure function of (config, data): no file I/O, easy to test."""
    fc = cfg.forecasting
    errors = backtest_all(ds.demand, fc)
    metrics = series_metrics(errors)
    selection = select_models(metrics, fc)
    forecasts = final_forecasts(ds.demand, selection, fc)
    ss = compute_safety_stock(
        forecasts, selection, fastest_path_lead_times(ds), cfg.inventory.service_level
    )
    summary = portfolio_summary(errors, selection)
    summary["selection_strategy"] = fc.selection_strategy
    summary.update(holdout_selection_summary(errors, fc))
    summary["service_level"] = cfg.inventory.service_level
    summary["total_annual_forecast"] = round(float(ss["annual_forecast"].sum()))
    summary["total_safety_stock"] = round(float(ss["safety_stock"].sum()))
    summary["safety_stock_pct_of_annual"] = round(
        100 * float(ss["safety_stock"].sum() / ss["annual_forecast"].sum()), 2
    )
    return ForecastOutputs(errors, metrics, selection, forecasts, ss, summary)


def run_forecast(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[ForecastOutputs, Path]:
    """Load this run's data, run Phase 3, and write outputs to ``<run>/forecast/``."""
    out = run_dir(cfg, run_id)
    data_dir = out / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found; run `pharmanet generate` first")
    result = forecast_and_safety_stock(cfg, load_dataset(data_dir))
    fdir = out / "forecast"
    fdir.mkdir(parents=True, exist_ok=True)
    result.errors.to_parquet(fdir / "backtest_errors.parquet", index=False)
    result.metrics.to_parquet(fdir / "metrics.parquet", index=False)
    result.selection.to_parquet(fdir / "selection.parquet", index=False)
    result.forecasts.to_parquet(fdir / "forecast.parquet", index=False)
    result.safety_stock.to_parquet(fdir / "safety_stock.parquet", index=False)
    (fdir / "summary.json").write_text(json.dumps(result.summary, indent=2))
    return result, fdir


# ----------------------------------------------------------------- Phase 4: optimize
@dataclass(frozen=True)
class OptimizeOutputs:
    instance: NetworkInstance
    milp: Solution
    greedy: Solution
    greedy_optimal_flows: Solution
    design_demand: pd.DataFrame
    summary: dict[str, Any]


def _solve_validated(inst: NetworkInstance, cfg: PharmaNetConfig) -> Solution:
    o = cfg.optimization
    sol = solve_network_design(inst, o)
    validate_solution(inst, sol, single_sourcing=o.single_sourcing, max_open_dcs=o.max_open_dcs)
    return sol


def optimize_network(
    cfg: PharmaNetConfig,
    ds: Dataset,
    forecasts: pd.DataFrame,
    selection: pd.DataFrame,
    safety_stock: pd.DataFrame,
) -> OptimizeOutputs:
    """Phase 4 as a pure function: solve, refine safety stock, re-solve, benchmark.

    Safety stock from Phase 3 assumed the fastest feasible path. After each
    solve we recompute it from the lead times of the network actually chosen
    and re-solve (``ss_refinement_iterations`` times), recording whether the
    design changed. Every solution is validated independently.
    """
    o = cfg.optimization
    ss = safety_stock
    inst = build_instance(ds, ss)
    sol = _solve_validated(inst, cfg)
    fallback = fastest_path_lead_times(ds)
    refinement: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "lead_time_basis": "fastest_path",
            "total_safety_stock": float(ss["safety_stock"].sum()),
            "open_dcs": sorted(sol.open_dcs),
            "total_cost": cost_breakdown(inst, sol)["total"],
        }
    ]
    for i in range(1, o.ss_refinement_iterations + 1):
        lt = flow_weighted_lead_times(ds, sol.x, sol.z, fallback)
        ss = compute_safety_stock(forecasts, selection, lt, cfg.inventory.service_level)
        inst = build_instance(ds, ss)
        new = _solve_validated(inst, cfg)
        refinement.append(
            {
                "iteration": i,
                "lead_time_basis": "flow_weighted_previous_design",
                "total_safety_stock": float(ss["safety_stock"].sum()),
                "open_dcs": sorted(new.open_dcs),
                "design_changed": new.open_dcs != sol.open_dcs,
                "total_cost": cost_breakdown(inst, new)["total"],
            }
        )
        sol = new

    greedy, greedy_of, bench = run_benchmarks(inst, o, sol)
    validate_solution(inst, greedy)
    validate_solution(inst, greedy_of)

    assert sol.stats is not None
    summary: dict[str, Any] = {
        "instance": inst.size(),
        "model": {
            k: sol.stats.as_dict()[k]
            for k in ("n_variables", "n_binary", "n_constraints", "termination")
        },
        "solve_time_s": round(sol.stats.wall_time_s, 3),
        "mip_gap": sol.stats.mip_gap,
        "lp_relaxation": {
            "lp_bound": sol.stats.lp_bound,
            "integrality_gap": sol.stats.integrality_gap,
            "lp_bound_without_strong_link": sol.stats.lp_bound_weak,
            "integrality_gap_without_strong_link": sol.stats.integrality_gap_weak,
        },
        "open_dcs": sorted(sol.open_dcs),
        "costs": cost_breakdown(inst, sol),
        "service_level": service_level(inst, sol),
        "benchmark": {
            **bench,
            "greedy_open_dcs": sorted(greedy.open_dcs),
            "greedy_service_level": service_level(inst, greedy),
        },
        "safety_stock_refinement": refinement,
    }
    design_demand = ss[
        ["region_id", "product_id", "annual_forecast", "safety_stock", "design_demand"]
    ]
    return OptimizeOutputs(inst, sol, greedy, greedy_of, design_demand, summary)


def run_optimize(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[OptimizeOutputs, Path]:
    """Load this run's data and forecast outputs, run Phase 4, write ``<run>/optimize/``."""
    out = run_dir(cfg, run_id)
    fdir = out / "forecast"
    if not (fdir / "safety_stock.parquet").exists():
        raise FileNotFoundError(f"{fdir} has no forecast outputs; run `pharmanet forecast` first")
    result = optimize_network(
        cfg,
        load_dataset(out / "data"),
        pd.read_parquet(fdir / "forecast.parquet"),
        pd.read_parquet(fdir / "selection.parquet"),
        pd.read_parquet(fdir / "safety_stock.parquet"),
    )
    odir = out / "optimize"
    odir.mkdir(parents=True, exist_ok=True)
    for prefix, sol in [
        ("milp", result.milp),
        ("greedy", result.greedy),
        ("greedy_optimal_flows", result.greedy_optimal_flows),
    ]:
        for name, frame in solution_frames(sol).items():
            frame.to_parquet(odir / f"{prefix}_{name}.parquet", index=False)
    result.design_demand.to_parquet(odir / "design_demand.parquet", index=False)
    (odir / "summary.json").write_text(json.dumps(result.summary, indent=2, default=str))
    return result, odir


# ------------------------------------------------- Phase 4: multi-instance benchmark
def benchmark_across_seeds(cfg: PharmaNetConfig, seeds: list[int]) -> pd.DataFrame:
    """MILP vs greedy on many independently generated networks.

    One instance can be lucky or unlucky for the heuristic, so the headline
    saving is reported as a distribution. For speed, design demand here is the
    last 12 months of history (no forecasting step); that is fine for comparing
    *methods* on the same inputs, but these are not the pipeline's planning numbers.
    """
    rows = []
    for seed in seeds:
        scfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"seed": seed})})
        ds = generate(scfg)
        dd = annual_demand(ds.demand, ["region_id", "product_id"]).rename("design_demand")
        inst = build_instance(ds, dd.reset_index())
        milp = _solve_validated(inst, scfg)
        greedy, greedy_of, bench = run_benchmarks(inst, scfg.optimization, milp)
        validate_solution(inst, greedy)
        validate_solution(inst, greedy_of)
        assert milp.stats is not None
        rows.append(
            {
                "seed": seed,
                "milp_open_dcs": len(milp.open_dcs),
                "greedy_open_dcs": len(greedy.open_dcs),
                "same_dcs": milp.open_dcs == greedy.open_dcs,
                "savings_vs_greedy_pct": bench["savings_vs_greedy_pct"],
                "savings_from_dc_choice_pct": bench["savings_from_dc_choice_pct"],
                "savings_from_routing_pct": bench["savings_from_routing_pct"],
                "greedy_service_level": service_level(inst, greedy),
                "milp_service_level": service_level(inst, milp),
                "solve_time_s": milp.stats.wall_time_s,
                "integrality_gap": milp.stats.integrality_gap,
            }
        )
    return pd.DataFrame(rows)
