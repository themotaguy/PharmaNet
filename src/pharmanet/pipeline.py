"""End-to-end orchestration. Each stage is a function the CLI (or a Databricks
job) can call; stages communicate only through files in the run directory.

Run directory layout::

    outputs/<run_id>/
        config.yaml      # exact config used, for provenance
        data/*.parquet   # Phase 2: synthetic inputs
        forecast/*.parquet  # Phase 3: backtest, selection, forecasts, safety stock
        optimize/*.parquet  # Phase 4: MILP + baseline designs and flows, summary.json
        simulate/*.parquet  # Phase 5: Monte Carlo, disruptions, sensitivity, summary.json
        graph/              # Phase 6: node risk, SPOFs, cross-check, mitigations, GraphML
        stochastic/         # Phase 8: two-stage design, VSS/EVPI, out-of-sample comparison
        neo4j/              # Phase 8: CSVs, import.cypher, queries.cypher, expected answers
        benchmark/          # MILP vs greedy across generated networks
        manifest.json       # Phase 7: provenance (config hash, git commit, timings)
        results.md          # Phase 7: generated report
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import yaml

from pharmanet import __version__
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
from pharmanet.graph.build import flow_graph
from pharmanet.graph.neo4j_export import export_tables, write_neo4j_bundle
from pharmanet.graph.risk import (
    centrality,
    cross_check,
    exposure_detail,
    node_risk,
    structural_spofs,
)
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
from pharmanet.optimization.stochastic import (
    evaluate_design_on,
    expected,
    solve_stochastic,
    wait_and_see,
)
from pharmanet.optimization.validate import validate_solution
from pharmanet.report import write_report
from pharmanet.simulation.evaluate import (
    evaluate_design,
    evaluate_scenarios,
    run_sensitivity,
    summarize_disruptions,
    summarize_monte_carlo,
)
from pharmanet.simulation.mitigation import run_mitigations
from pharmanet.simulation.scenarios import (
    STOCHASTIC_IN_SAMPLE_STREAM,
    STOCHASTIC_OUT_OF_SAMPLE_STREAM,
    demand_scenarios,
    disruption_scenarios,
    simulation_rng,
    stochastic_scenarios,
)


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


# ------------------------------------------------------------------ Phase 5: simulate
@dataclass(frozen=True)
class SimulateOutputs:
    monte_carlo: pd.DataFrame
    disruptions: pd.DataFrame
    disruption_unmet: pd.DataFrame
    sensitivity: pd.DataFrame
    summary: dict[str, Any]


def simulate_design(
    cfg: PharmaNetConfig,
    ds: Dataset,
    design_demand: pd.DataFrame,
    open_dcs: frozenset[str],
) -> SimulateOutputs:
    """Phase 5 as a pure function: stress-test a fixed design.

    ``design_demand`` needs region_id, product_id, annual_forecast, sigma_monthly,
    design_demand. Scenarios use *forecast* demand (or draws around it): safety
    stock is the buffer the design reserved capacity for, not demand itself.
    Sensitivity re-designs on design demand, matching how the design was made.
    """
    sim, opt = cfg.simulation, cfg.optimization
    at_forecast = build_instance(
        ds, design_demand.assign(design_demand=design_demand["annual_forecast"])
    )
    base_sol = evaluate_design(at_forecast, open_dcs, opt)
    base_costs = cost_breakdown(at_forecast, base_sol)
    base_cost = base_costs["total"]
    delivered = at_forecast.total_demand() - sum(base_sol.u.values())
    base_cost_per_unit = (base_cost - base_costs["penalty"]) / delivered

    mc_scen = demand_scenarios(at_forecast, design_demand, sim, simulation_rng(cfg.run.seed))
    mc, _ = evaluate_scenarios(mc_scen, open_dcs, opt, sim.n_jobs)

    dis_scen = disruption_scenarios(at_forecast, base_sol, sim)
    dis, dis_unmet = evaluate_scenarios(dis_scen, open_dcs, opt, sim.n_jobs)

    sens = run_sensitivity(build_instance(ds, design_demand), open_dcs, opt, sim)

    ranked = summarize_disruptions(dis, base_cost_per_unit)
    summary: dict[str, Any] = {
        "design_open_dcs": sorted(open_dcs),
        "base_at_forecast": {
            "total_cost": base_cost,
            "service_level": service_level(at_forecast, base_sol),
            "open_capacity_headroom_pct": 100
            * (sum(at_forecast.cap_d[d] for d in open_dcs) / at_forecast.total_demand() - 1),
        },
        "monte_carlo": summarize_monte_carlo(mc, base_cost),
        "disruptions": {
            "n_scenarios": len(dis),
            "worst": ranked[:5],
            "n_causing_shortfall": int((dis["unmet"] > 1e-6).sum()),
        },
        "sensitivity": {
            "n_runs": len(sens),
            "design_changed_in": sens.loc[
                sens["design_changed"], ["parameter", "multiplier"]
            ].to_dict(orient="records"),
            "max_regret_pct": float(sens["regret_pct"].max()),
            "min_base_design_service_level": float(sens["base_design_service_level"].min()),
        },
    }
    return SimulateOutputs(mc, dis, dis_unmet, sens, summary)


def run_simulate(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[SimulateOutputs, Path]:
    """Load data, forecast, and design outputs; run Phase 5; write ``<run>/simulate/``."""
    out = run_dir(cfg, run_id)
    odir = out / "optimize"
    if not (odir / "milp_open_dcs.parquet").exists():
        raise FileNotFoundError(f"{odir} has no design; run `pharmanet optimize` first")
    sigma = pd.read_parquet(out / "forecast" / "safety_stock.parquet")[
        ["region_id", "product_id", "sigma_monthly"]
    ]
    dd = pd.read_parquet(odir / "design_demand.parquet").merge(
        sigma, on=["region_id", "product_id"], validate="one_to_one"
    )
    open_dcs = frozenset(pd.read_parquet(odir / "milp_open_dcs.parquet")["dc_id"])
    result = simulate_design(cfg, load_dataset(out / "data"), dd, open_dcs)

    sdir = out / "simulate"
    sdir.mkdir(parents=True, exist_ok=True)
    result.monte_carlo.to_parquet(sdir / "monte_carlo.parquet", index=False)
    result.disruptions.to_parquet(sdir / "disruptions.parquet", index=False)
    result.disruption_unmet.to_parquet(sdir / "disruption_unmet.parquet", index=False)
    result.sensitivity.to_parquet(sdir / "sensitivity.parquet", index=False)
    (sdir / "summary.json").write_text(json.dumps(result.summary, indent=2, default=str))
    return result, sdir


# --------------------------------------------------------------------- Phase 6: graph
@dataclass(frozen=True)
class GraphOutputs:
    flow_graph: nx.DiGraph
    node_risk: pd.DataFrame
    exposure: pd.DataFrame
    spofs: pd.DataFrame
    centrality: pd.DataFrame
    cross_check: pd.DataFrame
    mitigations: pd.DataFrame
    dual_qualification_actions: list[dict[str, Any]]
    summary: dict[str, Any]


def analyze_graph(
    cfg: PharmaNetConfig,
    ds: Dataset,
    design_demand: pd.DataFrame,
    open_dcs: frozenset[str],
    disruptions: pd.DataFrame,
) -> GraphOutputs:
    """Phase 6 as a pure function: graph risk, cross-check vs simulation, mitigations.

    The flow graph is built from the design's optimal flows at *forecast*
    demand, the same basis as the Phase 5 disruption scenarios, so the graph
    bounds and simulated losses are directly comparable.
    """
    opt = cfg.optimization
    at_forecast = build_instance(
        ds, design_demand.assign(design_demand=design_demand["annual_forecast"])
    )
    base_sol = evaluate_design(at_forecast, open_dcs, opt)
    g = flow_graph(at_forecast, base_sol)
    exposure = exposure_detail(at_forecast, base_sol)
    spofs = structural_spofs(at_forecast, open_dcs)
    risk = node_risk(at_forecast, base_sol, exposure, spofs)
    cent = centrality(g)
    risk = risk.merge(cent[["node", "betweenness", "flow_through"]], on="node", how="left")

    comparable = cfg.simulation.outage_capacity_loss == 1.0
    check = cross_check(risk, disruptions if comparable else disruptions.iloc[0:0])

    mitig, actions = run_mitigations(
        build_instance(ds, design_demand), at_forecast, open_dcs, opt, cfg.simulation,
        cfg.graph.backup_capacity_share,
    )  # fmt: skip

    def best(df: pd.DataFrame) -> dict[str, Any]:
        row = df.sort_values(["worst_failure_unmet", "design_cost"]).iloc[0]
        return {str(k): v for k, v in row.to_dict().items()}

    total = at_forecast.total_demand()
    top = check.iloc[0]
    summary: dict[str, Any] = {
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()},
        "worst_single_node_failure": {
            "node": top["node"],
            "simulated_demand_at_risk_pct": 100 * float(top["simulated_loss"]) / total,
            "exposure_pct": 100 * float(top["exposure"]) / total,
            "structural_pct": 100 * float(top["structural_loss"]) / total,
        },
        "structural_spofs": {
            str(n): sorted(set(grp["product_id"]))
            for n, grp in spofs[spofs["node"] != "UNREACHABLE"].groupby("node")
        },
        "unreachable_demand_before_failures": float(
            spofs.loc[spofs["node"] == "UNREACHABLE", "demand"].sum()
        ),
        "cross_check": {
            "compared_nodes": int(check["simulated_loss"].notna().sum()),
            "lower_bound_holds": bool(check["lower_bound_ok"].all()),
            "upper_bound_holds": bool(check["upper_bound_ok"].all()),
            "comparable": comparable,
        },
        "highest_betweenness": cent.sort_values("betweenness", ascending=False)
        .head(3)[["node", "betweenness"]]
        .to_dict(orient="records"),
        "mitigations": mitig.to_dict(orient="records"),
        "best_mitigation": best(mitig),
    }
    if not comparable or math.isnan(float(top["simulated_loss"])):
        summary["worst_single_node_failure"]["note"] = "simulated losses not comparable"
    return GraphOutputs(g, risk, exposure, spofs, cent, check, mitig, actions, summary)


def run_graph(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[GraphOutputs, Path]:
    """Load design + simulation outputs, run Phase 6, write ``<run>/graph/``."""
    out = run_dir(cfg, run_id)
    sim = out / "simulate" / "disruptions.parquet"
    if not sim.exists():
        raise FileNotFoundError(f"{sim} not found; run `pharmanet simulate` first")
    odir = out / "optimize"
    dd = pd.read_parquet(odir / "design_demand.parquet")
    open_dcs = frozenset(pd.read_parquet(odir / "milp_open_dcs.parquet")["dc_id"])
    result = analyze_graph(cfg, load_dataset(out / "data"), dd, open_dcs, pd.read_parquet(sim))

    gdir = out / "graph"
    gdir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(result.flow_graph, gdir / "flow_graph.graphml")
    result.node_risk.to_parquet(gdir / "node_risk.parquet", index=False)
    result.exposure.to_parquet(gdir / "exposure_detail.parquet", index=False)
    result.spofs.to_parquet(gdir / "structural_spofs.parquet", index=False)
    result.cross_check.to_parquet(gdir / "cross_check.parquet", index=False)
    result.mitigations.to_parquet(gdir / "mitigations.parquet", index=False)
    (gdir / "summary.json").write_text(json.dumps(result.summary, indent=2, default=str))
    return result, gdir


# -------------------------------------------------------------- Phase 7: end to end
def run_benchmark(
    cfg: PharmaNetConfig, n_seeds: int, run_id: str | None = None
) -> tuple[pd.DataFrame, Path]:
    df = benchmark_across_seeds(cfg, list(range(1, n_seeds + 1)))
    out = run_dir(cfg, run_id) / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "milp_vs_greedy.parquet"
    df.to_parquet(path, index=False)
    return df, path


def _git_commit() -> str | None:
    """Short commit hash (+ "-dirty" if the tree has local changes), or None outside git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def config_fingerprint(cfg: PharmaNetConfig) -> str:
    """sha256 of the validated config: same hash => same inputs."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_all(
    cfg: PharmaNetConfig,
    run_id: str | None = None,
    config_file: str | None = None,
    publish_to: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Every stage in order, then the manifest and ``results.md``.

    Stages communicate only through files in the run directory, so any stage
    can also be re-run on its own with the same result.
    """
    started = datetime.now(UTC)
    timings: dict[str, float] = {}
    stages: list[tuple[str, Any]] = [
        ("generate", lambda: run_generate(cfg, run_id)),
        ("forecast", lambda: run_forecast(cfg, run_id)),
        ("optimize", lambda: run_optimize(cfg, run_id)),
        ("simulate", lambda: run_simulate(cfg, run_id)),
        ("graph", lambda: run_graph(cfg, run_id)),
    ]
    stages.append(("neo4j", lambda: run_neo4j_export(cfg, run_id)))
    if cfg.stochastic.enabled:
        stages.append(("stochastic", lambda: run_stochastic(cfg, run_id)))
    if cfg.report.benchmark_seeds > 0:
        stages.append(("benchmark", lambda: run_benchmark(cfg, cfg.report.benchmark_seeds, run_id)))
    for name, fn in stages:
        t0 = time.perf_counter()
        fn()
        timings[name] = time.perf_counter() - t0

    out = run_dir(cfg, run_id)
    manifest = {
        "run_id": out.name,
        "config_file": config_file,
        "config_sha256": config_fingerprint(cfg),
        "seed": cfg.run.seed,
        "package_version": __version__,
        "git_commit": _git_commit(),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage_seconds": timings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report(out, publish_to)
    return out, manifest


# --------------------------------------------------------- Phase 8: stochastic design
def _design_stats(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    demand_rows = df[df["scenario"].str.contains("_demand_")]
    outage_rows = df[~df["scenario"].str.contains("_demand_")]
    out = {
        f"{prefix}expected_cost": expected(df, "total_cost"),
        f"{prefix}p95_cost_demand_scenarios": float(demand_rows["total_cost"].quantile(0.95)),
        f"{prefix}mean_service_demand_scenarios": float(demand_rows["service_level"].mean()),
        f"{prefix}prob_shortfall_demand_scenarios": float((demand_rows["unmet"] > 1e-6).mean()),
    }
    if len(outage_rows):
        out[f"{prefix}worst_outage_service"] = float(outage_rows["service_level"].min())
        out[f"{prefix}mean_outage_service"] = float(outage_rows["service_level"].mean())
    return out


def stochastic_design(
    cfg: PharmaNetConfig,
    ds: Dataset,
    design_demand: pd.DataFrame,
    deterministic_dcs: frozenset[str],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Two-stage design vs the deterministic one: RP, EEV, VSS, WS, EVPI, out-of-sample.

    ``design_demand`` needs annual_forecast and sigma_monthly. Scenario demand is
    drawn around the forecast with *no* safety stock: the stochastic model
    hedges explicitly, while the deterministic design hedges via safety stock.
    """
    st, sim, opt = cfg.stochastic, cfg.simulation, cfg.optimization
    base = build_instance(ds, design_demand.assign(design_demand=design_demand["annual_forecast"]))

    def scenarios(n: int, stream: int, prefix: str) -> list[Any]:
        return stochastic_scenarios(
            base, design_demand, n, st.disruption_weight, sim.demand_common_shock_share,
            sim.outage_capacity_loss, simulation_rng(cfg.run.seed, stream), prefix,
        )  # fmt: skip

    ins = scenarios(st.n_scenarios, STOCHASTIC_IN_SAMPLE_STREAM, "in")
    sp = solve_stochastic(ins, opt)
    eev = evaluate_design_on(ins, deterministic_dcs, opt, sim.n_jobs)
    rp, eev_v = sp.objective, expected(eev, "total_cost")
    if eev_v < rp - 1e-4 * abs(rp):
        raise AssertionError(
            f"EEV ({eev_v:.2f}) < RP ({rp:.2f}): the SP was not solved to optimality"
        )

    summary: dict[str, Any] = {
        "stochastic_open_dcs": sorted(sp.open_dcs),
        "deterministic_open_dcs": sorted(deterministic_dcs),
        "same_design": sp.open_dcs == deterministic_dcs,
        "in_sample": {
            "n_scenarios": len(ins),
            "RP": rp,
            "EEV": eev_v,
            "VSS": eev_v - rp,
            "VSS_pct_of_EEV": 100 * (eev_v - rp) / eev_v,
        },
        "model": sp.stats,
    }
    tables = {"sp_in_sample": sp.scenario_costs, "deterministic_in_sample": eev}

    if st.compute_wait_and_see:
        ws = wait_and_see(ins, opt, sim.n_jobs)
        ws_v = expected(ws, "ws_cost")
        if ws_v > rp + 1e-4 * abs(rp):
            raise AssertionError(f"WS ({ws_v:.2f}) > RP ({rp:.2f}): inconsistent solves")
        summary["in_sample"].update({"WS": ws_v, "EVPI": rp - ws_v,
                                     "EVPI_pct_of_RP": 100 * (rp - ws_v) / rp})  # fmt: skip
        tables["wait_and_see"] = ws

    oos = scenarios(st.n_eval_scenarios, STOCHASTIC_OUT_OF_SAMPLE_STREAM, "oos")
    oos_sp = evaluate_design_on(oos, sp.open_dcs, opt, sim.n_jobs)
    oos_det = evaluate_design_on(oos, deterministic_dcs, opt, sim.n_jobs)
    s_sp, s_det = _design_stats(oos_sp, ""), _design_stats(oos_det, "")
    summary["out_of_sample"] = {
        "n_scenarios": len(oos),
        "stochastic": s_sp,
        "deterministic": s_det,
        "VSS": s_det["expected_cost"] - s_sp["expected_cost"],
        "VSS_pct": 100 * (s_det["expected_cost"] - s_sp["expected_cost"]) / s_det["expected_cost"],
    }
    tables.update({"sp_out_of_sample": oos_sp, "deterministic_out_of_sample": oos_det})
    # Cost at design demand, for comparison with the deterministic objective.
    design_inst = build_instance(ds, design_demand)
    summary["cost_at_design_demand"] = {
        "stochastic": cost_breakdown(design_inst, evaluate_design(design_inst, sp.open_dcs, opt))[
            "total"
        ],
        "deterministic": cost_breakdown(
            design_inst, evaluate_design(design_inst, deterministic_dcs, opt)
        )["total"],
    }
    return summary, tables


def run_stochastic(cfg: PharmaNetConfig, run_id: str | None = None) -> tuple[dict[str, Any], Path]:
    out = run_dir(cfg, run_id)
    odir = out / "optimize"
    if not (odir / "milp_open_dcs.parquet").exists():
        raise FileNotFoundError(f"{odir} has no design; run `pharmanet optimize` first")
    sigma = pd.read_parquet(out / "forecast" / "safety_stock.parquet")[
        ["region_id", "product_id", "sigma_monthly"]
    ]
    dd = pd.read_parquet(odir / "design_demand.parquet").merge(
        sigma, on=["region_id", "product_id"], validate="one_to_one"
    )
    det = frozenset(pd.read_parquet(odir / "milp_open_dcs.parquet")["dc_id"])
    summary, tables = stochastic_design(cfg, load_dataset(out / "data"), dd, det)
    sdir = out / "stochastic"
    sdir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(sdir / f"{name}.parquet", index=False)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary, sdir


# --------------------------------------------------------------- Phase 8: Neo4j export
def run_neo4j_export(
    cfg: PharmaNetConfig, run_id: str | None = None, plant: str | None = None
) -> tuple[dict[str, pd.DataFrame], Path]:
    """Export the chosen design (flows at forecast demand) as a Neo4j bundle."""
    out = run_dir(cfg, run_id)
    odir = out / "optimize"
    if not (odir / "milp_open_dcs.parquet").exists():
        raise FileNotFoundError(f"{odir} has no design; run `pharmanet optimize` first")
    dd = pd.read_parquet(odir / "design_demand.parquet")
    open_dcs = frozenset(pd.read_parquet(odir / "milp_open_dcs.parquet")["dc_id"])
    ds = load_dataset(out / "data")
    inst = build_instance(ds, dd.assign(design_demand=dd["annual_forecast"]))
    sol = evaluate_design(inst, open_dcs, cfg.optimization)
    ndir = out / "neo4j"
    answers = write_neo4j_bundle(ndir, export_tables(ds, inst, sol), plant)
    return answers, ndir
