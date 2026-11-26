"""Build ``results.md`` from a run's output files.

The report is generated from the parquet/JSON outputs only, never from
hand-typed numbers, so it cannot drift from what the pipeline actually
produced. Every headline figure in the "Resume numbers" section carries its
source file and its caveat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

DISCLAIMER = (
    "> **All data is synthetic.** Every number below comes from a seeded random network "
    "built to resemble a pharma distribution network. None of it describes a real company, "
    "product, or network."
)


# ------------------------------------------------------------------------ loading
def load_results(run_path: Path) -> dict[str, Any]:
    """Collect every stage's summary from ``outputs/<run_id>/``.

    Raises FileNotFoundError naming the first missing stage, so a partial run
    fails loudly instead of producing a report with gaps.
    """
    out: dict[str, Any] = {"run_path": str(run_path), "run_id": run_path.name}
    for stage in ("forecast", "optimize", "simulate", "graph"):
        f = run_path / stage / "summary.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; run the `{stage}` stage (or `run-all`) first")
        out[stage] = json.loads(f.read_text())
    bench = run_path / "benchmark" / "milp_vs_greedy.parquet"
    out["benchmark"] = pd.read_parquet(bench) if bench.exists() else None
    manifest = run_path / "manifest.json"
    out["manifest"] = json.loads(manifest.read_text()) if manifest.exists() else None
    out["mitigations"] = pd.read_parquet(run_path / "graph" / "mitigations.parquet")
    out["cross_check"] = pd.read_parquet(run_path / "graph" / "cross_check.parquet")
    out["sensitivity"] = pd.read_parquet(run_path / "simulate" / "sensitivity.parquet")
    return out


# ----------------------------------------------------------------------- formatting
def _pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"


def _money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x / 1e3:,.0f}k" if abs(x) >= 1e3 else f"${x:,.0f}"


def _units(x: float) -> str:
    return f"{x / 1e3:,.0f}k" if abs(x) >= 1e4 else f"{x:,.0f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ resume numbers
def resume_numbers(res: dict[str, Any]) -> list[dict[str, str]]:
    """The quotable figures, each with value, source, and caveat."""
    f, o, s, g = res["forecast"], res["optimize"], res["simulate"], res["graph"]
    mc = s["monte_carlo"]
    nums: list[dict[str, str]] = []

    bench = res["benchmark"]
    if bench is not None and len(bench):
        sv = bench["savings_vs_greedy_pct"]
        nums.append(
            {
                "claim": "MILP cost savings vs greedy heuristic",
                "value": f"mean {_pct(sv.mean())} (range {_pct(sv.min())}–{_pct(sv.max())}) "
                f"across {len(bench)} generated networks",
                "source": "benchmark/milp_vs_greedy.parquet",
                "caveat": "Synthetic networks; design demand = last 12 months of history. On "
                f"this run's network: {_pct(o['benchmark']['savings_vs_greedy_pct'])}.",
            }
        )
    else:
        nums.append(
            {
                "claim": "MILP cost savings vs greedy heuristic",
                "value": _pct(o["benchmark"]["savings_vs_greedy_pct"]) + " (single network)",
                "source": "optimize/summary.json",
                "caveat": "One instance only; run the benchmark for a distribution.",
            }
        )

    strat = f["selection_strategy"]
    other = "global" if strat == "per_series" else "per_series"
    nums.append(
        {
            "claim": "Forecast WAPE improvement over seasonal naive",
            "value": f"{_pct(f['holdout_improvement_vs_naive_pct'])} ({strat} selection, "
            f"out-of-sample holdout); {other}: "
            f"{_pct(f[f'holdout_improvement_{other}_pct'])}",
            "source": "forecast/summary.json",
            "caveat": f"Last {f['holdout_months']} months held out from model choice. In-sample "
            f"figure ({_pct(f['wape_improvement_vs_naive_pct'])}) is optimistic; do not quote it.",
        }
    )
    lo, _, hi = mc["cost_p5_p50_p95"]
    nums.append(
        {
            "claim": "Service level and cost range under demand uncertainty",
            "value": f"mean service {_pct(100 * mc['service_level_mean'], 2)}, worst "
            f"{_pct(100 * mc['service_level_min'], 2)}; cost p5–p95 {_money(lo)}–{_money(hi)} "
            f"over {mc['n_scenarios']} Monte Carlo scenarios",
            "source": "simulate/summary.json",
            "caveat": "Annual spread = monthly RMSE × √12 (assumes independent monthly errors; "
            "likely optimistic).",
        }
    )
    mit = res["mitigations"].set_index("mitigation")
    base, best = mit.loc["base"], mit.loc[g["best_mitigation"]["mitigation"]]
    nums.append(
        {
            "claim": "Demand at risk from worst single-node failure, and mitigation",
            "value": f"{_pct(base['worst_failure_demand_at_risk_pct'])} ({base['worst_failure']} "
            f"down) → {_pct(best['worst_failure_demand_at_risk_pct'])} with "
            f"{g['best_mitigation']['mitigation']} "
            f"(−{_pct(best['worst_failure_reduction_pct'])}) for "
            f"{best['cost_premium_pct']:+.1f}% network cost",
            "source": "graph/mitigations.parquet",
            "caveat": "Full-year single-node outages at forecast demand. Dual-qualification "
            "(tech transfer) cost is not modelled.",
        }
    )
    m = o["model"]
    nums.append(
        {
            "claim": "MILP instance size and solve time",
            "value": f"{m['n_variables']:,} variables ({m['n_binary']} binary), "
            f"{m['n_constraints']:,} constraints; solved to proven optimality "
            f"(MIP gap {_pct(100 * o['mip_gap'], 2)}) in {o['solve_time_s']:.1f} s with HiGHS",
            "source": "optimize/summary.json",
            "caveat": f"Integrality gap {_pct(100 * o['lp_relaxation']['integrality_gap'])} "
            f"(vs {_pct(100 * o['lp_relaxation']['integrality_gap_without_strong_link'])} "
            "without strong-link constraints).",
        }
    )
    return nums


# ----------------------------------------------------------------------- markdown
def render_markdown(res: dict[str, Any]) -> str:
    f, o, s, g = res["forecast"], res["optimize"], res["simulate"], res["graph"]
    man = res["manifest"] or {}
    parts: list[str] = [f"# PharmaNet results: `{res['run_id']}`", "", DISCLAIMER, ""]

    if man:
        parts += [
            "## Run",
            "",
            _table(
                ["Field", "Value"],
                [
                    [
                        "Config",
                        f"`{man.get('config_file', '?')}` (sha256 "
                        f"`{str(man.get('config_sha256', ''))[:12]}`)",
                    ],
                    ["Seed", str(man.get("seed", "?"))],
                    [
                        "Code",
                        f"pharmanet {man.get('package_version', '?')}, git "
                        f"`{man.get('git_commit') or 'n/a'}`",
                    ],
                    ["Finished", str(man.get("finished_at", "?"))],
                    [
                        "Stage timings (s)",
                        ", ".join(f"{k} {v:.1f}" for k, v in man.get("stage_seconds", {}).items()),
                    ],
                ],
            ),
            "",
        ]

    inst = o["instance"]
    parts += [
        "## Instance",
        "",
        f"{inst['plants']} plants, {inst['dcs']} candidate DCs, {inst['regions']} regions, "
        f"{inst['products']} products.",
        "",
        "## 1. Forecasting",
        "",
        _table(
            ["Measure", "Value"],
            [
                [
                    "Portfolio WAPE by model (12-month backtest)",
                    ", ".join(f"{k} {v:.3f}" for k, v in f["portfolio_wape_by_model"].items()),
                ],
                ["Selection strategy (config)", f["selection_strategy"]],
                [
                    f"Holdout WAPE, last {f['holdout_months']} months: naive / per-series / global",
                    f"{f['holdout_wape_naive']:.3f} / {f['holdout_wape_per_series']:.3f} / "
                    f"{f['holdout_wape_global']:.3f} ({f['holdout_global_model']})",
                ],
                [
                    "Holdout improvement vs naive: per-series / global",
                    f"{_pct(f['holdout_improvement_per_series_pct'])} / "
                    f"{_pct(f['holdout_improvement_global_pct'])}",
                ],
                [
                    "In-sample improvement (optimistic, selection bias)",
                    _pct(f["wape_improvement_vs_naive_pct"]),
                ],
                ["Fit fallbacks to seasonal naive", str(f["series_with_fallbacks"])],
            ],
        ),
        "",
        "## 2. Safety stock",
        "",
        f"Service level {_pct(100 * f['service_level'], 0)}: safety stock "
        f"{_units(f['total_safety_stock'])} units = {_pct(f['safety_stock_pct_of_annual'], 2)} "
        f"of annual forecast ({_units(f['total_annual_forecast'])} units).",
    ]
    ref = o["safety_stock_refinement"]
    if len(ref) > 1:
        first, last = ref[0], ref[-1]
        parts.append(
            "Recomputed with the chosen network's lead times: "
            f"{_units(first['total_safety_stock'])} → {_units(last['total_safety_stock'])} "
            "units; design changed: "
            f"{'yes' if last.get('design_changed', False) else 'no'}."
        )

    c, lp, b = o["costs"], o["lp_relaxation"], o["benchmark"]
    m = o["model"]
    parts += [
        "",
        "## 3. Network design (MILP)",
        "",
        _table(
            ["Measure", "Value"],
            [
                [
                    "Size",
                    f"{m['n_variables']:,} vars ({m['n_binary']} binary), "
                    f"{m['n_constraints']:,} constraints",
                ],
                [
                    "Solve",
                    f"{m['termination']}, {o['solve_time_s']:.2f} s, MIP gap "
                    f"{_pct(100 * o['mip_gap'], 3)}",
                ],
                [
                    "LP bound / integrality gap",
                    f"{_money(lp['lp_bound'])} / "
                    f"{_pct(100 * lp['integrality_gap'], 2)} (without C6: "
                    f"{_pct(100 * lp['integrality_gap_without_strong_link'], 2)})",
                ],
                ["Open DCs", ", ".join(o["open_dcs"])],
                [
                    "Annual cost",
                    f"{_money(c['total'])} = fixed {_money(c['fixed'])} + inbound "
                    f"{_money(c['inbound'])} + outbound {_money(c['outbound'])} + penalty "
                    f"{_money(c['penalty'])}",
                ],
                ["Service level (design demand)", _pct(100 * o["service_level"], 2)],
            ],
        ),
        "",
        "## 4. MILP vs greedy baseline",
        "",
        f"This network: greedy {_money(b['greedy_total_cost'])} vs MILP "
        f"{_money(b['milp_total_cost'])} → **{_pct(b['savings_vs_greedy_pct'], 2)}** "
        f"(DC choice {_pct(b['savings_from_dc_choice_pct'], 2)}, routing "
        f"{_pct(b['savings_from_routing_pct'], 2)}); greedy service "
        f"{_pct(100 * b['greedy_service_level'], 2)}.",
    ]
    bench = res["benchmark"]
    if bench is not None and len(bench):
        sv = bench["savings_vs_greedy_pct"]
        parts += [
            "",
            f"Across {len(bench)} generated networks (seeds {int(bench['seed'].min())}–"
            f"{int(bench['seed'].max())}): mean **{_pct(sv.mean())}**, median "
            f"{_pct(sv.median())}, range {_pct(sv.min())}–{_pct(sv.max())}; mean saving from "
            f"DC choice {_pct(bench['savings_from_dc_choice_pct'].mean())}, from routing "
            f"{_pct(bench['savings_from_routing_pct'].mean())}.",
        ]

    mc, dis = s["monte_carlo"], s["disruptions"]
    p5, p50, p95 = mc["cost_p5_p50_p95"]
    parts += [
        "",
        "## 5. Simulation (design fixed, flows re-optimized)",
        "",
        _table(
            ["Measure", "Value"],
            [
                ["Monte Carlo scenarios", str(mc["n_scenarios"])],
                ["Cost p5 / p50 / p95", f"{_money(p5)} / {_money(p50)} / {_money(p95)}"],
                [
                    "Service level mean / p5 / worst",
                    f"{_pct(100 * mc['service_level_mean'], 2)}"
                    f" / {_pct(100 * mc['service_level_p5'], 2)} / "
                    f"{_pct(100 * mc['service_level_min'], 2)}",
                ],
                ["P(any shortfall)", _pct(100 * mc["prob_any_shortfall"], 1)],
                [
                    "Open-capacity headroom over forecast",
                    _pct(s["base_at_forecast"]["open_capacity_headroom_pct"]),
                ],
                [
                    "Single failures causing shortfall (plants, DCs, cold lanes)",
                    f"{dis['n_causing_shortfall']} of {dis['n_scenarios']}",
                ],
            ],
        ),
        "",
        "Worst single failures:",
        "",
        _table(
            ["Scenario", "Unmet units", "Service level", "Cost / delivered unit"],
            [
                [
                    w["scenario_id"],
                    _units(w["unmet"]),
                    _pct(100 * w["service_level"], 1),
                    f"{w['cost_per_unit_change_pct']:+.1f}%",
                ]
                for w in dis["worst"]
            ],
        ),
    ]
    sens = res["sensitivity"]
    changed = sens[sens["design_changed"]]
    parts += [
        "",
        "Sensitivity (re-designed per case): design changed in "
        + (
            ", ".join(f"{r.parameter} ×{r.multiplier}" for r in changed.itertuples())
            if len(changed)
            else "no case"
        )
        + "; lowest service of the kept base design "
        + _pct(100 * sens["base_design_service_level"].min(), 1)
        + ".",
    ]

    cc = res["cross_check"]
    cx = g["cross_check"]
    parts += [
        "",
        "## 6. Graph risk and mitigation",
        "",
        "Structural single points of failure: "
        + (", ".join(f"{n} → {', '.join(p)}" for n, p in g["structural_spofs"].items()) or "none")
        + ".",
        "",
        f"Cross-check structural ≤ simulated ≤ exposure on {cx['compared_nodes']} nodes: lower "
        f"bound {'holds' if cx['lower_bound_holds'] else '**FAILS**'}, upper bound "
        f"{'holds' if cx['upper_bound_holds'] else '**FAILS**'}.",
        "",
        _table(
            ["Node", "Structural", "Simulated", "Exposure", "Betweenness"],
            [
                [
                    r.node,
                    _units(r.structural_loss),
                    _units(r.simulated_loss),
                    _units(r.exposure),
                    f"{r.betweenness:.3f}",
                ]
                for r in cc.head(6).itertuples()
            ],
        ),
        "",
        _table(
            ["Mitigation", "Open DCs", "Cost premium", "Worst failure (at risk)", "Mean at risk"],
            [
                [
                    r.mitigation,
                    str(r.n_open),
                    f"{r.cost_premium_pct:+.1f}%",
                    f"{r.worst_failure} ({_pct(r.worst_failure_demand_at_risk_pct)})",
                    _pct(r.mean_demand_at_risk_pct),
                ]
                for r in res["mitigations"].itertuples()
            ],
        ),
        "",
        "Cost premium is network cost at design demand. Dual qualification's one-off cost "
        "(tech transfer, regulatory approval) is not modelled, so its premium is understated. "
        '"At risk" is unmet demand as a share of forecast demand, over every single plant '
        "and DC outage.",
        "",
        "## Resume numbers",
        "",
        "Each figure is computed by the pipeline; quote it with its caveat.",
        "",
    ]
    for i, n in enumerate(resume_numbers(res), 1):
        parts += [
            f"{i}. **{n['claim']}:** {n['value']}  ",
            f"   _Source:_ `{n['source']}`. _Caveat:_ {n['caveat']}",
        ]
    parts.append("")
    return "\n".join(parts)


def write_report(run_path: Path, publish_to: Path | None = None) -> Path:
    """Render ``results.md`` into the run directory (and optionally publish a copy)."""
    md = render_markdown(load_results(run_path))
    target = run_path / "results.md"
    target.write_text(md)
    if publish_to is not None:
        publish_to.parent.mkdir(parents=True, exist_ok=True)
        publish_to.write_text(md)
    return target
