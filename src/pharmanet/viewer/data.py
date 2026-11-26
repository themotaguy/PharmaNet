"""Load a finished run's output files for the viewer.

The viewer never recomputes anything: it shows exactly what the pipeline wrote,
so it can never disagree with ``results.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pharmanet.report import load_results, resume_numbers


@dataclass(frozen=True)
class RunData:
    path: Path
    results: dict[str, Any]  # every stage summary (see report.load_results)
    resume: list[dict[str, str]]
    demand: pd.DataFrame
    forecast: pd.DataFrame
    selection: pd.DataFrame
    plants: pd.DataFrame
    dcs: pd.DataFrame
    regions: pd.DataFrame
    flows_inbound: pd.DataFrame
    flows_outbound: pd.DataFrame
    monte_carlo: pd.DataFrame
    disruptions: pd.DataFrame
    stochastic: dict[str, Any] | None = None
    stochastic_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def list_runs(output_dir: Path) -> list[Path]:
    """Run folders that contain at least a complete graph stage, newest first."""
    if not output_dir.exists():
        return []
    runs = [p for p in output_dir.iterdir() if (p / "graph" / "summary.json").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(path: Path) -> RunData:
    res = load_results(path)

    def pq(rel: str) -> pd.DataFrame:
        return pd.read_parquet(path / rel)

    stoch_dir = path / "stochastic"
    stochastic = None
    tables: dict[str, pd.DataFrame] = {}
    if (stoch_dir / "summary.json").exists():
        stochastic = json.loads((stoch_dir / "summary.json").read_text())
        tables = {p.stem: pd.read_parquet(p) for p in sorted(stoch_dir.glob("*.parquet"))}
    return RunData(
        path=path,
        results=res,
        resume=resume_numbers(res),
        demand=pq("data/demand.parquet"),
        forecast=pq("forecast/forecast.parquet"),
        selection=pq("forecast/selection.parquet"),
        plants=pq("data/plants.parquet"),
        dcs=pq("data/dcs.parquet"),
        regions=pq("data/regions.parquet"),
        flows_inbound=pq("optimize/milp_flows_inbound.parquet"),
        flows_outbound=pq("optimize/milp_flows_outbound.parquet"),
        monte_carlo=pq("simulate/monte_carlo.parquet"),
        disruptions=pq("simulate/disruptions.parquet"),
        stochastic=stochastic,
        stochastic_tables=tables,
    )


def network_frames(run: RunData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(nodes, edges) for the network map. Edges aggregate flow over products."""
    open_dcs = set(run.results["optimize"]["open_dcs"])
    nodes = pd.concat(
        [
            run.plants.rename(columns={"plant_id": "id"}).assign(kind="Plant"),
            run.dcs.rename(columns={"dc_id": "id"}).assign(
                kind=lambda f: f["id"].map(lambda d: "Open DC" if d in open_dcs else "Closed DC")
            ),
            run.regions.rename(columns={"region_id": "id"}).assign(kind="Region"),
        ],
        ignore_index=True,
    )[["id", "kind", "x", "y"]]
    xy = nodes.set_index("id")[["x", "y"]]
    legs = []
    for df, a, b, leg in [
        (run.flows_inbound, "plant_id", "dc_id", "plant → DC"),
        (run.flows_outbound, "dc_id", "region_id", "DC → region"),
    ]:
        e = df.groupby([a, b])["flow"].sum().reset_index().rename(columns={a: "src", b: "dst"})
        legs.append(e.assign(leg=leg))
    edges = pd.concat(legs, ignore_index=True)
    src_xy = xy
    dst_xy = xy.rename(columns={"x": "x2", "y": "y2"})
    edges = edges.join(src_xy, on="src").join(dst_xy, on="dst")
    return nodes, edges


def series_frame(run: RunData, region: str, product: str) -> pd.DataFrame:
    """History + 12-month forecast for one (region, product), long format."""
    h = run.demand[(run.demand["region_id"] == region) & (run.demand["product_id"] == product)]
    f = run.forecast[
        (run.forecast["region_id"] == region) & (run.forecast["product_id"] == product)
    ]
    return pd.concat(
        [
            pd.DataFrame({"month": h["month"], "units": h["demand"], "series": "Actual"}),
            pd.DataFrame({"month": f["month"], "units": f["forecast"], "series": "Forecast"}),
        ],
        ignore_index=True,
    )
