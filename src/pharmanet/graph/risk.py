"""Graph-based risk metrics and their cross-check against simulation.

For a failure of node v (plant or open DC), three estimates of lost demand:

1. **Exposure** (flow graph, *upper* bound): demand whose current flow passes
   through v, found by **proportional flow tracing**: a plant's share of a
   DC's inflow of product k is assumed to make up the same share of that DC's
   outflow of k to every region. Ignores re-routing.
2. **Structural loss** (capability graph, *lower* bound): demand for (r, k)
   where v **dominates** r, i.e. *every* supply path to r for k passes
   through v (computed with the Cooper-Harvey-Kennedy dominator algorithm from a
   super-source). Ignores capacity; that demand is lost no matter what.
3. **Simulated loss** (Phase 5): the LP re-routes within capacity.

Expected: structural <= simulated <= exposure, node by node. The lower bound
is exact (no path means no flow). The upper bound holds in practice but is
not a theorem: keeping every unaffected flow stays feasible after v fails, so
a solution with loss = exposure always exists. The cost-minimizing LP picks
it or one with *less* unmet demand whenever the unmet penalty outweighs the
extra re-routing cost (true by a wide margin with our penalty, though
min-cost is not strictly min-unmet). Checking both bounds node by node is a
consistency test across three independently built modules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import pandas as pd

from pharmanet.graph.build import SOURCE, capability_graph
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.solution import Solution

BOUND_TOL = 1e-3  # units; LP solutions are floating point


def exposure_detail(inst: NetworkInstance, sol: Solution) -> pd.DataFrame:
    """Per failed node, the (region, product) demand currently flowing through it."""
    inflow: dict[tuple[str, str], float] = defaultdict(float)
    for (_, d, k), v in sol.x.items():
        inflow[(d, k)] += v
    rows: list[dict[str, Any]] = []
    for (d, r, k), v in sol.z.items():
        rows.append({"node": d, "kind": "dc", "region_id": r, "product_id": k, "exposed": v})
    for (p, d, k), v in sol.x.items():
        share = v / inflow[(d, k)]
        for r in inst.regions:
            z = sol.z.get((d, r, k), 0.0)
            if z > 0:
                rows.append(
                    {"node": p, "kind": "plant", "region_id": r, "product_id": k,
                     "exposed": share * z}
                )  # fmt: skip
    df = pd.DataFrame(rows, columns=["node", "kind", "region_id", "product_id", "exposed"])
    return df.groupby(["node", "kind", "region_id", "product_id"])["exposed"].sum().reset_index()


def structural_spofs(inst: NetworkInstance, open_dcs: frozenset[str]) -> pd.DataFrame:
    """(node, region, product) triples where the node is on every supply path.

    Rows with node == "UNREACHABLE" are demands with no path at all even before
    any failure (e.g. no qualified plant with capacity left).
    """
    rows: list[dict[str, Any]] = []
    for k in inst.products:
        g = capability_graph(inst, open_dcs, k)
        idom = nx.immediate_dominators(g, SOURCE)
        for r in inst.regions:
            demand = inst.dem[(r, k)]
            if r not in idom:
                rows.append({"node": "UNREACHABLE", "kind": "none", "region_id": r,
                             "product_id": k, "demand": demand})  # fmt: skip
                continue
            v = idom[r]
            while v != SOURCE:
                rows.append({"node": v, "kind": g.nodes[v]["kind"], "region_id": r,
                             "product_id": k, "demand": demand})  # fmt: skip
                v = idom[v]
    return pd.DataFrame(rows, columns=["node", "kind", "region_id", "product_id", "demand"])


def centrality(g: nx.DiGraph) -> pd.DataFrame:
    """Betweenness and flow-through per node of the flow graph.

    In a 3-layer plant -> DC -> region graph every supply path has length 2,
    so betweenness is non-zero only for DCs and essentially counts the
    (plant, region) pairs a DC connects. It ignores volume; ``flow_through``
    (units handled) is usually the more meaningful criticality measure here.
    """
    btw = nx.betweenness_centrality(g, normalized=True)
    rows = []
    for v, data in g.nodes(data=True):
        out_flow = sum(e["flow"] for _, _, e in g.out_edges(v, data=True))
        in_flow = sum(e["flow"] for _, _, e in g.in_edges(v, data=True))
        rows.append(
            {
                "node": v,
                "kind": data["kind"],
                "betweenness": btw[v],
                "flow_through": out_flow if data["kind"] == "plant" else in_flow,
                "out_degree": g.out_degree(v),
                "in_degree": g.in_degree(v),
            }
        )
    return pd.DataFrame(rows)


def node_risk(
    inst: NetworkInstance, sol: Solution, exposure: pd.DataFrame, spofs: pd.DataFrame
) -> pd.DataFrame:
    """One row per plant and open DC: exposure (upper) and structural loss (lower)."""
    total = inst.total_demand()
    nodes = [(p, "plant") for p in inst.plants] + [(d, "dc") for d in sorted(sol.open_dcs)]
    exp = exposure.groupby("node")["exposed"].sum()
    cold_exp = (
        exposure[exposure["product_id"].isin(inst.cold_products)].groupby("node")["exposed"].sum()
    )
    stranded = spofs.groupby("node")["demand"].sum()
    stranded_products = spofs.groupby("node")["product_id"].agg(lambda s: ",".join(sorted(set(s))))
    regions = exposure[exposure["exposed"] > 1e-6].groupby("node")["region_id"].nunique()
    rows = []
    for v, kind in nodes:
        rows.append(
            {
                "node": v,
                "kind": kind,
                "exposure": float(exp.get(v, 0.0)),
                "exposure_share": float(exp.get(v, 0.0)) / total,
                "cold_exposure": float(cold_exp.get(v, 0.0)),
                "regions_exposed": int(regions.get(v, 0)),
                "structural_loss": float(stranded.get(v, 0.0)),
                "structural_share": float(stranded.get(v, 0.0)) / total,
                "structural_spof_products": stranded_products.get(v, ""),
            }
        )
    return pd.DataFrame(rows)


def cross_check(risk: pd.DataFrame, disruptions: pd.DataFrame) -> pd.DataFrame:
    """Join graph bounds with simulated losses and test structural <= simulated <= exposure.

    Only full-removal node failures (plant and DC outages) are comparable; a
    partial plant outage is not a node removal, so callers should pass only
    full outages.
    """
    sim = disruptions[disruptions["kind"].isin(["plant_outage", "dc_outage"])][
        ["target", "unmet", "service_level"]
    ].rename(columns={"target": "node", "unmet": "simulated_loss"})
    df = risk.merge(sim, on="node", how="left")
    df["lower_bound_ok"] = df["structural_loss"] <= df["simulated_loss"] + BOUND_TOL
    df["upper_bound_ok"] = df["simulated_loss"] <= df["exposure"] + BOUND_TOL
    # Where the true loss sits between the bounds: 0 = at structural, 1 = at exposure.
    width = df["exposure"] - df["structural_loss"]
    df["position_in_bounds"] = (df["simulated_loss"] - df["structural_loss"]).where(
        width > BOUND_TOL
    ) / width.where(width > BOUND_TOL)
    return df.sort_values("simulated_loss", ascending=False).reset_index(drop=True)
