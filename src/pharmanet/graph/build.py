"""Graphs of the supply network (NetworkX).

Two views, because they answer different questions:

* **Flow graph**: plants -> DCs -> regions, edges only where the solution
  actually ships product, weighted by flow. Answers "what passes through this
  node *today*?" Assumes nothing can be re-routed, so it overstates loss.
* **Capability graph** (one per product): edges wherever the product *could*
  flow: qualified plants -> open DCs (cold-capable if the product is cold) ->
  every region, plus a super-source feeding all qualified plants. Answers
  "is there *any* alternative path?" It ignores capacity, so it understates loss.

Node IDs are the instance IDs (unique across kinds: PL.., DC.., RG..), with a
``kind`` attribute of "plant", "dc" or "region".
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.solution import Solution

SOURCE = "__SOURCE__"  # super-source in capability graphs


def flow_graph(inst: NetworkInstance, sol: Solution) -> nx.DiGraph:
    """Directed graph of the flows in ``sol``; edge attrs: flow, cold_flow, per-product flow."""
    g = nx.DiGraph()
    for p in inst.plants:
        g.add_node(p, kind="plant")
    for d in sorted(sol.open_dcs):
        g.add_node(d, kind="dc", capacity=inst.cap_d[d], cold_capacity=inst.cold_d[d])
    for r in inst.regions:
        g.add_node(r, kind="region", demand=sum(inst.dem[(r, k)] for k in inst.products))

    edge_flow: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (p, d, k), v in sol.x.items():
        edge_flow[(p, d)][k] += v
    for (d, r, k), v in sol.z.items():
        edge_flow[(d, r)][k] += v
    for (a, b), by_k in edge_flow.items():
        g.add_edge(
            a,
            b,
            flow=sum(by_k.values()),
            cold_flow=sum(v for k, v in by_k.items() if k in inst.cold_products),
            products=",".join(sorted(by_k)),
        )
    return g


def capability_graph(inst: NetworkInstance, open_dcs: frozenset[str], product: str) -> nx.DiGraph:
    """Every path ``product`` could take through the open network (capacity ignored).

    Blocked lanes and zero-capacity nodes (disruptions) are excluded, so the
    same function describes a disrupted network too.
    """
    cold = product in inst.cold_products
    plants = [
        p for p in inst.plants if inst.qualified[(p, product)] and inst.cap_pk[(p, product)] > 0
    ]
    dcs = [d for d in sorted(open_dcs) if inst.cap_d[d] > 0 and (not cold or inst.cold_d[d] > 0)]
    g = nx.DiGraph()
    g.add_node(SOURCE, kind="source")
    for r in inst.regions:
        g.add_node(r, kind="region")
    for p in plants:
        g.add_edge(SOURCE, p)
        g.nodes[p]["kind"] = "plant"
        for d in dcs:
            if (p, d, product) not in inst.blocked_in:
                g.add_edge(p, d)
                g.nodes[d]["kind"] = "dc"
    for d in dcs:
        if d not in g:
            continue
        for r in inst.regions:
            if (d, r, product) not in inst.blocked_out:
                g.add_edge(d, r)
    return g
