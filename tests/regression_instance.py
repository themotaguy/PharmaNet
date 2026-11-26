"""Hand-verifiable regression instance (2 plants, 3 DCs, 3 regions, 2 products).

The optimum and every alternative design are derived by hand in
docs/formulation.md ("Regression instance"). Shared by optimization and
simulation tests.
"""

from __future__ import annotations

from pharmanet.optimization.instance import NetworkInstance

P, D, R, K = ("P1", "P2"), ("D1", "D2", "D3"), ("R1", "R2", "R3"), ("A", "C")
IN_COST = {("P1", "D1"): 1, ("P1", "D2"): 2, ("P1", "D3"): 3,
           ("P2", "D1"): 3, ("P2", "D2"): 2, ("P2", "D3"): 1}  # fmt: skip
OUT_COST = {("D1", "R1"): 1, ("D1", "R2"): 3, ("D1", "R3"): 5,
            ("D2", "R1"): 3, ("D2", "R2"): 1, ("D2", "R3"): 3,
            ("D3", "R1"): 5, ("D3", "R2"): 3, ("D3", "R3"): 1}  # fmt: skip
COLD_PREMIUM = 2.0


def regression_instance() -> NetworkInstance:
    """A = ambient, C = cold. Only P1 may make C; D3 has no cold storage."""
    prem = {"A": 1.0, "C": COLD_PREMIUM}
    return NetworkInstance(
        plants=P,
        dcs=D,
        regions=R,
        products=K,
        cold_products=frozenset({"C"}),
        fixed_cost={"D1": 500.0, "D2": 350.0, "D3": 400.0},
        c_in={(p, d, k): IN_COST[p, d] * prem[k] for p in P for d in D for k in K},
        c_out={(d, r, k): OUT_COST[d, r] * prem[k] for d in D for r in R for k in K},
        dem={(r, k): 100.0 if k == "A" else 50.0 for r in R for k in K},
        cap_pk={("P1", "A"): 1000.0, ("P1", "C"): 1000.0, ("P2", "A"): 1000.0, ("P2", "C"): 0.0},
        qualified={("P1", "A"): 1, ("P1", "C"): 1, ("P2", "A"): 1, ("P2", "C"): 0},
        cap_d={d: 1000.0 for d in D},
        cold_d={"D1": 1000.0, "D2": 1000.0, "D3": 0.0},
        penalty={"A": 100.0, "C": 100.0},
    )
