"""The network design problem instance: sets and parameters, solver-independent.

``NetworkInstance`` is the single input to the MILP, the greedy baseline, and
the solution validator. Keeping it as plain immutable Python data (lists and
dicts keyed by IDs) instead of DataFrames has two benefits:

* the formulation in ``docs/formulation.md`` maps onto it symbol-for-symbol
  (``dem[r, k]``, ``cap_pk[p, k]``, ...), and
* scenario analysis (Phase 5) can derive modified instances (demand scaled,
  a DC or plant removed) with ``dataclasses.replace`` without touching the model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from pharmanet.data.generate import Dataset

PDK = tuple[str, str, str]
DRK = tuple[str, str, str]
RK = tuple[str, str]
PK = tuple[str, str]


@dataclass(frozen=True)
class NetworkInstance:
    # Sets
    plants: tuple[str, ...]
    dcs: tuple[str, ...]
    regions: tuple[str, ...]
    products: tuple[str, ...]
    cold_products: frozenset[str]
    # Parameters (symbols as in docs/formulation.md)
    fixed_cost: dict[str, float]  # f_d
    c_in: dict[PDK, float]  # c^in_{pdk}, cold premium already applied for k in K^c
    c_out: dict[DRK, float]  # c^out_{drk}
    dem: dict[RK, float]  # dem_{rk} = annual forecast + safety stock
    cap_pk: dict[PK, float]  # plant capacity
    qualified: dict[PK, int]  # q_{pk} in {0, 1}
    cap_d: dict[str, float]  # DC throughput capacity
    cold_d: dict[str, float]  # DC cold capacity
    penalty: dict[str, float]  # pen_k

    def total_demand(self) -> float:
        return sum(self.dem.values())

    def size(self) -> dict[str, int]:
        P, D, R, K = len(self.plants), len(self.dcs), len(self.regions), len(self.products)
        return {"plants": P, "dcs": D, "regions": R, "products": K}

    def with_demand(self, dem: dict[RK, float]) -> NetworkInstance:
        return replace(self, dem=dict(dem))

    def scale_demand(self, factor: float) -> NetworkInstance:
        return self.with_demand({key: v * factor for key, v in self.dem.items()})


def build_instance(ds: Dataset, design_demand: pd.DataFrame) -> NetworkInstance:
    """Assemble the instance from Phase 2 data and Phase 3 design demand.

    ``design_demand`` needs columns region_id, product_id, design_demand.
    Missing (region, product) pairs are an error, not a silent zero.
    """
    plants = tuple(ds.plants["plant_id"])
    dcs = tuple(ds.dcs["dc_id"])
    regions = tuple(ds.regions["region_id"])
    products = tuple(ds.products["product_id"])
    cold = frozenset(ds.products.loc[ds.products["is_cold"], "product_id"])

    def lane_rates(lanes: pd.DataFrame, a: str, b: str) -> tuple[dict[Any, float], ...]:
        idx = lanes.set_index([a, b])
        return (idx["cost_per_unit"].to_dict(), idx["cold_cost_per_unit"].to_dict())

    in_amb, in_cold = lane_rates(ds.lanes_inbound, "plant_id", "dc_id")
    out_amb, out_cold = lane_rates(ds.lanes_outbound, "dc_id", "region_id")
    dem_s: dict[Any, float] = design_demand.set_index(["region_id", "product_id"])[
        "design_demand"
    ].to_dict()
    dem: dict[RK, float] = {}
    for r in regions:
        for k in products:
            if (r, k) not in dem_s:
                raise ValueError(f"design_demand missing for ({r}, {k})")
            dem[(r, k)] = float(dem_s[(r, k)])

    pp = ds.plant_products.set_index(["plant_id", "product_id"])
    pp_cap: dict[Any, float] = pp["capacity"].to_dict()
    pp_q: dict[Any, bool] = pp["qualified"].to_dict()
    dct = ds.dcs.set_index("dc_id")
    fixed: dict[Any, float] = dct["fixed_cost"].to_dict()
    cap_d: dict[Any, float] = dct["throughput_capacity"].to_dict()
    cold_d: dict[Any, float] = dct["cold_capacity"].to_dict()
    return NetworkInstance(
        plants=plants,
        dcs=dcs,
        regions=regions,
        products=products,
        cold_products=cold,
        fixed_cost={d: float(fixed[d]) for d in dcs},
        c_in={
            (p, d, k): float((in_cold if k in cold else in_amb)[(p, d)])
            for p in plants
            for d in dcs
            for k in products
        },
        c_out={
            (d, r, k): float((out_cold if k in cold else out_amb)[(d, r)])
            for d in dcs
            for r in regions
            for k in products
        },
        dem=dem,
        cap_pk={(p, k): float(pp_cap[(p, k)]) for p in plants for k in products},
        qualified={(p, k): int(pp_q[(p, k)]) for p in plants for k in products},
        cap_d={d: float(cap_d[d]) for d in dcs},
        cold_d={d: float(cold_d[d]) for d in dcs},
        penalty=dict(
            zip(ds.products["product_id"], ds.products["unmet_penalty"].astype(float), strict=True)
        ),
    )
