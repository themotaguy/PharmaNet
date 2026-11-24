"""Synthetic data generator for a pharma distribution network.

All data produced here is synthetic. The goal is not realism in every detail
but realistic *structure*: the features that make the downstream problems
non-trivial (cold chain, site qualification, seasonality, intermittent demand,
capacity that is tight enough to force trade-offs).

Design choices worth knowing:

* **Independent random streams.** The master seed is split with
  ``numpy.random.SeedSequence.spawn`` into one stream per component (products,
  locations, demand, ...). Each component's draws are then isolated from the
  others: changing how DCs are generated does not reshuffle demand history.
* **Capacities are sized from demand.** Plant and DC capacities are drawn as
  multiples ("slack") of generated demand, so every instance is feasible in
  aggregate while still forcing the optimizer to choose among DCs.
* **Feasibility guarantees are explicit.** Every product gets at least one
  qualified plant, and at least one DC has cold storage when cold products
  exist. ``validate_dataset`` re-checks this independently.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from pharmanet.config import PharmaNetConfig, Range

MONTHS_PER_YEAR = 12  # physical constant, not a tunable parameter

_STREAMS = ("products", "locations", "demand", "plants", "dcs")


class DataValidationError(ValueError):
    """Raised when a generated or loaded dataset breaks a structural invariant."""


@dataclass(frozen=True)
class Dataset:
    """All tables describing one synthetic network instance.

    Every table is a "long" DataFrame keyed by string IDs, which is the shape
    that maps cleanly to parquet files, Delta tables, and Pyomo index sets.
    """

    products: pd.DataFrame
    plants: pd.DataFrame
    plant_products: pd.DataFrame
    dcs: pd.DataFrame
    regions: pd.DataFrame
    lanes_inbound: pd.DataFrame
    lanes_outbound: pd.DataFrame
    demand: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- helpers
def _ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(1, n + 1)]


def _uniform(rng: np.random.Generator, r: Range, size: int | tuple[int, ...]) -> np.ndarray:
    return rng.uniform(r.low, r.high, size)


def _count(fraction: float, n: int) -> int:
    """Deterministic count (not Bernoulli) so e.g. 30% of 12 products is always 4."""
    return round(fraction * n)


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between point sets a (n,2) and b (m,2) -> (n,m)."""
    diff = a[:, None, :] - b[None, :, :]
    out: np.ndarray = np.sqrt((diff**2).sum(axis=-1))
    return out


def _locations(rng: np.random.Generator, n: int, grid: float) -> np.ndarray:
    return rng.uniform(0.0, grid, size=(n, 2))


# -------------------------------------------------------------------------- products
def generate_products(cfg: PharmaNetConfig, rng: np.random.Generator) -> pd.DataFrame:
    g = cfg.generation.products
    n = cfg.network.n_products
    is_cold = np.zeros(n, dtype=bool)
    is_cold[rng.permutation(n)[: _count(g.cold_fraction, n)]] = True
    return pd.DataFrame(
        {
            "product_id": _ids("SKU", n),
            "is_cold": is_cold,
            "unit_value": _uniform(rng, g.unit_value, n).round(2),
            "shelf_life_months": np.round(_uniform(rng, g.shelf_life_months, n)).astype(int),
        }
    )


# ---------------------------------------------------------------------------- demand
def generate_demand(
    cfg: PharmaNetConfig,
    rng: np.random.Generator,
    region_ids: list[str],
    product_ids: list[str],
) -> pd.DataFrame:
    """Monthly demand = base x trend x seasonality x noise, per (region, product).

    Trend, seasonal amplitude, and seasonal phase are drawn *per product* (a flu
    product peaks in winter everywhere); base level and noise are drawn *per
    region-product pair*. Noise is lognormal with mean 1, so it scales demand
    without biasing it and never makes it negative.
    """
    d = cfg.generation.demand
    R, K, T = len(region_ids), len(product_ids), cfg.network.n_months
    t = np.arange(T)

    trend = _uniform(rng, d.annual_trend, K)  # (K,)
    amp = _uniform(rng, d.seasonal_amplitude, K)  # (K,)
    phase = rng.uniform(0, MONTHS_PER_YEAR, K)  # (K,) month of the seasonal peak offset
    base = _uniform(rng, d.base_level, (R, K))  # (R,K)
    cv = _uniform(rng, d.noise_cv, (R, K))  # (R,K)

    trend_f = (1 + trend[:, None]) ** (t[None, :] / MONTHS_PER_YEAR)  # (K,T)
    season_f = 1 + amp[:, None] * np.sin(
        2 * np.pi * (t[None, :] - phase[:, None]) / MONTHS_PER_YEAR
    )

    # Lognormal with E[X]=1 and CV=cv: sigma^2 = ln(1+cv^2), mu = -sigma^2/2.
    sigma = np.sqrt(np.log1p(cv**2))[:, :, None]  # (R,K,1)
    noise = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=(R, K, T))

    values = base[:, :, None] * trend_f[None, :, :] * season_f[None, :, :] * noise

    # Intermittent series: zero out months at random, scale up the rest so the
    # long-run mean is unchanged (lumpy, not smaller).
    n_int = _count(cfg.generation.products.intermittent_fraction, R * K)
    if n_int > 0:
        p0 = d.intermittent_zero_prob
        flat = rng.choice(R * K, size=n_int, replace=False)
        ri, ki = np.unravel_index(flat, (R, K))
        zero_mask = rng.random((n_int, T)) < p0
        values[ri, ki, :] = np.where(zero_mask, 0.0, values[ri, ki, :] / (1 - p0))

    months = pd.date_range(cfg.generation.start_month, periods=T, freq="MS")
    idx = pd.MultiIndex.from_product(
        [region_ids, product_ids, months], names=["region_id", "product_id", "month"]
    )
    return pd.DataFrame(
        {"demand": np.round(values).astype(np.int64).ravel()}, index=idx
    ).reset_index()


def annual_demand(demand: pd.DataFrame, by: list[str]) -> pd.Series:
    """Annualized demand over the most recent 12 months (or all, if shorter).

    Used only to size synthetic capacities; real planning uses the forecast.
    """
    months = np.sort(demand["month"].unique())
    window = months[-MONTHS_PER_YEAR:]
    recent = demand[demand["month"].isin(window)]
    scale = MONTHS_PER_YEAR / len(window)
    return recent.groupby(by)["demand"].sum() * scale


# ---------------------------------------------------------------------------- plants
def generate_plant_products(
    cfg: PharmaNetConfig,
    rng: np.random.Generator,
    plant_ids: list[str],
    product_annual_demand: pd.Series,
) -> pd.DataFrame:
    """Qualification matrix and annual capacity per (plant, product).

    Total capacity for product k across all qualified plants is
    ``slack_k x annual_demand_k``, split among those plants with a uniform-random
    (Dirichlet(1,...,1)) share. Unqualified pairs have zero capacity.
    """
    g = cfg.generation.plants
    P = len(plant_ids)
    product_ids = list(product_annual_demand.index)
    K = len(product_ids)

    qual = rng.random((P, K)) < g.qualification_prob
    for orphan in np.flatnonzero(~qual.any(axis=0)):
        qual[rng.integers(P), orphan] = True  # every marketed product has >= 1 approved site

    slack = _uniform(rng, g.capacity_slack, K)
    capacity = np.zeros((P, K))
    for k in range(K):
        plants_k = np.flatnonzero(qual[:, k])
        shares = rng.dirichlet(np.ones(len(plants_k)))
        capacity[plants_k, k] = shares * slack[k] * float(product_annual_demand.iloc[k])

    idx = pd.MultiIndex.from_product([plant_ids, product_ids], names=["plant_id", "product_id"])
    return pd.DataFrame(
        {"qualified": qual.ravel(), "capacity": np.round(capacity.ravel(), 1)}, index=idx
    ).reset_index()


# ------------------------------------------------------------------------------- DCs
def generate_dcs(
    cfg: PharmaNetConfig,
    rng: np.random.Generator,
    locations: np.ndarray,
    total_annual_demand: float,
    any_cold_products: bool,
) -> pd.DataFrame:
    g = cfg.generation.dcs
    D = cfg.network.n_dcs
    throughput = _uniform(rng, g.throughput_slack, D) * total_annual_demand / D

    n_no_cold = _count(g.no_cold_fraction, D)
    if any_cold_products:
        n_no_cold = min(n_no_cold, D - 1)  # at least one DC must be able to hold cold stock
    has_cold = np.ones(D, dtype=bool)
    has_cold[rng.permutation(D)[:n_no_cold]] = False
    cold_capacity = np.where(has_cold, _uniform(rng, g.cold_share, D) * throughput, 0.0)

    return pd.DataFrame(
        {
            "dc_id": _ids("DC", D),
            "x": locations[:, 0],
            "y": locations[:, 1],
            "fixed_cost": _uniform(rng, g.fixed_cost, D).round(0),
            "throughput_capacity": throughput.round(1),
            "has_cold": has_cold,
            "cold_capacity": cold_capacity.round(1),
        }
    )


# ----------------------------------------------------------------------------- lanes
def generate_lanes(
    cfg: PharmaNetConfig,
    origin_ids: list[str],
    origin_xy: np.ndarray,
    dest_ids: list[str],
    dest_xy: np.ndarray,
    origin_col: str,
    dest_col: str,
    base_days: float,
) -> pd.DataFrame:
    """Every origin->destination lane with distance, per-unit cost, and lead time.

    Lanes are deterministic given locations (no randomness): cost and mean lead
    time grow linearly with distance; lead-time std is ``cv x mean``.
    """
    c, lt = cfg.costs, cfg.lead_times
    dist = _distances(origin_xy, dest_xy).ravel()
    cost = c.transport_per_unit_distance * dist
    lt_mean = base_days + lt.days_per_unit_distance * dist
    idx = pd.MultiIndex.from_product([origin_ids, dest_ids], names=[origin_col, dest_col])
    return pd.DataFrame(
        {
            "distance": dist.round(2),
            "cost_per_unit": cost.round(4),
            "cold_cost_per_unit": (cost * c.cold_chain_premium).round(4),
            "lead_time_mean_days": lt_mean.round(2),
            "lead_time_std_days": (lt.cv * lt_mean).round(2),
        },
        index=idx,
    ).reset_index()


def unmet_penalties(
    products: pd.DataFrame, lanes_in: pd.DataFrame, lanes_out: pd.DataFrame, multiplier: float
) -> pd.Series:
    """Per-unit penalty for unmet demand of each product.

    ``multiplier x (most expensive inbound lane + most expensive outbound lane)``
    using the cold rate for cold products. Because every possible path costs
    less than the penalty, the optimizer leaves demand unmet only when serving
    it is physically impossible, never because it is cheaper.
    """
    amb = lanes_in["cost_per_unit"].max() + lanes_out["cost_per_unit"].max()
    cold = lanes_in["cold_cost_per_unit"].max() + lanes_out["cold_cost_per_unit"].max()
    worst_path = np.where(products["is_cold"], cold, amb)
    return pd.Series(np.round(multiplier * worst_path, 4), index=products.index)


# ------------------------------------------------------------------------ top level
def generate(cfg: PharmaNetConfig) -> Dataset:
    """Generate a full synthetic dataset, then validate it. Same config -> same data."""
    streams = dict(
        zip(
            _STREAMS,
            (np.random.default_rng(s) for s in np.random.SeedSequence(cfg.run.seed).spawn(5)),
            strict=True,
        )
    )
    n, grid = cfg.network, cfg.network.grid_size

    products = generate_products(cfg, streams["products"])

    loc_rng = streams["locations"]
    plant_xy = _locations(loc_rng, n.n_plants, grid)
    dc_xy = _locations(loc_rng, n.n_dcs, grid)
    region_xy = _locations(loc_rng, n.n_regions, grid)

    plant_ids, region_ids = _ids("PL", n.n_plants), _ids("RG", n.n_regions)
    plants = pd.DataFrame({"plant_id": plant_ids, "x": plant_xy[:, 0], "y": plant_xy[:, 1]})
    regions = pd.DataFrame({"region_id": region_ids, "x": region_xy[:, 0], "y": region_xy[:, 1]})

    product_ids = list(products["product_id"])
    demand = generate_demand(cfg, streams["demand"], region_ids, product_ids)
    by_product = annual_demand(demand, ["product_id"]).reindex(product_ids)

    plant_products = generate_plant_products(cfg, streams["plants"], plant_ids, by_product)
    dcs = generate_dcs(
        cfg, streams["dcs"], dc_xy, float(by_product.sum()), bool(products["is_cold"].any())
    )

    lt = cfg.lead_times
    dc_ids = list(dcs["dc_id"])
    lanes_in = generate_lanes(
        cfg, plant_ids, plant_xy, dc_ids, dc_xy, "plant_id", "dc_id", lt.plant_dc_base_days
    )
    lanes_out = generate_lanes(
        cfg, dc_ids, dc_xy, region_ids, region_xy, "dc_id", "region_id", lt.dc_region_base_days
    )
    products = products.assign(
        unmet_penalty=unmet_penalties(
            products, lanes_in, lanes_out, cfg.costs.unmet_penalty_multiplier
        )
    )

    ds = Dataset(
        products=products,
        plants=plants,
        plant_products=plant_products,
        dcs=dcs,
        regions=regions,
        lanes_inbound=lanes_in,
        lanes_outbound=lanes_out,
        demand=demand,
    )
    validate_dataset(ds)
    return ds


# ------------------------------------------------------------------------ validation
def validate_dataset(ds: Dataset) -> None:
    """Independent structural checks. Collects *all* failures, then raises once.

    The aggregate capacity checks are *necessary* conditions for a feasible
    network design, not sufficient ones (geography and per-DC limits still
    matter); they catch badly configured instances early.
    """
    errors: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    for name, table, key in [
        ("products", ds.products, ["product_id"]),
        ("plants", ds.plants, ["plant_id"]),
        ("dcs", ds.dcs, ["dc_id"]),
        ("regions", ds.regions, ["region_id"]),
        ("plant_products", ds.plant_products, ["plant_id", "product_id"]),
        ("lanes_inbound", ds.lanes_inbound, ["plant_id", "dc_id"]),
        ("lanes_outbound", ds.lanes_outbound, ["dc_id", "region_id"]),
        ("demand", ds.demand, ["region_id", "product_id", "month"]),
    ]:
        check(not table.duplicated(key).any(), f"{name}: duplicate keys on {key}")
        check(not table.isna().any().any(), f"{name}: contains missing values")

    P, D, R, K = len(ds.plants), len(ds.dcs), len(ds.regions), len(ds.products)
    T = ds.demand["month"].nunique()
    check(len(ds.plant_products) == P * K, "plant_products: not one row per (plant, product)")
    check(len(ds.lanes_inbound) == P * D, "lanes_inbound: not one row per (plant, DC)")
    check(len(ds.lanes_outbound) == D * R, "lanes_outbound: not one row per (DC, region)")
    check(len(ds.demand) == R * K * T, "demand: not one row per (region, product, month)")

    check(bool((ds.demand["demand"] >= 0).all()), "demand: negative values")

    pp = ds.plant_products
    check(bool((pp["capacity"] >= 0).all()), "plant_products: negative capacity")
    check(
        bool(((pp["capacity"] > 0) == pp["qualified"]).all()),
        "plant_products: capacity must be > 0 exactly when qualified",
    )
    unqualified = set(ds.products["product_id"]) - set(pp.loc[pp["qualified"], "product_id"])
    check(not unqualified, f"products with no qualified plant: {sorted(unqualified)}")

    dcs = ds.dcs
    check(bool((dcs["fixed_cost"] > 0).all()), "dcs: non-positive fixed cost")
    check(bool((dcs["throughput_capacity"] > 0).all()), "dcs: non-positive throughput")
    check(
        bool((dcs["cold_capacity"] <= dcs["throughput_capacity"]).all()),
        "dcs: cold capacity exceeds throughput",
    )
    check(
        bool((dcs.loc[~dcs["has_cold"], "cold_capacity"] == 0).all()),
        "dcs: cold capacity on a DC without cold storage",
    )

    for name, lanes in [("lanes_inbound", ds.lanes_inbound), ("lanes_outbound", ds.lanes_outbound)]:
        check(bool((lanes["cost_per_unit"] >= 0).all()), f"{name}: negative cost")
        check(
            bool((lanes["cold_cost_per_unit"] >= lanes["cost_per_unit"]).all()),
            f"{name}: cold cost below ambient cost",
        )
        check(bool((lanes["lead_time_std_days"] >= 0).all()), f"{name}: negative lead-time std")

    max_amb = ds.lanes_inbound["cost_per_unit"].max() + ds.lanes_outbound["cost_per_unit"].max()
    max_cold = (
        ds.lanes_inbound["cold_cost_per_unit"].max() + ds.lanes_outbound["cold_cost_per_unit"].max()
    )
    worst = np.where(ds.products["is_cold"], max_cold, max_amb)
    check(
        bool((ds.products["unmet_penalty"].to_numpy() > worst).all()),
        "products: unmet penalty not above the most expensive path",
    )

    # Aggregate (necessary) feasibility.
    if not errors:
        by_product = annual_demand(ds.demand, ["product_id"])
        plant_cap = pp.groupby("product_id")["capacity"].sum().reindex(by_product.index)
        short = by_product[plant_cap < by_product]
        check(short.empty, f"plant capacity below annual demand for: {list(short.index)}")
        check(
            dcs["throughput_capacity"].sum() >= by_product.sum(),
            "total DC throughput below total annual demand",
        )
        cold_ids = ds.products.loc[ds.products["is_cold"], "product_id"]
        cold_demand = by_product.reindex(cold_ids).sum()
        check(
            dcs["cold_capacity"].sum() >= cold_demand,
            f"total DC cold capacity ({dcs['cold_capacity'].sum():.0f}) below "
            f"cold annual demand ({cold_demand:.0f})",
        )

    if errors:
        raise DataValidationError("Dataset validation failed:\n  - " + "\n  - ".join(errors))


def summarize(ds: Dataset) -> dict[str, float | int]:
    """Headline numbers about an instance, for the CLI and docs."""
    by_product = annual_demand(ds.demand, ["product_id"])
    cold_ids = ds.products.loc[ds.products["is_cold"], "product_id"]
    total = float(by_product.sum())
    series = ds.demand.groupby(["region_id", "product_id"])["demand"]
    return {
        "plants": len(ds.plants),
        "dcs": len(ds.dcs),
        "regions": len(ds.regions),
        "products": len(ds.products),
        "cold_products": int(ds.products["is_cold"].sum()),
        "months": int(ds.demand["month"].nunique()),
        "annual_demand_units": round(total),
        "cold_demand_share": round(float(by_product.reindex(cold_ids).sum()) / total, 3),
        "qualified_plant_product_pairs": int(ds.plant_products["qualified"].sum()),
        "single_source_products": int(
            (ds.plant_products.groupby("product_id")["qualified"].sum() == 1).sum()
        ),
        "dcs_with_cold": int(ds.dcs["has_cold"].sum()),
        "dc_throughput_to_demand": round(float(ds.dcs["throughput_capacity"].sum()) / total, 2),
        "series_over_20pct_zero_months": int((series.apply(lambda s: (s == 0).mean()) > 0.2).sum()),
    }
