"""Safety stock from forecast error and lead-time variability.

Standard formula (demand and lead time independent, both roughly normal):

    SS = z * sqrt( L * sigma_d^2 + d_bar^2 * sigma_L^2 )

with L, sigma_L in months, sigma_d the one-step-ahead forecast RMSE (units per
month), d_bar mean monthly forecast demand, and z = Phi^{-1}(service level).

Lead time and the circular dependency
-------------------------------------
L depends on which plant and DC serve a region, which the network design
(Phase 4) decides, and that design itself needs safety stock as input. We
break the loop by estimating L from the *fastest feasible path*: a qualified
plant -> a DC able to store the product (cold storage for cold products) ->
the region. This is optimistic; Phase 4 recomputes SS with the chosen
assignments and reports the difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from pharmanet.data.generate import MONTHS_PER_YEAR, Dataset

DAYS_PER_MONTH = 365.25 / 12  # physical constant


def z_value(service_level: float) -> float:
    """Number of standard deviations of protection for a cycle service level."""
    if not 0 < service_level < 1:
        raise ValueError(f"service_level must be in (0, 1), got {service_level}")
    return float(norm.ppf(service_level))


def safety_stock(
    sigma_d: np.ndarray,
    mean_d: np.ndarray,
    lt_mean_months: np.ndarray,
    lt_std_months: np.ndarray,
    z: float,
) -> np.ndarray:
    """Vectorized safety stock formula (see module docstring). All inputs >= 0."""
    for name, arr in [
        ("sigma_d", sigma_d),
        ("mean_d", mean_d),
        ("lt_mean_months", lt_mean_months),
        ("lt_std_months", lt_std_months),
    ]:
        if np.any(np.asarray(arr) < 0):
            raise ValueError(f"{name} must be non-negative")
    variance = lt_mean_months * sigma_d**2 + mean_d**2 * lt_std_months**2
    out: np.ndarray = z * np.sqrt(variance)
    return out


def fastest_path_lead_times(ds: Dataset) -> pd.DataFrame:
    """Per (region, product): mean/std lead time of the fastest feasible plant->DC->region path.

    Feasible = plant qualified for the product, and DC has cold storage if the
    product is cold. Path std assumes the two legs vary independently:
    sigma_path = sqrt(sigma_in^2 + sigma_out^2).
    """
    lin = ds.lanes_inbound[["plant_id", "dc_id", "lead_time_mean_days", "lead_time_std_days"]]
    lout = ds.lanes_outbound[["dc_id", "region_id", "lead_time_mean_days", "lead_time_std_days"]]
    paths = lin.merge(lout, on="dc_id", suffixes=("_in", "_out"))
    paths["lt_mean_days"] = paths["lead_time_mean_days_in"] + paths["lead_time_mean_days_out"]
    paths["lt_std_days"] = np.sqrt(
        paths["lead_time_std_days_in"] ** 2 + paths["lead_time_std_days_out"] ** 2
    )

    qual = ds.plant_products.loc[ds.plant_products["qualified"], ["plant_id", "product_id"]]
    prod = ds.products[["product_id", "is_cold"]]
    dcs = ds.dcs[["dc_id", "has_cold"]]

    feasible = (
        paths.merge(qual, on="plant_id")
        .merge(prod, on="product_id")
        .merge(dcs, on="dc_id")
        .query("has_cold or not is_cold")
    )
    best = feasible.sort_values(["region_id", "product_id", "lt_mean_days", "plant_id", "dc_id"])
    best = best.groupby(["region_id", "product_id"], sort=True).head(1)
    return best[
        ["region_id", "product_id", "plant_id", "dc_id", "lt_mean_days", "lt_std_days"]
    ].reset_index(drop=True)


def compute_safety_stock(
    forecasts: pd.DataFrame,
    selection: pd.DataFrame,
    lead_times: pd.DataFrame,
    service_level: float,
) -> pd.DataFrame:
    """Safety stock and design demand per (region, product).

    ``design_demand = annual_forecast + safety_stock`` is the ``dem_rk`` input
    to the network design MILP. Note this adds a stock *level* to an annual
    *flow*: it reserves network capacity for holding the buffer, a common
    simplification in single-period strategic design.
    """
    key = ["region_id", "product_id"]
    horizon = forecasts.groupby(key)["month"].nunique().iloc[0]
    annual = (forecasts.groupby(key)["forecast"].sum() * MONTHS_PER_YEAR / horizon).rename(
        "annual_forecast"
    )
    df = (
        annual.reset_index()
        .merge(selection[[*key, "best_model", "rmse"]], on=key, validate="one_to_one")
        .merge(lead_times, on=key, how="left", validate="one_to_one")
    )
    if df["lt_mean_days"].isna().any():
        missing = df.loc[df["lt_mean_days"].isna(), key].to_numpy().tolist()
        raise ValueError(f"No feasible supply path for: {missing[:5]}")

    z = z_value(service_level)
    df["mean_monthly_forecast"] = df["annual_forecast"] / MONTHS_PER_YEAR
    df["sigma_monthly"] = df["rmse"]
    df["z"] = z
    df["safety_stock"] = safety_stock(
        df["sigma_monthly"].to_numpy(),
        df["mean_monthly_forecast"].to_numpy(),
        df["lt_mean_days"].to_numpy() / DAYS_PER_MONTH,
        df["lt_std_days"].to_numpy() / DAYS_PER_MONTH,
        z,
    )
    df["design_demand"] = df["annual_forecast"] + df["safety_stock"]
    return df.drop(columns="rmse").rename(columns={"plant_id": "lt_plant_id", "dc_id": "lt_dc_id"})


def flow_weighted_lead_times(
    ds: Dataset,
    inbound: dict[tuple[str, str, str], float],
    outbound: dict[tuple[str, str, str], float],
    fallback: pd.DataFrame,
) -> pd.DataFrame:
    """Lead times implied by an actual network design (closes the Phase 3 loop).

    For each (region, product), average the plant->DC->region path lead time over
    the DCs that actually serve it, weighted by flow; each DC's inbound leg is in
    turn weighted by which plants feed it that product. Variances are averaged
    the same way (between-path spread is ignored, a mild simplification).
    Pairs with no flow (fully unmet) keep their ``fallback`` (fastest-path) values.

    ``inbound`` maps (plant, dc, product) -> flow and ``outbound`` maps
    (dc, region, product) -> flow, as in an optimization ``Solution``.
    """
    lin = ds.lanes_inbound.set_index(["plant_id", "dc_id"])
    lout = ds.lanes_outbound.set_index(["dc_id", "region_id"])
    in_mean: dict[Any, float] = lin["lead_time_mean_days"].to_dict()
    in_sd: dict[Any, float] = lin["lead_time_std_days"].to_dict()
    out_mean: dict[Any, float] = lout["lead_time_mean_days"].to_dict()
    out_sd: dict[Any, float] = lout["lead_time_std_days"].to_dict()

    # Inbound leg per (dc, product): flow-weighted over feeding plants.
    by_dk: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for (p, d, k), q in inbound.items():
        by_dk.setdefault((d, k), []).append((p, q))
    in_leg: dict[tuple[str, str], tuple[float, float, str]] = {}
    for (d, k), feeds in by_dk.items():
        total = sum(q for _, q in feeds)
        mean = sum(q * in_mean[(p, d)] for p, q in feeds) / total
        var = sum(q * in_sd[(p, d)] ** 2 for p, q in feeds) / total
        main_plant = max(feeds, key=lambda t: t[1])[0]
        in_leg[(d, k)] = (mean, var, main_plant)

    by_rk: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for (d, r, k), q in outbound.items():
        by_rk.setdefault((r, k), []).append((d, q))

    out = fallback.set_index(["region_id", "product_id"]).copy()
    for (r, k), serves in by_rk.items():
        total = sum(q for _, q in serves)
        mean = var = 0.0
        for d, q in serves:
            m_in, v_in, _ = in_leg[(d, k)]
            w = q / total
            mean += w * (m_in + out_mean[(d, r)])
            var += w * (v_in + out_sd[(d, r)] ** 2)
        main_dc = max(serves, key=lambda t: t[1])[0]
        out.loc[(r, k), ["lt_mean_days", "lt_std_days"]] = [mean, np.sqrt(var)]
        out.loc[(r, k), ["plant_id", "dc_id"]] = [in_leg[(main_dc, k)][2], main_dc]
    return out.reset_index()
