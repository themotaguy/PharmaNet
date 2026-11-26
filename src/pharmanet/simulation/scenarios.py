"""Scenario generation: Monte Carlo demand and single-point disruptions.

Demand scenarios
----------------
Annual demand for each (region, product) is drawn around the forecast F with
the forecast's own error: sigma_annual = sigma_monthly * sqrt(12), i.e. monthly
one-step errors treated as independent (optimistic: real errors persist, which
would widen the spread). Draws are lognormal with mean F, so demand stays
positive and is unbiased:

    D = F * exp(s * Z - s^2 / 2),   s^2 = ln(1 + cv^2),   cv = sigma_annual / F

Shocks are correlated across regions *for the same product* via a common
factor: Z_rk = sqrt(rho) * G_k + sqrt(1 - rho) * E_rk with G, E standard normal.
rho = ``demand_common_shock_share``. If regions were independent, their
surprises would largely cancel in aggregate and capacity risk would look
smaller than it is.

Disruption scenarios
--------------------
One failure at a time, at forecast demand: each plant down, each *open* DC
down, and each lane that carries cold product in the base design losing its
cold transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pharmanet.config import SimulationConfig
from pharmanet.data.generate import MONTHS_PER_YEAR
from pharmanet.optimization.instance import RK, NetworkInstance
from pharmanet.optimization.solution import Solution
from pharmanet.optimization.stochastic import WeightedScenario

SIMULATION_STREAM = 100  # SeedSequence spawn key: independent of the data generator's streams


STOCHASTIC_IN_SAMPLE_STREAM = 200
STOCHASTIC_OUT_OF_SAMPLE_STREAM = 300


def simulation_rng(seed: int, stream: int = SIMULATION_STREAM) -> np.random.Generator:
    """Independent random stream per purpose (Phase 5 Monte Carlo, SP in/out of sample)."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    kind: str  # "demand", "plant_outage", "dc_outage", "cold_lane_failure"
    target: str  # what failed ("" for demand scenarios)
    instance: NetworkInstance


def sample_demand(
    forecast: pd.DataFrame, n: int, common_share: float, rng: np.random.Generator
) -> tuple[list[RK], np.ndarray]:
    """Draw ``n`` annual demand scenarios.

    ``forecast`` needs columns region_id, product_id, annual_forecast, sigma_monthly.
    Returns (keys, array of shape (n, len(keys))).
    """
    f = forecast.sort_values(["region_id", "product_id"]).reset_index(drop=True)
    keys: list[RK] = list(zip(f["region_id"], f["product_id"], strict=True))
    mean = f["annual_forecast"].to_numpy(dtype=float)
    sigma = f["sigma_monthly"].to_numpy(dtype=float) * np.sqrt(MONTHS_PER_YEAR)
    cv = np.divide(sigma, mean, out=np.zeros_like(mean), where=mean > 0)
    s = np.sqrt(np.log1p(cv**2))

    products, prod_idx = np.unique(f["product_id"].to_numpy(), return_inverse=True)
    common = rng.standard_normal((n, len(products)))[:, prod_idx]
    idio = rng.standard_normal((n, len(keys)))
    z = np.sqrt(common_share) * common + np.sqrt(1 - common_share) * idio
    draws = mean * np.exp(s * z - s**2 / 2)
    return keys, draws


def demand_scenarios(
    base: NetworkInstance, forecast: pd.DataFrame, cfg: SimulationConfig, rng: np.random.Generator
) -> list[Scenario]:
    keys, draws = sample_demand(forecast, cfg.n_scenarios, cfg.demand_common_shock_share, rng)
    return [
        Scenario(
            scenario_id=f"mc_{i:04d}",
            kind="demand",
            target="",
            instance=base.with_demand(dict(zip(keys, row.tolist(), strict=True))),
        )
        for i, row in enumerate(draws)
    ]


def disruption_scenarios(
    base: NetworkInstance, design: Solution, cfg: SimulationConfig
) -> list[Scenario]:
    """Single-point failures relevant to the chosen design."""
    out: list[Scenario] = []
    flags = cfg.disruptions
    if flags.plant_outage:
        for p in base.plants:
            inst = base.plant_outage(p, cfg.outage_capacity_loss)
            out.append(Scenario(f"plant_{p}", "plant_outage", p, inst))
    if flags.dc_outage:
        for d in sorted(design.open_dcs):
            out.append(Scenario(f"dc_{d}", "dc_outage", d, base.dc_outage(d)))
    if flags.cold_lane_failure:
        cold = sorted(base.cold_products)
        in_lanes = sorted({(p, d) for (p, d, k) in design.x if k in base.cold_products})
        out_lanes = sorted({(d, r) for (d, r, k) in design.z if k in base.cold_products})
        for p, d in in_lanes:
            inst = base.block_inbound(frozenset((p, d, k) for k in cold))
            out.append(Scenario(f"cold_{p}_{d}", "cold_lane_failure", f"{p}->{d}", inst))
        for d, r in out_lanes:
            inst = base.block_outbound(frozenset((d, r, k) for k in cold))
            out.append(Scenario(f"cold_{d}_{r}", "cold_lane_failure", f"{d}->{r}", inst))
    return out


def stochastic_scenarios(
    base: NetworkInstance,
    forecast: pd.DataFrame,
    n_demand: int,
    disruption_weight: float,
    common_share: float,
    outage_capacity_loss: float,
    rng: np.random.Generator,
    prefix: str,
) -> list[WeightedScenario]:
    """Weighted scenarios for the two-stage model or its evaluation.

    ``n_demand`` demand draws share probability 1 - w. Every single plant outage
    and every single *candidate* DC outage (at forecast demand) share w, so the
    model can weigh resilience against cost when choosing which DCs to open.
    """
    keys, draws = sample_demand(forecast, n_demand, common_share, rng)
    outages: list[tuple[str, NetworkInstance]] = []
    if disruption_weight > 0:
        outages += [(f"plant_{p}", base.plant_outage(p, outage_capacity_loss)) for p in base.plants]
        outages += [(f"dc_{d}", base.dc_outage(d)) for d in base.dcs]
    w = disruption_weight if outages else 0.0
    out = [
        WeightedScenario(
            f"{prefix}_demand_{i:03d}",
            (1 - w) / n_demand,
            base.with_demand(dict(zip(keys, row.tolist(), strict=True))),
        )
        for i, row in enumerate(draws)
    ]
    out += [WeightedScenario(f"{prefix}_{name}", w / len(outages), inst) for name, inst in outages]
    return out
