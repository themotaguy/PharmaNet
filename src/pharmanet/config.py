"""Typed configuration for PharmaNet.

Why this exists: every number that shapes a result (network size, costs, seeds,
service level, solver settings) lives in YAML, not in code. Pydantic validates
the YAML at load time so a bad config fails immediately with a clear message,
instead of silently producing a nonsensical optimization or simulation run.

Models are frozen (immutable) so a config object cannot be mutated mid-pipeline;
sensitivity analysis creates modified *copies* via ``model_copy(update=...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    """Base for all config sections: immutable, and unknown keys are errors.

    ``extra="forbid"`` catches YAML typos (e.g. ``n_plant`` instead of
    ``n_plants``) that would otherwise be silently ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Range(_Frozen):
    """Closed interval [low, high] used to sample synthetic parameters uniformly."""

    low: float
    high: float

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"Range low ({self.low}) must be <= high ({self.high})")
        return self


# --------------------------------------------------------------------------- run
class RunConfig(_Frozen):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0, description="Master seed; module seeds are derived from it.")
    output_dir: Path = Path("outputs")


# ----------------------------------------------------------------------- network
class NetworkSize(_Frozen):
    n_plants: int = Field(ge=1)
    n_dcs: int = Field(ge=1)
    n_regions: int = Field(ge=1)
    n_products: int = Field(ge=1)
    n_months: int = Field(ge=1, description="Months of demand history to generate.")
    grid_size: float = Field(gt=0, description="Locations are sampled in [0, grid_size]^2.")


# -------------------------------------------------------------------- generation
class ProductGen(_Frozen):
    cold_fraction: float = Field(ge=0, le=1, description="Share of products needing cold chain.")
    intermittent_fraction: float = Field(
        ge=0, le=1, description="Share of product-region series with intermittent demand."
    )
    unit_value: Range
    shelf_life_months: Range


class PlantGen(_Frozen):
    qualification_prob: float = Field(
        gt=0, le=1, description="P(plant is approved to make a given product)."
    )
    capacity_slack: Range = Field(
        description="Total plant capacity per product as a multiple of total annual demand."
    )


class DCGen(_Frozen):
    fixed_cost: Range
    throughput_slack: Range = Field(
        description="DC throughput capacity as a multiple of (total demand / n_dcs)."
    )
    no_cold_fraction: float = Field(ge=0, le=1, description="Share of DCs with no cold storage.")
    cold_share: Range = Field(description="Cold capacity as a share of throughput, if any.")


class DemandGen(_Frozen):
    base_level: Range = Field(description="Mean monthly units per product-region.")
    annual_trend: Range = Field(description="Fractional growth per year, e.g. 0.05 = +5%.")
    seasonal_amplitude: Range = Field(description="Peak seasonal deviation as a fraction.")
    noise_cv: Range = Field(description="Coefficient of variation of multiplicative noise.")
    intermittent_zero_prob: float = Field(
        ge=0, lt=1, description="P(month has zero demand) for intermittent series."
    )


class GenerationConfig(_Frozen):
    start_month: str = Field(
        pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="First month of history, 'YYYY-MM'."
    )
    products: ProductGen
    plants: PlantGen
    dcs: DCGen
    demand: DemandGen


# ------------------------------------------------------------------------- costs
class CostConfig(_Frozen):
    transport_per_unit_distance: float = Field(gt=0)
    cold_chain_premium: float = Field(
        ge=1, description="Multiplier on transport cost for cold products."
    )
    unmet_penalty_multiplier: float = Field(
        gt=1,
        description=(
            "Unmet-demand penalty = multiplier x max possible transport cost for the product, "
            "so leaving demand unmet is never cheaper than serving it."
        ),
    )


# -------------------------------------------------------------------- lead times
class LeadTimeConfig(_Frozen):
    plant_dc_base_days: float = Field(ge=0)
    dc_region_base_days: float = Field(ge=0)
    days_per_unit_distance: float = Field(ge=0)
    cv: float = Field(ge=0, description="Coefficient of variation of lead time.")


# ------------------------------------------------------------------- forecasting
ForecastModelName = Literal["seasonal_naive", "ets", "sarima"]


class ForecastConfig(_Frozen):
    models: list[ForecastModelName] = Field(min_length=1)
    seasonal_period: int = Field(ge=1)
    horizon: int = Field(ge=1, description="Months to forecast forward.")
    backtest_months: int = Field(ge=1, description="Rolling-origin evaluation window.")
    selection_metric: Literal["wape", "mape", "bias"] = "wape"

    @model_validator(mode="after")
    def _baseline_present(self) -> Self:
        # Improvement over seasonal naive is a headline result, so it must always run.
        if "seasonal_naive" not in self.models:
            raise ValueError("forecasting.models must include 'seasonal_naive' (the baseline)")
        return self


# --------------------------------------------------------------------- inventory
class InventoryConfig(_Frozen):
    service_level: float = Field(gt=0.5, lt=1, description="Target cycle service level.")


# ------------------------------------------------------------------ optimization
class SolverConfig(_Frozen):
    name: Literal["highs", "gurobi", "cbc"] = "highs"
    time_limit_s: float = Field(gt=0)
    mip_gap: float = Field(ge=0, lt=1)
    threads: int | None = Field(default=None, ge=1)


class OptimizationConfig(_Frozen):
    solver: SolverConfig
    single_sourcing: bool = False
    max_open_dcs: int | None = Field(default=None, ge=1)


# -------------------------------------------------------------------- simulation
class DisruptionConfig(_Frozen):
    plant_outage: bool = True
    dc_outage: bool = True
    cold_lane_failure: bool = True


class SensitivityConfig(_Frozen):
    fixed_cost_multipliers: list[float] = Field(min_length=1)
    transport_cost_multipliers: list[float] = Field(min_length=1)
    demand_multipliers: list[float] = Field(min_length=1)


class SimulationConfig(_Frozen):
    n_scenarios: int = Field(ge=1)
    disruptions: DisruptionConfig
    sensitivity: SensitivityConfig


# ------------------------------------------------------------------------- root
class PharmaNetConfig(_Frozen):
    run: RunConfig
    network: NetworkSize
    generation: GenerationConfig
    costs: CostConfig
    lead_times: LeadTimeConfig
    forecasting: ForecastConfig
    inventory: InventoryConfig
    optimization: OptimizationConfig
    simulation: SimulationConfig

    @model_validator(mode="after")
    def _cross_section_checks(self) -> Self:
        """Checks that span sections, which per-field validators can't express."""
        n = self.network
        f = self.forecasting
        # Need at least two seasonal cycles to fit seasonal models, plus the backtest window.
        min_history = 2 * f.seasonal_period + f.backtest_months
        if n.n_months < min_history:
            raise ValueError(
                f"network.n_months={n.n_months} is too short: need >= 2*seasonal_period + "
                f"backtest_months = {min_history}"
            )
        cap = self.optimization.max_open_dcs
        if cap is not None and cap > n.n_dcs:
            raise ValueError(f"optimization.max_open_dcs={cap} exceeds network.n_dcs={n.n_dcs}")
        return self


def load_config(path: str | Path) -> PharmaNetConfig:
    """Load and validate a YAML config file.

    Raises:
        FileNotFoundError: if the path does not exist.
        pydantic.ValidationError: if any value is missing, mistyped, or out of range.
    """
    p = Path(path)
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {p} must be a YAML mapping at the top level")
    return PharmaNetConfig.model_validate(raw)
