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


class ETSConfig(_Frozen):
    """Holt-Winters exponential smoothing specification (statsmodels)."""

    trend: Literal["add", "mul"] | None = "add"
    damped_trend: bool = True
    seasonal: Literal["add", "mul"] | None = "add"


class SARIMAConfig(_Frozen):
    """SARIMA(p,d,q)(P,D,Q)_m; the seasonal period m comes from ``seasonal_period``."""

    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: tuple[int, int, int] = (0, 1, 1)
    trend: Literal["n", "c", "t", "ct"] = "c"


class ForecastConfig(_Frozen):
    models: list[ForecastModelName] = Field(min_length=1)
    ets: ETSConfig = ETSConfig()
    sarima: SARIMAConfig = SARIMAConfig()
    seasonal_period: int = Field(ge=1)
    horizon: int = Field(ge=1, description="Months to forecast forward.")
    backtest_months: int = Field(ge=1, description="Rolling-origin evaluation window.")
    selection_metric: Literal["wape", "mape"] = "wape"
    selection_strategy: Literal["per_series", "global"] = Field(
        default="per_series",
        description=(
            "per_series: each region-product gets its own best model. global: one model "
            "(lowest pooled error across all series) is used everywhere."
        ),
    )
    selection_holdout_months: int = Field(
        ge=1,
        description=(
            "For the unbiased accuracy estimate: choose models on the earlier backtest "
            "months and score on the last N, which the choice never saw."
        ),
    )
    n_jobs: int | None = Field(
        default=None, ge=1, description="Parallel worker processes; null = all CPU cores."
    )

    @model_validator(mode="after")
    def _holdout_fits(self) -> Self:
        if self.selection_holdout_months >= self.backtest_months:
            raise ValueError("selection_holdout_months must be < backtest_months")
        return self

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
    strong_linking: bool = Field(
        default=True,
        description="Add z_drk <= dem_rk * y_d (C6): same optimum, much tighter LP relaxation.",
    )
    single_sourcing: bool = False
    max_open_dcs: int | None = Field(default=None, ge=1)
    n_minus_1_dc: bool = Field(
        default=False,
        description=(
            "Resilient design (C8/C9): open DC capacity must still cover demand after "
            "losing any one open DC."
        ),
    )
    ss_refinement_iterations: int = Field(
        default=1,
        ge=0,
        description=(
            "Re-solves after recomputing safety stock from the chosen network's actual "
            "lead times (closes the forecast <-> design loop)."
        ),
    )


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
    demand_common_shock_share: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "Share of each product's demand-shock variance common to all regions "
            "(0 = independent regions, 1 = all regions move together)."
        ),
    )
    outage_capacity_loss: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Share of a plant's annual capacity lost in a plant-outage scenario.",
    )
    n_jobs: int | None = Field(default=None, ge=1, description="Parallel processes; null = all.")
    disruptions: DisruptionConfig
    sensitivity: SensitivityConfig


# -------------------------------------------------------------------------- graph
class GraphConfig(_Frozen):
    backup_capacity_share: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Dual-qualification mitigation: capacity given to a single-source product's new "
            "backup plant, as a share of that product's total design demand."
        ),
    )


# --------------------------------------------------------------------- stochastic
class StochasticConfig(_Frozen):
    enabled: bool = True
    n_scenarios: int = Field(default=30, ge=1, description="In-sample scenarios in the SP.")
    disruption_weight: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description=(
            "Total probability on single plant / candidate-DC outage scenarios "
            "(0 = demand uncertainty only)."
        ),
    )
    n_eval_scenarios: int = Field(
        default=200, ge=1, description="Fresh out-of-sample demand scenarios for evaluation."
    )
    compute_wait_and_see: bool = Field(
        default=True, description="Solve one MILP per in-sample scenario for WS and EVPI."
    )


# ------------------------------------------------------------------------- report
class ReportConfig(_Frozen):
    benchmark_seeds: int = Field(
        default=10,
        ge=0,
        description="Generated networks for the MILP-vs-greedy benchmark in run-all (0 = skip).",
    )


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
    graph: GraphConfig = GraphConfig()
    report: ReportConfig = ReportConfig()
    stochastic: StochasticConfig = StochasticConfig()

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
