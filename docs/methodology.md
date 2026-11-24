# Methodology

_All data is synthetic._

## Data dictionary

All tables are **synthetic**, produced by `pharmanet generate` (`src/pharmanet/data/generate.py`)
and written to `outputs/<run_id>/data/<table>.parquet`. IDs are zero-padded strings:
plants `PL01`, DCs `DC01`, regions `RG01`, products `SKU01`.

### `products` (one row per product)
| Column | Type | Meaning |
|---|---|---|
| `product_id` | str | Product key |
| `is_cold` | bool | Requires cold chain (exactly `round(cold_fraction × n_products)` products) |
| `unit_value` | float | $ per unit, uniform in `unit_value` range |
| `shelf_life_months` | int | Shelf life, uniform in `shelf_life_months` range |
| `unmet_penalty` | float | $ per unit of unmet demand = `unmet_penalty_multiplier × (max inbound lane cost + max outbound lane cost)` at the product's (cold or ambient) rate. Always above the cost of any physical path. |

### `plants` / `regions` / (location part of) `dcs`
| Column | Type | Meaning |
|---|---|---|
| `*_id` | str | Key |
| `x`, `y` | float | Coordinates, uniform in `[0, grid_size]²` |

### `plant_products` (one row per plant × product)
| Column | Type | Meaning |
|---|---|---|
| `plant_id`, `product_id` | str | Key |
| `qualified` | bool | Plant is approved to make the product (models regulatory site approval). Drawn with prob. `qualification_prob`; every product is forced to have ≥ 1 qualified plant. |
| `capacity` | float | Annual units. For product k, total capacity over qualified plants = `slack_k × annual_demand_k` with `slack_k ~ U(capacity_slack)`, split by a Dirichlet(1,…,1) share. 0 when not qualified. |

### `dcs` (one row per candidate DC)
| Column | Type | Meaning |
|---|---|---|
| `dc_id`, `x`, `y` | | Key and location |
| `fixed_cost` | float | $ per year to operate if opened, uniform in `fixed_cost` range (independent of size; a simplification) |
| `throughput_capacity` | float | Annual units through the DC = `U(throughput_slack) × total_annual_demand / n_dcs` |
| `has_cold` | bool | Has cold storage. `round(no_cold_fraction × n_dcs)` DCs have none; at least one always has it if cold products exist. |
| `cold_capacity` | float | Annual cold units = `U(cold_share) × throughput_capacity` if `has_cold`, else 0 |

### `lanes_inbound` (plant × DC) and `lanes_outbound` (DC × region)
| Column | Type | Meaning |
|---|---|---|
| `distance` | float | Euclidean distance |
| `cost_per_unit` | float | `transport_per_unit_distance × distance` ($/unit, ambient) |
| `cold_cost_per_unit` | float | `cost_per_unit × cold_chain_premium` |
| `lead_time_mean_days` | float | `base_days + days_per_unit_distance × distance` (base differs for inbound vs outbound) |
| `lead_time_std_days` | float | `lead_time.cv × lead_time_mean_days` |

### `demand` (one row per region × product × month)
| Column | Type | Meaning |
|---|---|---|
| `region_id`, `product_id` | str | Key |
| `month` | timestamp | Month start, `n_months` consecutive months from `start_month` |
| `demand` | int | Units demanded |

**Demand model.** For region r, product k, month t:

$$
d_{rkt} = \text{round}\Big( b_{rk} \cdot (1+g_k)^{t/12} \cdot \big(1 + A_k \sin\tfrac{2\pi (t-\phi_k)}{12}\big) \cdot \varepsilon_{rkt} \Big)
$$

- $b_{rk}$: base level (per pair); $g_k$: annual trend, $A_k$: seasonal amplitude, $\phi_k$: seasonal phase (per **product**, so a product's peak season is shared across regions)
- $\varepsilon_{rkt} \sim \text{LogNormal}(\mu, \sigma^2)$ with $\sigma^2 = \ln(1+cv_{rk}^2)$, $\mu = -\sigma^2/2$, so $E[\varepsilon]=1$ and the noise has coefficient of variation $cv_{rk}$
- **Intermittent series** (`round(intermittent_fraction × R × K)` pairs): each month is zero with probability $p_0$ and non-zero months are scaled by $1/(1-p_0)$, preserving the long-run mean

**Capacity sizing** uses annualized demand over the last 12 months of history. This is only to make
the synthetic instance sensible; planning (Phase 4) uses forecast demand plus safety stock.

**Validation** (`validate_dataset`, run on generate *and* on every load): unique keys, no missing
values, complete grids, non-negative demand/capacity, capacity > 0 iff qualified, every product has
a qualified plant, cold capacity only on cold DCs and ≤ throughput, cold cost ≥ ambient cost, penalty
above the worst path, and aggregate capacity ≥ annual demand (plant per product, DC total, DC cold).
The aggregate checks are necessary but not sufficient for feasibility.

**Reproducibility.** The master `run.seed` is split with `numpy.random.SeedSequence.spawn` into
independent streams for products, locations, demand, plants and DCs.

## Forecasting

Code: `src/pharmanet/forecasting/` (`models.py`, `backtest.py`, `forecast.py`). Run with
`pharmanet forecast`; outputs go to `outputs/<run_id>/forecast/`.

### Models
| Model | Idea | Spec (config `forecasting`) |
|---|---|---|
| Seasonal naive (baseline) | $\hat y_{T+h} = y_{T+h-m\lceil h/m\rceil}$: same month last season | $m$ = `seasonal_period` = 12 |
| ETS / Holt-Winters | Exponentially weighted level, trend and seasonal components | additive damped trend, additive seasonality (additive so zero months are allowed) |
| SARIMA | Autoregression on past values and seasonal differences | $(1,0,0)(0,1,1)_{12}$ with constant (a drift on the seasonally differenced series, i.e. a linear trend) |

All forecasts are clipped at 0. If a statistical model fails to fit or returns non-finite
values, it falls back to seasonal naive and the fallback is counted and reported.

### Rolling-origin backtest
For each of the last `backtest_months` (12) months $t$: fit on $y_{1..t-1}$, forecast $y_t$
one step ahead. There are 12 out-of-sample points per series and model. One step ahead is the
horizon safety stock must cover, because replenishment lead times are days. A unit test checks
that the model at origin $t$ only ever sees $y_{1..t-1}$ (no leakage).

### Metrics
- **WAPE** $=\sum|A-F| / \sum A$ (primary): volume-weighted, defined with zero-demand months.
- **MAPE** $=\text{mean}(|A-F|/A)$ over months with $A>0$; undefined on zeros, which is why it is secondary.
- **Bias** $=\sum(F-A)/\sum A$; positive = over-forecasting.
- **RMSE**, in units, used as $\sigma_d$ for safety stock.

Portfolio figures pool all series, so high-volume series weigh more.

### Model selection, two strategies, and the selection-bias caveat
`forecasting.selection_strategy` sets how models are chosen (ties go to the simpler model):

- **`per_series`** (spec default): each (region, product) gets its lowest-WAPE model.
- **`global`**: one model, the one with the lowest *pooled* WAPE across all series, is used for
  every series. It makes far fewer choices, so there is less room to fit noise.

Choosing and scoring on the **same** 12 months is optimistic. Some "wins" are noise, and the
reported error of the winner is biased downward. Every run therefore also reports an **honest
holdout estimate for both strategies**: choose on the first
`backtest_months − selection_holdout_months` months and score on the last
`selection_holdout_months` (6), which the choice never saw. It reuses the backtest errors, so it
needs no extra fits.

**Base instance (synthetic, seed 42):**

| Measure | per_series | global (→ ETS for all) |
|---|---|---|
| WAPE on the months used to choose (optimistic) | 0.207 | 0.222 |
| **Honest holdout WAPE (last 6 mo)** | 0.219 | **0.214** |
| **Holdout improvement vs seasonal naive (0.235)** | 7.2% | **9.2%** |

| Other diagnostics | |
|---|---|
| Portfolio WAPE, all 12 mo: seasonal naive / ETS / SARIMA | 0.239 / 0.222 / 0.235 |
| WAPE on regular vs intermittent series (ETS) | 0.13 vs 1.04 |

Takeaways: (1) per-series selection looks best in-sample and is worst out-of-sample, which is
overfitting at the model-selection level. With short histories, one good global model wins.
(2) Quote holdout figures only. (3) Intermittent series dominate the error. Croston-type
methods would be the next step for them.

## Safety stock

Code: `src/pharmanet/inventory/safety_stock.py`.

$$
SS_{rk} = z \cdot \sqrt{L_{rk}\,\sigma_{d,rk}^2 + \bar d_{rk}^2\,\sigma_{L,rk}^2}, \qquad z = \Phi^{-1}(\text{service level})
$$

- $\sigma_{d,rk}$: one-step backtest **RMSE** of the selected model (units/month). RMSE rather than
  the error std, so bias increases protection instead of being ignored.
- $\bar d_{rk}$: mean monthly forecast over the 12-month horizon.
- $L_{rk}, \sigma_{L,rk}$: lead-time mean and std in months (days / 30.44).
- The first term covers demand uncertainty during the lead time; the second covers lead-time
  uncertainty at average demand. It assumes demand and lead time are independent and roughly
  normal. That is weak for intermittent series, so their safety stock is less reliable.

**Lead time and the circular dependency.** $L_{rk}$ depends on which plant and DC serve region
$r$, and the network design decides that using safety stock as an input. We break the loop with
the **fastest feasible path**: a qualified plant → a DC that can store the product (cold storage
if needed) → the region, with path std $\sqrt{\sigma_{in}^2+\sigma_{out}^2}$ (independent legs).
This is optimistic. Phase 4 recomputes safety stock with the chosen assignments and reports the
gap.

**Design demand** for the MILP: $dem_{rk} = \text{annual forecast}_{rk} + SS_{rk}$. This adds a
stock *level* to an annual *flow*, so it reserves capacity for the buffer. It is a standard
simplification in single-period strategic design (revisited in Phase 4). On the base instance,
safety stock is 2.3% of annual volume at a 95% service level, because lead times (5–16 days) are
short compared with the monthly forecast error.

## Simulation
_Phase 5._

## Graph risk analysis
_Phase 6._
