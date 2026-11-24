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
_Phase 3._

## Safety stock
_Phase 3._

## Simulation
_Phase 5._

## Graph risk analysis
_Phase 6._
