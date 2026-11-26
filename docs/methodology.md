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

## Optimization: baseline, validation, benchmarks

The MILP itself is specified in [formulation.md](formulation.md).

**Greedy baseline** (`optimization/baseline.py`):
1. Open DCs in order of increasing fixed cost until open capacity covers total demand *and* open cold capacity covers cold demand.
2. Route (region, product) demand **cold first**, then largest first, to the nearest open DC that can hold it. Source it from the nearest qualified plant with capacity left, splitting when capacity runs out. Anything left over is unmet.

*Why cold first:* an earlier version routed largest-first. Ambient volume then filled the cold-capable DCs, 42k cold units went unmet, and the MILP's "40% saving" was mostly the artificial unmet-demand penalty. That baseline was a strawman. Allocating the scarce resource first is what a planner would do, and it gives a fair comparison.

**Decomposing the saving.** We also solve *greedy DCs + optimal flows* (y fixed to greedy's choice, flows re-optimized):
routing saving = greedy − (greedy DCs + optimal flows); DC-choice saving = (greedy DCs + optimal flows) − MILP.
Penalty-free logistics cost per delivered unit is reported too, so a saving cannot come from penalty accounting alone.

**Validation** (`optimization/validate.py`) runs on every solution from every method, independent of the solver: non-negativity; demand = delivered + unmet; flow conservation per DC and product; plant capacity and qualification; no flow through closed DCs; DC throughput and cold capacity; single-sourcing and max-open rules when enabled; and the objective recomputed from flows must match the solver's. Any failure raises.

**Benchmark across instances** (`pharmanet benchmark`). One instance can be lucky or unlucky for the heuristic, so we compare on 10 independently generated networks (seeds 1–10, design demand = last 12 months of history):

| | Savings vs greedy |
|---|---|
| Mean / median | **16.6% / 16.4%** |
| Range | 10.2% – 24.1% |
| Main source | DC choice (greedy opens 1–2 more DCs than needed in most instances) |

On the seed-42 base instance, greedy happens to choose the optimal DCs, so the saving is only 3.2% (all routing). That is the low end of the distribution, which is why the multi-instance figure is the one to quote.

## Simulation and scenario analysis

Code: `src/pharmanet/simulation/` (`scenarios.py`, `evaluate.py`). Run with `pharmanet simulate`
(needs `generate`, `forecast`, `optimize` first); outputs go to `outputs/<run_id>/simulate/`.

**Principle: fix the strategic decision, re-optimize the operational one.** DCs cannot be opened
overnight, but product can be re-routed. Every scenario fixes $y$ to the base design and re-solves
only the flows. With all binaries fixed the MILP is an LP (≈0.1 s per scenario). Scenarios run in
parallel, and every scenario solution goes through the same independent validator.

Scenarios are evaluated at **forecast** demand (or draws around it), not design demand. Safety
stock is the buffer the design reserved capacity for; it is not demand.

### Monte Carlo demand
For each (region, product), annual demand is drawn around the forecast $F$:

$$
D = F\, e^{sZ - s^2/2},\quad s^2 = \ln(1+cv^2),\quad cv = \frac{\sigma_{monthly}\sqrt{12}}{F},\quad
Z_{rk} = \sqrt{\rho}\,G_k + \sqrt{1-\rho}\,E_{rk}
$$

- Lognormal with mean $F$: positive and unbiased.
- $\sigma_{monthly}$ is Phase 3's one-step RMSE. The $\sqrt{12}$ treats monthly errors as independent. This is **optimistic**: persistent errors would widen the spread.
- $G_k$ is a product-level shock shared by all regions, with $\rho$ = `demand_common_shock_share` (0.5). Independent regions would diversify away aggregate risk and understate it.

### Disruptions (one at a time, at forecast demand)
- **Plant outage:** the plant loses `outage_capacity_loss` (default 100%) of its annual capacity. A single-period model cannot represent "down for March", so the default is a whole-year worst case.
- **DC outage:** each *open* DC handles nothing. Its fixed cost is still paid, and no new DC can be opened.
- **Cold-lane failure:** each lane carrying cold product in the base design loses cold transport.

Impact is reported as **unmet units and service level**. Cost change is reported as logistics cost
per *delivered* unit. Total cost would be dominated by the artificial unmet-demand penalty, and total
logistics cost *falls* when less is shipped, so neither is a fair headline.

### Sensitivity analysis
One-at-a-time multipliers on fixed costs, transport costs and demand (0.8–1.2). For each, we
**re-optimize the design** (full MILP) and also **price the base design** under the new parameters.
Regret = cost(base design) − cost(re-optimized design) ≥ 0.

### Results (base instance, synthetic, seed 42)

| Monte Carlo (500 scenarios) | |
|---|---|
| Cost p5 / p50 / p95 | $4.07M / $4.13M / $4.21M (±1.8% around $4.13M at forecast) |
| Mean service level / worst scenario | 100.0% / 99.2% |
| P(any shortfall) | 0.2% |

| Worst single failures | Unmet units | Service level | Why |
|---|---|---|---|
| DC02 down | 362k | 71.5% | the remaining open DCs lack capacity (only 8.2% headroom) |
| PL03 down | 324k | 74.4% | SKU01 and SKU06 are made *only* at PL03; SKU02/SKU07 lack backup capacity |
| PL02 down | 272k | 78.5% | all of it cold product |
| Cold-lane failures | 29 of 30 rerouted | ≥ 99.98% | cold capacity at other open DCs absorbs them |

Every single plant or DC outage causes a shortfall (8 of 8). Cost per delivered unit rises 9–14%.

| Sensitivity | Result |
|---|---|
| Fixed or transport cost ±20% | design unchanged, zero regret |
| Demand −20% | best design closes DC01; keeping it costs 12.7% more |
| Demand +10% / +20% | best design swaps in the large DC05; the base design serves only 96% / 88% |

**Takeaway.** The cost-optimal design is robust to cost assumptions and ordinary forecast error,
but fragile to capacity shocks. Cost minimization opened just enough capacity (8% headroom), so
any single-node failure or +10% demand causes shortages. That is the trade-off between efficiency
and resilience. Phase 6 measures it with graph analysis and evaluates mitigations.

## Graph risk analysis

Code: `src/pharmanet/graph/` (`build.py`, `risk.py`) and `src/pharmanet/simulation/mitigation.py`.
Run with `pharmanet graph` (needs `simulate` first); outputs go to `outputs/<run_id>/graph/`,
including `flow_graph.graphml` for Gephi or a Neo4j import.

The graph is built from the design's optimal flows at **forecast** demand, the same basis as the
Phase 5 disruptions, so the numbers are directly comparable.

### Two graphs, two biases
| Graph | Edges | Measures | Bias |
|---|---|---|---|
| Flow graph | lanes actually used, weighted by flow | **exposure**: demand currently passing through a node | upper bound (no re-routing) |
| Capability graph (per product) | every lane that *could* carry it: qualified plants → open DCs (cold-capable for cold) → all regions | **structural loss**: demand with *no* path left | lower bound (ignores capacity) |

- **Exposure** uses **proportional flow tracing**: a plant supplying share $s$ of DC $d$'s inflow of $k$ is exposed for $s \cdot z_{drk}$ in every region $r$.
- **Structural single points of failure** use **dominators**. With a super-source feeding all qualified plants, node $v$ dominates region $r$ for product $k$ if every path from the source to $r$ passes through $v$. We use NetworkX's `immediate_dominators` (Cooper-Harvey-Kennedy) and walk each region's dominator chain.
- **Betweenness centrality** is reported, but in a 3-layer network it is non-zero only for DCs and ignores volume.

### Cross-check with simulation
For every full plant or DC outage: **structural ≤ simulated (Phase 5 LP) ≤ exposure**.
- The lower bound is exact: no path means no flow.
- The upper bound holds whenever the unmet-demand penalty outweighs re-routing costs. That is true here, but it is not a theorem.

This checks three independently built modules against each other. **Base instance: both bounds hold for all 8 nodes.**

| Node | Structural | **Simulated** | Exposure | Betweenness |
|---|---|---|---|---|
| DC02 | 0 | **362k (28.5%)** | 465k | 0.026 |
| PL03 | 169k | **324k** | 324k | 0 |
| PL02 | 103k | **272k** | 367k | 0 |
| DC07 | 0 | **252k** | 355k | 0.041 (highest) |

Readings:
- PL03's loss equals its exposure: nothing it supplies can be re-routed.
- DC02 has zero structural loss (other DCs could reach every region) but loses 28.5% through *capacity*. A capacity-blind graph check would call it safe.
- Betweenness ranks DC07 first, yet DC02 is the worst failure. Connectivity is not criticality when volume matters.

Structural single points of failure found: PL03 → SKU01, SKU06; PL02 → SKU08 (cold); PL04 → SKU12. These are exactly the four single-qualified products.

### Mitigations: the price of resilience
Each option is costed at design demand and stress-tested against every single plant or DC outage at forecast demand.

| Mitigation | Open DCs | Cost premium | Worst single failure (demand at risk) | Failures with shortfall | Mean demand at risk |
|---|---|---|---|---|---|
| Base (cost-optimal) | 4 | — | DC02: **28.5%** | 8 / 8 | 19.6% |
| N-1 DC design (C8/C9) | 5 (+DC05) | +15.1% | PL03: 25.6% | 4 / 8 | 9.0% |
| Dual-qualify the 4 single-source products | 4 | −3.3%* | DC02: 28.5% | 8 / 8 | 16.0% |
| **Both** | 5 | **+12.5%** | PL01: **15.5%** | 4 / 8 | **5.8%** |

\*Network cost only. More sourcing options make flows cheaper, but the one-off cost of tech transfer
and regulatory approval is **not modelled**. In reality it is substantial.

**Takeaways.** Combining both cuts worst-case demand at risk by 46% (28.5% → 15.5%) and mean demand
at risk by 70% (19.6% → 5.8%) for a 12.5% network cost premium. Neither alone is enough: N-1 fixes
DC failures but not single-source plants, and dual qualification fixes single-source plants but not
DC capacity. The remaining risk (PL01 → 15.5%) comes from *plant capacity* on multi-source products,
not from structure. Closing it would need backup capacity, a third lever not modelled here.


## Two-stage stochastic design (Phase 8)

Code: `src/pharmanet/optimization/stochastic.py`. Math: [formulation.md](formulation.md#two-stage-stochastic-network-design-phase-8). Run with `pharmanet stochastic`.

- **Scenarios.** 30 demand draws (Phase 5 sampler, own random stream) share probability 0.9. Every single plant and candidate-DC outage (12) shares 0.1 (`stochastic.disruption_weight`). Scenario demand is the forecast with **no safety stock**, because the model hedges explicitly.
- **Solve.** Extensive form: 106,856 variables (8 binary), 97,440 constraints, proven optimal by HiGHS in 100–150 s (varies with machine load).
- **Evaluation.** In-sample RP/EEV/VSS and WS/EVPI, then both designs priced on **212 fresh scenarios** from a different random stream.

**Results (base instance, synthetic)**

| | Deterministic | Stochastic |
|---|---|---|
| Open DCs | DC01, DC02, DC07, DC08 | DC02, **DC05**, DC07, DC08 |
| Cost at design demand | $4.19M | $4.36M (+4.1%) |
| Out-of-sample expected cost | $4.954M | **$4.935M** (VSS 0.39%) |
| Mean / worst service under single outages | 86.9% / 71.5% | **90.3% / 74.4%** |
| In-sample VSS / EVPI | | 0.38% / 7.7% |

**Reading it honestly.**
1. Out-of-sample VSS ≈ in-sample VSS, so the 42-scenario design is not overfit.
2. **With demand uncertainty only (`disruption_weight = 0`), the stochastic model picks exactly the deterministic design: VSS = 0.** Forecast + safety stock already hedges normal demand variation optimally. All of the VSS comes from outage scenarios.
3. That value depends on outage probability × the unmet-demand penalty, both assumptions. The stochastic design is a hedge: it pays 4.1% more in a normal year to lose less in an outage year.
4. EVPI (7.7%) ≫ VSS (0.4%). Knowing *which* node fails would be worth far more than modelling that some might. This is an argument for monitoring and early warning, not just a better design.

## Neo4j export (Phase 8)

Code: `src/pharmanet/graph/neo4j_export.py`. Run with `pharmanet neo4j-export`. It writes `outputs/<run>/neo4j/` with node and relationship CSVs, `import.cypher` (constraints + `LOAD CSV`), `queries.cypher`, a README, and `expected_answers/`.

Graph model: `(:Plant)-[:SHIPS]->(:DC)-[:SHIPS]->(:Region)` with `product_id`, `flow`, `is_cold` on each `SHIPS`; `(:Plant)-[:QUALIFIED_FOR]->(:Product)`; `(:Region)-[:DEMANDS]->(:Product)`.

| Query | Business question | Base answer |
|---|---|---|
| `single_plant_cold_regions` | Which regions depend on a single plant for a cold product? | 72 of 80 cold (region, product) pairs, *in current flows* |
| `single_qualified_products` | Which products have only one approved plant? | SKU01, SKU06 (PL03), SKU08 (PL02, cold), SKU12 (PL04) |
| `dc_exposure` | How much demand flows through each DC? | DC02 465k units (36.7%) … |
| `single_dc_regions` | Which regions are served by only one DC? | RG07 (DC08) |
| `plant_failure_impact` | If plant `$plant` fails, what does each region lose? | proportional flow tracing (matches Phase 6) |

Every query has a pandas twin (`answer_queries`). Tests check the twins against hand-derived answers and, for `plant_failure_impact`, against Phase 6's `exposure_detail`. Load the bundle into Neo4j 5 with `cypher-shell -f import.cypher` (see the bundle README). Against a live database, compare with `expected_answers/`. *Not yet run against a live Neo4j server here.*

**Structural vs operational dependence.** Only 4 products have a single *approved* plant (query 2), but 72 of 80 cold region-product pairs are *currently* supplied by just one plant (query 1). Cost optimization concentrates each product on its cheapest source, so dependence in practice is much higher than dependence on paper. Re-routing is possible (Phase 5), but only within spare capacity.

## Results viewer (Phase 8)

`pharmanet view` opens a Streamlit app (`uv sync --extra viewer`). It reads a finished run's files and computes nothing. It has six tabs: overview (resume numbers + provenance), forecasting (WAPE by model, series explorer), network map (edge width = flow), simulation (cost distribution, worst failures, sensitivity), risk (graph bounds vs simulated loss, mitigations), and stochastic design.

Charts follow the project's data-viz rules:
- a fixed, colorblind-validated 3-color order with separate light/dark steps;
- one axis per chart;
- thin marks and selective labels;
- a tooltip on every mark and a table view under every chart.

A Streamlit `AppTest` runs the whole app against a real run in CI.
