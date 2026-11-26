# PharmaNet results: `base`

> **All data is synthetic.** Every number below comes from a seeded random network built to resemble a pharma distribution network. None of it describes a real company, product, or network.

## Run

| Field | Value |
|---|---|
| Config | `configs/base.yaml` (sha256 `76521ae513e9`) |
| Seed | 42 |
| Code | pharmanet 0.1.0, git `747353a-dirty` |
| Finished | 2026-11-26T03:01:52+00:00 |
| Stage timings (s) | generate 0.1, forecast 84.2, optimize 2.9, simulate 39.4, graph 19.1, benchmark 13.9 |

## Instance

4 plants, 8 candidate DCs, 20 regions, 12 products.

## 1. Forecasting

| Measure | Value |
|---|---|
| Portfolio WAPE by model (12-month backtest) | ets 0.222, sarima 0.235, seasonal_naive 0.239 |
| Selection strategy (config) | per_series |
| Holdout WAPE, last 6 months: naive / per-series / global | 0.235 / 0.218 / 0.214 (ets) |
| Holdout improvement vs naive: per-series / global | 7.2% / 9.2% |
| In-sample improvement (optimistic, selection bias) | 13.6% |
| Fit fallbacks to seasonal naive | 0 |

## 2. Safety stock

Service level 95%: safety stock 29k units = 2.31% of annual forecast (1,268k units).
Recomputed with the chosen network's lead times: 29k → 31k units; design changed: False.

## 3. Network design (MILP)

| Measure | Value |
|---|---|
| Size | 2,552 vars (8 binary), 2,320 constraints |
| Solve | optimal, 1.07 s, MIP gap 0.013% |
| LP bound / integrality gap | $4.01M / 4.20% (without C6: 6.01%) |
| Open DCs | DC01, DC02, DC07, DC08 |
| Annual cost | $4.19M = fixed $2.07M + inbound $852,699 + outbound $1.27M + penalty $0 |
| Service level (design demand) | 100.00% |

## 4. MILP vs greedy baseline

This network: greedy $4.32M vs MILP $4.19M → **3.17%** (DC choice 0.00%, routing 3.17%); greedy service 100.00%.

Across 10 generated networks (seeds 1–10): mean **16.6%**, median 16.4%, range 10.2%–24.1%; mean saving from DC choice 11.4%, from routing 5.2%.

## 5. Simulation (design fixed, flows re-optimized)

| Measure | Value |
|---|---|
| Monte Carlo scenarios | 500 |
| Cost p5 / p50 / p95 | $4.07M / $4.13M / $4.21M |
| Service level mean / p5 / worst | 100.00% / 100.00% / 99.17% |
| P(any shortfall) | 0.2% |
| Open-capacity headroom over forecast | 8.2% |
| Single failures causing shortfall | 9 of 38 |

Worst single failures:

| Scenario | Unmet units | Service level | Cost / delivered unit |
|---|---|---|---|
| dc_DC02 | 362k | 71.5% | +14.4% |
| plant_PL03 | 324k | 74.4% | +13.8% |
| plant_PL02 | 272k | 78.5% | +9.6% |
| dc_DC07 | 252k | 80.1% | +9.1% |
| plant_PL04 | 238k | 81.2% | +11.0% |

Sensitivity (re-designed per case): design changed in demand ×0.8, demand ×1.1, demand ×1.2; lowest service of the kept base design 88.0%.

## 6. Graph risk and mitigation

Structural single points of failure: PL02 → SKU08, PL03 → SKU01, SKU06, PL04 → SKU12.

Cross-check structural ≤ simulated ≤ exposure on 8 nodes: lower bound holds, upper bound holds.

| Node | Structural | Simulated | Exposure | Betweenness |
|---|---|---|---|---|
| DC02 | 0 | 362k | 465k | 0.026 |
| PL03 | 169k | 324k | 324k | 0.000 |
| PL02 | 103k | 272k | 367k | 0.000 |
| DC07 | 0 | 252k | 355k | 0.041 |
| PL04 | 98k | 238k | 260k | 0.000 |
| DC08 | 0 | 210k | 210k | 0.032 |

| Mitigation | Open DCs | Cost premium | Worst failure (at risk) | Mean at risk |
|---|---|---|---|---|
| base | 4 | +0.0% | DC02 (28.5%) | 19.6% |
| n_minus_1 | 5 | +15.1% | PL03 (25.6%) | 9.0% |
| dual_qualification | 4 | -3.3% | DC02 (28.5%) | 16.0% |
| n_minus_1+dual_qualification | 5 | +12.5% | PL01 (15.5%) | 5.8% |

## Resume numbers

Each figure is computed by the pipeline; quote it with its caveat.

1. **MILP cost savings vs greedy heuristic:** mean 16.6% (range 10.2%–24.1%) across 10 generated networks  
   _Source:_ `benchmark/milp_vs_greedy.parquet`. _Caveat:_ Synthetic networks; design demand = last 12 months of history. On this run's network: 3.2%.
2. **Forecast WAPE improvement over seasonal naive:** 7.2% (per_series selection, out-of-sample holdout); global: 9.2%  
   _Source:_ `forecast/summary.json`. _Caveat:_ Last 6 months held out from model choice. In-sample figure (13.6%) is optimistic; do not quote it.
3. **Service level and cost range under demand uncertainty:** mean service 100.00%, worst 99.17%; cost p5–p95 $4.07M–$4.21M over 500 Monte Carlo scenarios  
   _Source:_ `simulate/summary.json`. _Caveat:_ Annual spread = monthly RMSE × √12 (assumes independent monthly errors; likely optimistic).
4. **Demand at risk from worst single-node failure, and mitigation:** 28.5% (DC02 down) → 15.5% with n_minus_1+dual_qualification (−45.8%) for +12.5% network cost  
   _Source:_ `graph/mitigations.parquet`. _Caveat:_ Full-year single-node outages at forecast demand. Dual-qualification (tech transfer) cost is not modelled.
5. **MILP instance size and solve time:** 2,552 variables (8 binary), 2,320 constraints; solved to proven optimality (MIP gap 0.01%) in 1.1 s with HiGHS  
   _Source:_ `optimize/summary.json`. _Caveat:_ Integrality gap 4.2% (vs 6.0% without strong-link constraints).
