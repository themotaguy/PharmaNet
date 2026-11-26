# MILP Formulation: Network Design

> **Status: implemented** in `src/pharmanet/optimization/network_design.py`. The code matches this document one-to-one: each constraint has the same name in code (`demand`, `flow_balance`, `plant_capacity`, `dc_throughput`, `dc_cold`, `strong_link`, `max_open_dcs`, `single_source_one_dc`, `single_source_open`, `single_source_link`). Review notes 1–4 were approved as proposed.

All data is synthetic. Planning horizon: **one year** (single period). Units: product flows in units/year, costs in $/year.

## What the model decides

Choose **which candidate DCs to open** and **how much of each product flows** plant → DC → region, to minimise total annual cost:
fixed DC costs + inbound transport + outbound transport + penalty for any demand left unmet.

The core trade-off: opening more DCs costs more in fixed costs but puts stock closer to regions, which cuts transport. Opening fewer saves fixed cost but ships further.

## Sets

| Symbol | Meaning | Code / data source |
|---|---|---|
| $P$ | plants | `plants.plant_id` |
| $D$ | candidate DCs | `dcs.dc_id` |
| $R$ | demand regions | `regions.region_id` |
| $K$ | products | `products.product_id` |
| $K^c \subseteq K$ | cold-chain products | `products.is_cold` |

## Parameters

| Symbol | Meaning | Source |
|---|---|---|
| $f_d$ | annual fixed cost of operating DC $d$ ($) | `dcs.fixed_cost` |
| $c^{in}_{pdk}$ | transport cost plant $p$ → DC $d$, per unit of $k$ ($) | `lanes_inbound.cost_per_unit`, or `cold_cost_per_unit` if $k \in K^c$ |
| $c^{out}_{drk}$ | transport cost DC $d$ → region $r$, per unit of $k$ ($) | `lanes_outbound` (same rule) |
| $dem_{rk}$ | design demand (units/yr) = annual forecast + safety stock | Phase 3 `safety_stock.design_demand` |
| $cap_{pk}$ | annual capacity of plant $p$ for product $k$ | `plant_products.capacity` |
| $q_{pk} \in \{0,1\}$ | 1 if plant $p$ is qualified (approved) for $k$ | `plant_products.qualified` |
| $cap_d$ | annual throughput capacity of DC $d$ | `dcs.throughput_capacity` |
| $cold_d$ | annual cold throughput capacity of DC $d$ (0 if none) | `dcs.cold_capacity` |
| $pen_k$ | penalty per unit of unmet demand ($) | `products.unmet_penalty` |
| $N^{max}$ | optional max number of open DCs | `optimization.max_open_dcs` |

## Decision variables

| Symbol | Domain | Meaning |
|---|---|---|
| $y_d$ | $\{0,1\}$ | 1 if DC $d$ is opened |
| $x_{pdk}$ | $\geq 0$ | units of $k$ shipped plant $p$ → DC $d$ per year |
| $z_{drk}$ | $\geq 0$ | units of $k$ shipped DC $d$ → region $r$ per year |
| $u_{rk}$ | $\geq 0$ | units of demand for $k$ in $r$ left unmet |
| $a_{dr}$ | $\{0,1\}$ | *(only with single-sourcing)* 1 if region $r$ is assigned to DC $d$ |

## Objective

$$
\min \;\; \underbrace{\sum_{d} f_d\, y_d}_{\text{fixed}}
\;+\; \underbrace{\sum_{p,d,k} c^{in}_{pdk}\, x_{pdk}}_{\text{inbound}}
\;+\; \underbrace{\sum_{d,r,k} c^{out}_{drk}\, z_{drk}}_{\text{outbound}}
\;+\; \underbrace{\sum_{r,k} pen_k\, u_{rk}}_{\text{unmet penalty}}
$$

## Constraints

**(C1) `demand`: every unit of demand is either delivered or recorded as unmet.**
$$\sum_{d} z_{drk} + u_{rk} = dem_{rk} \qquad \forall r \in R,\; k \in K$$

**(C2) `flow_balance`: what leaves a DC must have arrived from plants (no stock is created at a DC).**
$$\sum_{p} x_{pdk} = \sum_{r} z_{drk} \qquad \forall d \in D,\; k \in K$$

**(C3) `plant_capacity`: plants ship only products they are approved for, within capacity.**
$$\sum_{d} x_{pdk} \leq cap_{pk}\, q_{pk} \qquad \forall p \in P,\; k \in K$$

**(C4) `dc_throughput`: total flow through a DC is within its capacity, and zero if it is closed.**
$$\sum_{k}\sum_{r} z_{drk} \leq cap_d\, y_d \qquad \forall d \in D$$

**(C5) `dc_cold`: cold products need cold capacity, and none flows if the DC is closed.**
$$\sum_{k \in K^c}\sum_{r} z_{drk} \leq cold_d\, y_d \qquad \forall d \in D$$

**(C6) `strong_link` *(proposed addition, see Review note 1)*: a closed DC cannot serve any lane.**
$$z_{drk} \leq dem_{rk}\, y_d \qquad \forall d \in D,\; r \in R,\; k \in K$$

**(C7) `max_open_dcs` *(optional, if `max_open_dcs` is set)*.**
$$\sum_{d} y_d \leq N^{max}$$

**Single-sourcing (optional, `single_sourcing: true`)**: each region is served by exactly one DC.

$$
\textbf{(S1) } \sum_{d} a_{dr} = 1 \;\; \forall r \qquad
\textbf{(S2) } a_{dr} \leq y_d \;\; \forall d, r \qquad
\textbf{(S3) } z_{drk} \leq dem_{rk}\, a_{dr} \;\; \forall d, r, k
$$

(S3) replaces (C6) when single-sourcing is on, because it implies it via (S2).

**Domains**: $y_d, a_{dr} \in \{0,1\}$; $x, z, u \geq 0$.

## Instance size (base config: 4 plants, 8 DCs, 20 regions, 12 products)

| | Count |
|---|---|
| Binary $y$ | 8 |
| Continuous $x$ / $z$ / $u$ | 384 / 1,920 / 240 |
| C1 / C2 / C3 / C4 / C5 / C6 | 240 / 96 / 48 / 8 / 8 / 1,920 |
| + single-sourcing ($a$; S1/S2/S3) | +160 binaries; +20 / +160 / replaces C6 |

## LP relaxation and the MIP gap (reported on every solve)

- **LP relaxation**: the same model with $y_d \in [0,1]$ instead of $\{0,1\}$. It is a pure LP, solves instantly, and because it has *more* freedom, its optimum is a **lower bound** on the true (integer) optimum: $z_{LP} \le z_{MIP}$.
- **Integrality gap** $= (z_{MIP} - z_{LP}) / z_{MIP}$. A small gap means the formulation is *tight*: the LP already "almost knows" the answer, and branch-and-bound has little work to do.
- **MIP gap** (from the solver) = (best solution found − best proven bound) / best solution. Our `mip_gap` setting (0.1%) tells the solver it may stop once it has *proven* it is within 0.1% of optimal.

## Review notes: issues flagged in the original spec formulation

**1. The spec's formulation is valid but *weak*; I propose adding (C6).**
With only (C4), the LP relaxation can "open" a DC fractionally, for example $y_d = 0.3$ if the DC is used at 30% of capacity, and so pays only 30% of its fixed cost. The LP bound then badly underestimates the cost, and the solver has to branch more. (C6) says a DC can only serve region $r$ up to $dem_{rk} \cdot y_d$, which forces $y_d$ to be large whenever a DC serves anyone substantially. It cuts off no integer solution (when $y_d=1$ it is implied by C1; when $y_d=0$ it is implied by C4), so **the optimum is unchanged**, but the LP bound gets much tighter. This is the classic "aggregated vs disaggregated linking constraint" idea from facility location. I'll measure the LP bound with and without it, which makes a good interview talking point.

**2. Design demand adds a stock level to an annual flow.**
$dem_{rk}$ = annual forecast + safety stock. Safety stock is inventory that sits on a shelf, not volume shipped every year, so adding it to flow means "reserve network capacity as if the buffer were shipped once a year." It's a common simplification in single-period strategic models. Here it is small (safety stock ≈ 2.3% of volume), so it barely changes the design. **Proposal: keep it, and document it.** The alternative (separate inventory-holding variables) is really multi-period planning, which is out of scope.

**3. Safety stock uses the fastest-path lead time (optimistic): close the loop after solving.**
**Proposal:** after solving, recompute each $(r,k)$'s lead time from the **actual flows** (a flow-weighted average over the DCs serving $r$ and the plants feeding those DCs), recompute safety stock, and report the change. If `ss_refinement_iterations > 0` (new config field, default 1), re-solve once with the updated $dem_{rk}$ and report whether the design changed.

**4. Single-sourcing granularity.**
The spec says each **region** gets exactly one DC, for **all** products. Consequence: a region with cold demand must be assigned to a cold-capable DC, or its cold demand goes unmet. That's realistic (one delivery point per customer), but stricter than "one DC per region *per product*." **Proposal: keep it per region, as the spec says.**

**5. Things deliberately left out (and why).**
- **DC handling cost per unit**: not in the spec. Easy to add later.
- **`unit_value`, `shelf_life_months`**: not used by a single-period flow model. They would matter for holding cost or expiry in multi-period planning, which is out of scope.
- **Qualification is enforced twice**: by $q_{pk}$ in (C3) and by the data ($cap_{pk}=0$ when unqualified). This is intentional: the model shouldn't depend on data hygiene.

## Solver interface

Pyomo model → `SolverFactory(optimization.solver.name)` (`appsi_highs` by default; `gurobi` or `cbc` when installed and configured). `time_limit_s`, `mip_gap` and `threads` are passed through. Each solve reports: status, objective, its breakdown by cost term, MIP gap, wall time, number of variables and constraints (binary/continuous), and the LP relaxation bound with the integrality gap.

## Regression instance (hand-verified)

Used by `tests/test_optimization.py`. Two products, **A** (ambient) and **C** (cold). Only **P1** is qualified for C, and **D3** has no cold storage. Capacities (1,000 everywhere) never bind, so for any set of open DCs each unit simply takes its cheapest feasible path. That makes it possible to enumerate all $2^3 - 1 = 7$ designs by hand.

| | |
|---|---|
| Inbound cost $c^{in}_{pd}$ (ambient) | P1: D1=1, D2=2, D3=3 · P2: D1=3, D2=2, D3=1 |
| Outbound cost $c^{out}_{dr}$ (ambient) | D1: R1=1, R2=3, R3=5 · D2: R1=3, R2=1, R3=3 · D3: R1=5, R2=3, R3=1 |
| Cold premium | ×2 on both legs |
| Fixed cost $f_d$ | D1=500, D2=350, D3=400 |
| Demand | A: 100 per region · C: 50 per region |
| Penalty | 100 per unit |

**Cheapest path cost per unit**, by DC used (inbound from the cheapest qualified plant + outbound):

| via | A→R1 | A→R2 | A→R3 | C→R1 | C→R2 | C→R3 |
|---|---|---|---|---|---|---|
| D1 (A from P1: 1) | 2 | 4 | 6 | (1+1)·2 = 4 | (1+3)·2 = 8 | (1+5)·2 = 12 |
| D2 (from P1 or P2: 2) | 5 | 3 | 5 | (2+3)·2 = 10 | (2+1)·2 = 6 | (2+3)·2 = 10 |
| D3 (A from P2: 1) | 6 | 4 | 2 | not allowed | not allowed | not allowed |

**All designs.** Transport = Σ over (r,k) of demand × the cheapest path among the open DCs:

| Open DCs | Fixed | A transport | C transport | Penalty | **Total** |
|---|---|---|---|---|---|
| {D1} | 500 | 100·(2+4+6) = 1200 | 50·(4+8+12) = 1200 | 0 | 2900 |
| {D2} | 350 | 100·(5+3+5) = 1300 | 50·(10+6+10) = 1300 | 0 | 2950 |
| {D3} | 400 | 100·(6+4+2) = 1200 | none | 150·100 = 15000 | 16600 |
| **{D1, D2}** | 850 | 100·(2+3+5) = 1000 | 50·(4+6+10) = 1000 | 0 | **2850** ← optimum |
| {D1, D3} | 900 | 100·(2+4+2) = 800 | 50·(4+8+12) = 1200 | 0 | 2900 |
| {D2, D3} | 750 | 100·(5+3+2) = 1000 | 50·(10+6+10) = 1300 | 0 | 3050 |
| {D1, D2, D3} | 1250 | 100·(2+3+2) = 700 | 50·(4+6+10) = 1000 | 0 | 2950 |

Notice that opening D3 as well saves 300 in A transport but costs 400 in fixed cost, so the optimum stops at two DCs. That is the fixed-cost vs transport trade-off in miniature.

Tests also check: with `max_open_dcs = 1` the optimum is {D1} = 2900. Single-sourcing gives the same 2850 (R1→D1, R2→D2, R3→D2). Greedy opens only {D2} (cheapest fixed cost, enough capacity alone), costs 2950, and the MILP saves 100/2950 = 3.39%, all of it from DC choice.

## Results on the base instance (synthetic, seed 42)

| | |
|---|---|
| Size | 2,552 variables (8 binary), 2,320 constraints |
| Solve time (HiGHS) | ≈ 1.1 s, proven optimal (MIP gap 0.01%) |
| Optimal design | DC01, DC02, DC07, DC08; 100% of design demand served |
| Total cost | $4.19M = fixed 2.07M + inbound 0.85M + outbound 1.27M |
| LP bound with / without (C6) | 4.01M / 3.94M → integrality gap **4.2% / 6.0%** |
| Safety-stock refinement | actual-path lead times raise SS by 4.9% (29.3k → 30.8k units); design unchanged; cost +0.07% |

The strong-link constraint cut the integrality gap by about 30%. A gap remains because the LP can still split a DC's fixed cost across many partially served lanes, and capacity (not just demand) limits each DC.
