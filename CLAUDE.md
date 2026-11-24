# CLAUDE.md — PharmaNet: Supply Network Optimization & Resilience

## Who I am and why this project exists

I'm Ritik, an MS Data Science student at Northeastern with 3 years of data engineering / ML experience at ZS Associates (pharma and life sciences clients). I'm building this as a portfolio project to qualify for Data Science / Operations Research roles in supply chain. The target job asks for:

- Formulating and implementing optimization models (LP/MILP)
- Statistical/ML, time-series, simulation, and graph/network-based analytics
- Rigorous validation: benchmarks, sensitivity/scenario analysis, regression testing, output validation
- Modular, testable, production-grade Python (not notebook prototypes)
- Databricks, Spark, Delta Lake, Unity Catalog
- Graph/network modeling of supply networks and risk propagation (Neo4j/Cypher a plus)

Every module in this project maps to one of those requirements.

## How I want you to work with me (important)

I'm using this project to **learn** OR and supply chain modeling, and I'll have to defend every line of it in interviews. So:

1. **Explain before you implement.** Before writing each module, give me a short plain-English explanation of the method (e.g., what the MILP is deciding, why a constraint exists, why we use rolling-origin backtesting). Keep it brief, then build.
2. **Write the math down.** Every optimization model gets a formulation in `docs/formulation.md` (sets, parameters, variables, objective, constraints, in LaTeX) that matches the code one-to-one.
3. **Work in phases.** Complete one phase at a time (see Milestones). At the end of each phase, stop, summarize what was built, show me how to run it, and wait for me before moving on.
4. **Quiz me occasionally.** After each phase, ask me 2–3 short questions I'd likely get in an interview about that phase (e.g., "What happens to the solution if we remove the integrality constraint on y?"). Don't give the answer unless I ask.
5. **No silent magic.** Don't add dependencies, change the formulation, or restructure the repo without telling me why.
6. **Keep it honest.** All data is synthetic. Never present synthetic results as real-world results in the README or docs.

## Project summary

A pharma distribution network: manufacturing plants → distribution centers (DCs) → demand regions, for a set of products, some of which require cold chain. The system:

1. Generates realistic synthetic data (seeded, reproducible)
2. Forecasts demand per product-region and picks the best model by backtest
3. Computes safety stock from forecast error and lead times
4. Solves a MILP network design: which DCs to open and how product flows, at minimum cost
5. Stress-tests the design with Monte Carlo demand and disruption scenarios
6. Models the network as a graph to find single points of failure and how disruptions propagate
7. Runs end-to-end from a CLI, locally, with an optional Databricks/Delta deployment layer

## Tech stack

- Python 3.11, managed with `uv` (or `poetry` if `uv` causes issues)
- Optimization: **Pyomo** with the **HiGHS** solver (`highspy`), open-source and license-free. Structure the code so Gurobi could be swapped in via config.
- Forecasting: `statsmodels` (ETS, SARIMA) plus a seasonal-naive baseline
- Simulation: NumPy Monte Carlo; `simpy` only if we add the discrete-event stretch goal
- Graph: `networkx`; optional Neo4j export with Cypher queries
- Data: `pandas`, `pyarrow`
- Config and validation: `pydantic` + YAML
- CLI: `typer`
- Quality: `pytest`, `ruff`, `mypy`, `pre-commit`, GitHub Actions CI
- Optional: PySpark + Delta Lake for the Databricks layer; Streamlit for a results viewer

## Repository structure

```
pharmanet/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml            # network size, seeds, cost params, solver settings
│   └── small_test.yaml      # tiny instance for regression tests
├── docs/
│   ├── formulation.md       # LaTeX MILP formulation, matches code exactly
│   ├── methodology.md       # forecasting, safety stock, simulation, graph methods
│   └── results.md           # generated summary of the latest run
├── src/pharmanet/
│   ├── config.py            # pydantic models for config
│   ├── data/generate.py     # synthetic data generator
│   ├── forecasting/
│   │   ├── models.py        # seasonal naive, ETS, SARIMA wrappers
│   │   └── backtest.py      # rolling-origin backtesting, WAPE/MAPE/bias
│   ├── inventory/safety_stock.py
│   ├── optimization/
│   │   ├── network_design.py  # Pyomo MILP
│   │   ├── baseline.py        # greedy heuristic for benchmarking
│   │   └── validate.py        # flow conservation and constraint checks on solutions
│   ├── simulation/
│   │   ├── scenarios.py       # demand + disruption scenario generation
│   │   └── evaluate.py        # evaluate fixed design under scenarios
│   ├── graph/
│   │   ├── build.py           # NetworkX graph from solution
│   │   ├── risk.py            # disruption propagation, centrality, SPOFs
│   │   └── neo4j_export.py    # optional
│   ├── pipeline.py            # end-to-end orchestration
│   └── cli.py
├── databricks/              # optional Phase 8
│   └── jobs/
└── tests/
```

## Synthetic data spec

Seeded and configurable. Default size: 4 plants, 8 candidate DCs, 20 demand regions, 12 products, 36 months of demand history.

- **Products**: ID, cold-chain flag (~30% cold), unit value, shelf life in months
- **Plants**: location (x, y coordinates are fine), production capacity per product, and a **qualification matrix** (not every plant is approved to make every product, mirroring regulatory site approval in pharma)
- **DCs**: location, fixed annual opening cost, throughput capacity, cold-chain capacity (some DCs have none)
- **Regions**: location, monthly demand history per product with trend, seasonality, and noise; a few products with intermittent demand
- **Costs**: transport cost per unit-distance, with a cold-chain premium; unmet-demand penalty well above any transport cost
- **Lead times**: plant→DC and DC→region, with variability

Write a short data dictionary into `docs/methodology.md`.

## Module specs

### Forecasting
- Models: seasonal naive (baseline), ETS, SARIMA
- Rolling-origin backtest over the last 12 months
- Metrics: WAPE (primary), MAPE, bias
- Select the best model per product-region; report how much it improves over seasonal naive
- Output: 12-month forecast plus forecast error estimates (needed for safety stock)

### Safety stock
- Standard formula using forecast error and lead-time variability at a target service level (e.g., 95%, configurable)
- Explain the formula in `docs/methodology.md`

### Network design MILP (core of the project)
Starting formulation, single period (annual). Review it with me before coding and flag anything that looks wrong.

**Sets**: plants P, candidate DCs D, regions R, products K (cold products K_c ⊆ K)

**Parameters**: fixed DC cost f_d; transport costs c_pdk (plant→DC), c_drk (DC→region); forecast demand plus safety stock dem_rk; plant capacity cap_pk; qualification q_pk ∈ {0,1}; DC throughput cap_d; DC cold capacity cold_d; unmet-demand penalty pen_k

**Variables**: y_d ∈ {0,1} open DC; x_pdk ≥ 0 plant→DC flow; z_drk ≥ 0 DC→region flow; u_rk ≥ 0 unmet demand

**Objective**: minimize fixed costs + inbound transport + outbound transport + unmet-demand penalty

**Constraints**:
- Demand: Σ_d z_drk + u_rk = dem_rk for all r, k
- DC flow balance: Σ_p x_pdk = Σ_r z_drk for all d, k
- Plant capacity and qualification: Σ_d x_pdk ≤ cap_pk · q_pk for all p, k
- DC throughput, only if open: Σ_k Σ_r z_drk ≤ cap_d · y_d for all d
- Cold-chain capacity: Σ_{k∈K_c} Σ_r z_drk ≤ cold_d · y_d for all d
- Optional, behind a config flag: single-sourcing (each region served by exactly one DC, using binary assignment variables) and a maximum number of open DCs

Also report: solve time, MIP gap, number of variables and constraints, and the **LP relaxation bound** versus the integer solution (explain to me what that gap means).

### Baseline heuristic
A simple greedy approach (e.g., open the cheapest DCs until capacity covers demand, then assign regions to the nearest open DC). Used to benchmark how much the MILP saves. This comparison becomes a headline result.

### Solution validation
Independent checks on every solution, not relying on the solver: flow conservation, no flow through closed DCs, capacity and qualification respected, demand met or explicitly recorded as unmet. Fail loudly if any check breaks.

### Simulation and scenario analysis
- Monte Carlo demand scenarios sampled from forecast error distributions (e.g., 500 scenarios, configurable)
- Disruption scenarios: a plant goes down, a DC goes down, a cold-chain lane fails
- For each scenario, fix the DC decisions (y) from the base design and re-solve only the flows as an LP. Report cost distribution, service level, and unmet demand.
- Sensitivity analysis: vary key parameters (fixed costs, transport costs, demand level ±20%) and show how the design changes
- Stretch goal: two-stage stochastic version that chooses DCs while accounting for scenarios; compare against the deterministic design

### Graph and risk propagation
- Build a directed graph from the solution: plants → DCs → regions, edges weighted by flow
- Metrics: betweenness centrality, single points of failure (nodes whose removal strands demand), % of total demand at risk per node
- Propagation: when a node fails, which regions and products are affected and by how much
- Cross-check graph findings against the disruption scenarios from the simulation module
- Optional: export to Neo4j and write 3–5 Cypher queries answering business questions (e.g., "which regions depend on a single plant for cold-chain products?")

### Pipeline and CLI
- `pharmanet generate`, `pharmanet forecast`, `pharmanet optimize`, `pharmanet simulate`, `pharmanet graph`, `pharmanet run-all`
- Everything driven by config; outputs written to `outputs/<run_id>/` as parquet plus a `results.md` summary

### Databricks layer (optional, Phase 8)
- Bronze/silver/gold Delta tables: raw synthetic data → cleaned inputs → solution and scenario outputs
- Unity Catalog-style three-level naming (`catalog.schema.table`)
- A Databricks job that calls the same `pharmanet` package; no duplicated logic in notebooks
- Core project must still run fully locally without Databricks

## Testing requirements

- Unit tests for each module
- **Regression test on a tiny hand-verifiable instance** (2 plants, 3 DCs, 3 regions, 2 products) where the optimal solution is known; the test asserts the exact objective value and open DCs. Help me verify it by hand.
- Property-style tests: adding demand never decreases cost; removing a DC never decreases cost
- Validation module runs on every solve
- CI runs ruff, mypy, and pytest on every push

## Coding standards

- Type hints everywhere; mypy must pass
- Small, pure functions where possible; no business logic in the CLI
- Docstrings that explain the *why*, not just the *what*
- No hardcoded numbers; everything in config
- Fixed random seeds for reproducibility
- Notebooks only for exploration, never as the source of truth

## Milestones (stop after each and wait for me)

1. **Setup**: repo skeleton, tooling, CI, config system
2. **Data**: synthetic generator plus data dictionary
3. **Forecasting**: models, backtest, model selection, safety stock
4. **Optimization**: formulation doc first, then MILP, baseline heuristic, validation, regression test
5. **Simulation**: scenarios, re-solve evaluation, sensitivity analysis
6. **Graph**: network graph, SPOFs, propagation, cross-check with simulation
7. **Pipeline**: CLI, end-to-end run, results summary, README
8. **Optional**: Databricks/Delta layer, Neo4j, Streamlit viewer, two-stage stochastic model

## Results to capture for my resume

Make sure the pipeline reports these numbers so I can quote them accurately:

- Cost savings of the MILP versus the greedy baseline (%)
- Forecast WAPE improvement over seasonal naive (%)
- Service level and cost range across Monte Carlo scenarios
- Demand at risk from the worst single-node failure, and how much the design or a mitigation reduces it
- Instance size (variables, constraints) and solve time

## Out of scope

- Real company data of any kind
- Multi-period inventory planning (possible future extension, not now)
- Production scheduling at the plant level
- Fancy UI work before the core analytics are done
