# PharmaNet: Supply Network Optimization & Resilience

> **All data in this project is synthetic.** It is generated from a seeded random
> process to resemble a pharma distribution network. No result here describes a
> real company, product, or network.

PharmaNet models a pharma distribution network (plants → distribution centers →
demand regions, with cold-chain products) and:

1. Generates reproducible synthetic data
2. Forecasts demand and selects models by rolling-origin backtest
3. Computes safety stock from forecast error and lead-time variability
4. Solves a MILP network design (which DCs to open, how product flows)
5. Stress-tests the design with Monte Carlo demand and disruption scenarios
6. Analyzes the network as a graph for single points of failure

## Status

| Phase | Scope | Status |
|---|---|---|
| 1 | Setup: skeleton, tooling, CI, config | ✅ |
| 2 | Synthetic data generator | ✅ |
| 3 | Forecasting + safety stock | ✅ |
| 4 | Optimization (MILP, baseline, validation) | ✅ |
| 5 | Simulation + sensitivity | ✅ |
| 6 | Graph risk analysis | ⏳ |
| 7 | Pipeline, CLI, results | ⏳ |
| 8 | Optional: Databricks, Neo4j, Streamlit, stochastic | ⏳ |

## Quickstart

```bash
uv sync                                  # create .venv and install deps (Python 3.11)
uv run pharmanet check-config -c configs/base.yaml
uv run pharmanet generate -c configs/base.yaml   # synthetic data -> outputs/base/data/
uv run pharmanet forecast -c configs/base.yaml   # backtest, forecast, safety stock (~40s)
uv run pharmanet optimize -c configs/base.yaml   # MILP network design + greedy benchmark (~5s)
uv run pharmanet benchmark -c configs/base.yaml  # MILP vs greedy on 10 generated networks
uv run pharmanet simulate -c configs/base.yaml   # Monte Carlo, disruptions, sensitivity (~30s)
uv run pytest                            # tests
uv run ruff check . && uv run mypy       # lint + type check
uv run pre-commit install                # optional: run checks on every commit
```

Data tables and the demand model are documented in [docs/methodology.md](docs/methodology.md#data-dictionary).

## Troubleshooting

**`ModuleNotFoundError: No module named 'pharmanet'` on macOS.** When the project sits in an
iCloud-synced folder (e.g. `~/Documents`), iCloud marks dot-files such as `.venv/**` as hidden,
and recent Python releases skip hidden `.pth` files, which breaks the editable install. The fix
used here keeps the environment in `venv.nosync/` (iCloud ignores `*.nosync`), with `.venv` as a
symlink to it:

```bash
rm -rf .venv venv.nosync && uv venv venv.nosync && ln -s venv.nosync .venv && uv sync
```

Alternatively, move the project outside iCloud-synced folders.
