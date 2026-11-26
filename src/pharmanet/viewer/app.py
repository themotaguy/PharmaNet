"""PharmaNet results viewer. Launch with `pharmanet view` (or `streamlit run` on this file).

Reads ``$PHARMANET_OUTPUT_DIR`` (default ``outputs``) and shows one finished run.
All numbers come from the run's files; the viewer computes nothing new.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from pharmanet.viewer import charts
from pharmanet.viewer.data import RunData, list_runs, load_run, network_frames, series_frame


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> RunData:  # mtime busts the cache when a run is redone
    return load_run(Path(path))


def _table(df: pd.DataFrame, label: str = "Table view") -> None:
    with st.expander(label):
        st.dataframe(df, hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="PharmaNet results", layout="wide")
    out_dir = Path(os.environ.get("PHARMANET_OUTPUT_DIR", "outputs"))
    runs = list_runs(out_dir)
    if not runs:
        st.info(f"No finished runs in `{out_dir}`. Run `pharmanet run-all` first.")
        st.stop()
    choice = st.sidebar.selectbox("Run", [r.name for r in runs])
    path = out_dir / choice
    run = _load(str(path), path.stat().st_mtime)
    mode: charts.Mode = "dark" if st.context.theme.type == "dark" else "light"
    res = run.results

    st.title(f"PharmaNet: `{choice}`")
    st.caption(
        "**All data is synthetic.** A seeded random network built to resemble a pharma "
        "distribution network. Nothing here describes a real company, product, or network."
    )
    tabs = st.tabs(
        [
            "Overview",
            "Forecasting",
            "Network",
            "Simulation",
            "Risk & mitigation",
            "Stochastic design",
        ]
    )

    with tabs[0]:
        for n in run.resume:
            st.markdown(f"**{n['claim']}**  \n{n['value']}")
            st.caption(f"Source `{n['source']}`. {n['caveat']}")
        man = res.get("manifest")
        if man:
            st.caption(
                f"Config `{man.get('config_file')}` · sha256 `{str(man.get('config_sha256'))[:12]}`"
                f" · git `{man.get('git_commit')}` · finished {man.get('finished_at')}"
            )

    with tabs[1]:
        f = res["forecast"]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Holdout gain vs naive (configured)", f"{f['holdout_improvement_vs_naive_pct']:.1f}%"
        )
        c2.metric("Per-series selection", f"{f['holdout_improvement_per_series_pct']:.1f}%")
        c3.metric("Global model", f"{f['holdout_improvement_global_pct']:.1f}%")
        st.altair_chart(charts.wape_by_model(f["portfolio_wape_by_model"], mode), width="stretch")
        st.subheader("Series explorer")
        a, b = st.columns(2)
        region = a.selectbox("Region", sorted(run.demand["region_id"].unique()))
        product = b.selectbox("Product", sorted(run.demand["product_id"].unique()))
        sdf = series_frame(run, region, product)
        sel = run.selection[
            (run.selection["region_id"] == region) & (run.selection["product_id"] == product)
        ]
        if len(sel):
            row = sel.iloc[0]
            st.caption(
                f"Selected model: **{row['best_model']}** · backtest WAPE "
                f"{row['wape']:.3f} (seasonal naive {row['naive_wape']:.3f})"
            )
        st.altair_chart(charts.forecast_series(sdf, mode), width="stretch")
        _table(sdf)

    with tabs[2]:
        o = res["optimize"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open DCs", len(o["open_dcs"]))
        c2.metric("Annual cost", f"${o['costs']['total'] / 1e6:.2f}M")
        c3.metric("Solve time", f"{o['solve_time_s']:.1f} s")
        c4.metric("Integrality gap", f"{100 * o['lp_relaxation']['integrality_gap']:.1f}%")
        nodes, edges = network_frames(run)
        st.altair_chart(charts.network_map(nodes, edges, mode), width="stretch")
        st.caption(
            "Edge width = annual units shipped. Hollow diamonds are candidate DCs "
            "the optimizer did not open."
        )
        _table(edges[["src", "dst", "leg", "flow"]].sort_values("flow", ascending=False))

    with tabs[3]:
        s = res["simulate"]
        mc = s["monte_carlo"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean service (Monte Carlo)", f"{100 * mc['service_level_mean']:.2f}%")
        c2.metric("Worst scenario", f"{100 * mc['service_level_min']:.2f}%")
        c3.metric("P(any shortfall)", f"{100 * mc['prob_any_shortfall']:.1f}%")
        st.altair_chart(
            charts.cost_histogram(run.monte_carlo, s["base_at_forecast"]["total_cost"], mode),
            width="stretch",
        )
        st.subheader("Single-node failures")
        dis = run.disruptions[run.disruptions["unmet"] > 1e-6].nlargest(10, "unmet")
        st.altair_chart(
            charts.ranked_bars(dis, "scenario_id", "unmet", "Unmet units", mode),
            width="stretch",
        )
        _table(dis[["scenario_id", "kind", "unmet", "service_level", "regions_affected"]])
        _table(res["sensitivity"], "Sensitivity analysis")

    with tabs[4]:
        g = res["graph"]
        st.markdown(
            "**Structural single points of failure:** "
            + "; ".join(f"{n} → {', '.join(p)}" for n, p in g["structural_spofs"].items())
            or "none"
        )
        cx = g["cross_check"]
        st.caption(
            f"Cross-check on {cx['compared_nodes']} nodes: lower bound "
            f"{'holds' if cx['lower_bound_holds'] else 'FAILS'}, upper bound "
            f"{'holds' if cx['upper_bound_holds'] else 'FAILS'}."
        )
        st.altair_chart(charts.bounds_chart(res["cross_check"], mode), width="stretch")
        st.caption(
            "Gray bar: graph lower bound (no path left) to upper bound (no re-routing). "
            "Dot: loss simulated by re-optimizing flows."
        )
        _table(
            res["cross_check"][
                ["node", "kind", "structural_loss", "simulated_loss", "exposure", "betweenness"]
            ]
        )
        st.subheader("Mitigations")
        mit = res["mitigations"]
        st.altair_chart(
            charts.ranked_bars(
                mit,
                "mitigation",
                "worst_failure_demand_at_risk_pct",
                "Worst single-failure demand at risk (%)",
                mode,
                ".1f",
            ),
            width="stretch",
        )
        _table(
            mit[
                [
                    "mitigation",
                    "open_dcs",
                    "cost_premium_pct",
                    "worst_failure",
                    "worst_failure_demand_at_risk_pct",
                    "mean_demand_at_risk_pct",
                ]
            ]
        )

    with tabs[5]:
        sp = run.stochastic
        if sp is None:
            st.info("No stochastic results in this run. Run `pharmanet stochastic`.")
            return
        ins, oos = sp["in_sample"], sp["out_of_sample"]
        c1, c2, c3 = st.columns(3)
        c1.metric("VSS out of sample", f"{oos['VSS_pct']:.2f}%")
        c2.metric("VSS in sample", f"{ins['VSS_pct_of_EEV']:.2f}%")
        if "EVPI_pct_of_RP" in ins:
            c3.metric("EVPI", f"{ins['EVPI_pct_of_RP']:.1f}%")
        st.markdown(
            f"Deterministic design: `{', '.join(sp['deterministic_open_dcs'])}` · "
            f"stochastic design: `{', '.join(sp['stochastic_open_dcs'])}`"
        )
        t = run.stochastic_tables
        if "sp_out_of_sample" in t and "deterministic_out_of_sample" in t:
            st.altair_chart(
                charts.outage_service_comparison(
                    t["sp_out_of_sample"], t["deterministic_out_of_sample"], mode
                ),
                width="stretch",
            )
        _table(
            pd.DataFrame(
                {"deterministic": oos["deterministic"], "stochastic": oos["stochastic"]}
            ).reset_index(names="measure"),
            "Out-of-sample comparison",
        )


main()
