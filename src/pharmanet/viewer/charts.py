"""Altair chart builders for the viewer (pure functions: data in, chart out).

Visual rules (from the project's data-viz guidance):
* Categorical colors come from a fixed, validated order (slots 1-3 only, which
  pass the colorblind checks for all pairs in light and dark mode); they follow
  the entity, never its rank. Dark mode uses its own validated steps.
* One axis per chart, never two. Thin marks: 4px rounded bar ends, 2px lines,
  >= 8px points. Recessive grid. Selective direct labels, not one per mark.
* Every chart has a tooltip and the app shows a table view beside it (light-mode
  aqua is below 3:1 contrast, so color is never the only way to read a value).
"""

from __future__ import annotations

from typing import Literal

import altair as alt
import pandas as pd

Mode = Literal["light", "dark"]
ChartLike = alt.Chart | alt.LayerChart | alt.FacetChart

PALETTE: dict[Mode, dict[str, str]] = {
    "light": {
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "s3": "#1baf7a",
        "muted": "#898781",
        "edge": "#c3c2b7",
        "ink": "#52514e",
    },
    "dark": {
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#199e70",
        "muted": "#898781",
        "edge": "#383835",
        "ink": "#c3c2b7",
    },
}

BAR_RADIUS = 4  # rounded data-end, px
POINT_SIZE = 90  # area in px^2 (~9.5px diameter)


def _c(mode: Mode) -> dict[str, str]:
    return PALETTE[mode]


def wape_by_model(wape: dict[str, float], mode: Mode) -> ChartLike:
    """Backtest WAPE per model; one series, so one color and no legend."""
    df = pd.DataFrame({"model": list(wape), "wape": list(wape.values())})
    best = df["wape"].min()
    base = alt.Chart(df).encode(
        y=alt.Y("model:N", sort=list(df.sort_values("wape")["model"]), title=None),
        x=alt.X("wape:Q", title="Portfolio WAPE (lower is better)", axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("model:N"), alt.Tooltip("wape:Q", format=".3f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=BAR_RADIUS, color=_c(mode)["s1"], height=18)
    label = (
        base.transform_filter(alt.datum.wape == best)
        .mark_text(align="left", dx=6, color=_c(mode)["ink"])
        .encode(text=alt.Text("wape:Q", format=".3f"))
    )
    chart: ChartLike = (bars + label).properties(height=140)
    return chart


def forecast_series(df: pd.DataFrame, mode: Mode) -> ChartLike:
    """Actual history vs 12-month forecast with a crosshair tooltip."""
    c = _c(mode)
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["Actual", "Forecast"], range=[c["s1"], c["s2"]]),
        legend=alt.Legend(title=None, orient="top"),
    )
    hover = alt.selection_point(fields=["month"], nearest=True, on="pointerover", empty=False)
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("month:T", title=None), y=alt.Y("units:Q", title="Units / month"), color=color
        )
    )
    rule = (
        alt.Chart(df)
        .mark_rule(color=c["muted"])
        .encode(
            x="month:T",
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("month:T", format="%b %Y"),
                "series:N",
                alt.Tooltip("units:Q", format=",.0f"),
            ],
        )
        .add_params(hover)
    )
    pts = line.mark_point(filled=True, size=POINT_SIZE).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    )
    return alt.layer(line, rule, pts).properties(height=280)


def network_map(nodes: pd.DataFrame, edges: pd.DataFrame, mode: Mode) -> ChartLike:
    """Plants, DCs, regions on the plane; edge width = annual flow.

    Three node kinds take series slots 1-3 *and* distinct shapes (secondary
    encoding); closed candidate DCs are hollow muted gray.
    """
    c = _c(mode)
    kinds = ["Plant", "Open DC", "Region", "Closed DC"]
    edge = (
        alt.Chart(edges)
        .mark_rule(color=c["edge"], opacity=0.8)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            x2="x2:Q",
            y2="y2:Q",
            strokeWidth=alt.StrokeWidth("flow:Q", scale=alt.Scale(range=[0.5, 6]), legend=None),
            tooltip=["src:N", "dst:N", "leg:N", alt.Tooltip("flow:Q", format=",.0f")],
        )
    )
    node = (
        alt.Chart(nodes)
        .mark_point(size=POINT_SIZE * 1.6, strokeWidth=2)
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=kinds, range=[c["s1"], c["s2"], c["s3"], c["muted"]]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            shape=alt.Shape(
                "kind:N",
                scale=alt.Scale(domain=kinds, range=["square", "diamond", "circle", "diamond"]),
                legend=None,
            ),
            fill=alt.Fill(
                "kind:N",
                scale=alt.Scale(domain=kinds, range=[c["s1"], c["s2"], c["s3"], "transparent"]),
                legend=None,
            ),
            tooltip=["id:N", "kind:N"],
        )
    )
    labels = (
        alt.Chart(nodes)
        .transform_filter(alt.datum.kind != "Region")
        .mark_text(dy=-14, fontSize=11, color=c["ink"])
        .encode(x="x:Q", y="y:Q", text="id:N")
    )
    return alt.layer(edge, node, labels).properties(height=520)


def cost_histogram(mc: pd.DataFrame, base_cost: float, mode: Mode) -> ChartLike:
    """Distribution of annual cost over Monte Carlo scenarios, with the forecast-demand cost."""
    c = _c(mode)
    hist = (
        alt.Chart(mc)
        .mark_bar(cornerRadiusEnd=BAR_RADIUS, color=c["s1"], binSpacing=2)
        .encode(
            x=alt.X(
                "cost_total:Q",
                bin=alt.Bin(maxbins=30),
                title="Annual cost ($)",
                axis=alt.Axis(format="$,.2s"),
            ),
            y=alt.Y("count():Q", title="Scenarios"),
            tooltip=[alt.Tooltip("count():Q", title="scenarios")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"cost": [base_cost], "label": ["at forecast"]}))
        .mark_rule(color=c["ink"], strokeWidth=2)
        .encode(x="cost:Q", tooltip=[alt.Tooltip("cost:Q", format="$,.0f"), "label:N"])
    )
    return alt.layer(hist, rule).properties(height=260)


def ranked_bars(
    df: pd.DataFrame, label: str, value: str, title: str, mode: Mode, fmt: str = ",.0f"
) -> ChartLike:
    """Horizontal ranking (one series). Only the largest bar is labeled directly."""
    c = _c(mode)
    top = df[value].max()
    base = alt.Chart(df).encode(
        # Explicit order: sort="-x" is dropped when layers are combined.
        y=alt.Y(f"{label}:N", sort=list(df.sort_values(value, ascending=False)[label]), title=None),
        x=alt.X(f"{value}:Q", title=title),
        tooltip=[f"{label}:N", alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    bars = base.mark_bar(cornerRadiusEnd=BAR_RADIUS, color=c["s1"], height=16)
    text = (
        base.transform_filter(alt.datum[value] == top)
        .mark_text(align="left", dx=6, color=c["ink"])
        .encode(text=alt.Text(f"{value}:Q", format=fmt))
    )
    return alt.layer(bars, text).properties(height=max(120, 26 * len(df)))


def bounds_chart(check: pd.DataFrame, mode: Mode) -> ChartLike:
    """Per node: graph lower bound (structural) to upper bound (exposure), with the
    simulated loss as a dot. The dot must sit inside its bar for every node."""
    c = _c(mode)
    df = check.dropna(subset=["simulated_loss"])
    order = list(df.sort_values("simulated_loss", ascending=False)["node"])
    base = alt.Chart(df).encode(y=alt.Y("node:N", sort=order, title=None))
    span = base.mark_bar(cornerRadiusEnd=BAR_RADIUS, height=10, color=c["edge"]).encode(
        x=alt.X("structural_loss:Q", title="Units of demand lost if the node fails"),
        x2="exposure:Q",
        tooltip=[
            "node:N",
            alt.Tooltip("structural_loss:Q", format=",.0f", title="lower bound"),
            alt.Tooltip("simulated_loss:Q", format=",.0f", title="simulated"),
            alt.Tooltip("exposure:Q", format=",.0f", title="upper bound"),
        ],
    )
    dot = base.mark_point(filled=True, size=POINT_SIZE, color=c["s1"]).encode(
        x="simulated_loss:Q",
        tooltip=["node:N", alt.Tooltip("simulated_loss:Q", format=",.0f", title="simulated")],
    )
    return alt.layer(span, dot).properties(height=max(160, 30 * len(df)))


def outage_service_comparison(stoch: pd.DataFrame, det: pd.DataFrame, mode: Mode) -> ChartLike:
    """Service level per outage scenario: deterministic vs stochastic design (2 series)."""
    c = _c(mode)
    rows = []
    for name, df in [("Deterministic", det), ("Stochastic", stoch)]:
        o = df[~df["scenario"].str.contains("_demand_")]
        rows.append(o.assign(design=name, outage=o["scenario"].str.replace("oos_", "")))
    data = pd.concat(rows, ignore_index=True)
    chart: ChartLike = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=BAR_RADIUS, height=8)
        .encode(
            y=alt.Y("outage:N", title=None),
            yOffset=alt.YOffset("design:N", sort=["Deterministic", "Stochastic"]),
            x=alt.X(
                "service_level:Q",
                title="Service level",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "design:N",
                scale=alt.Scale(domain=["Deterministic", "Stochastic"], range=[c["s1"], c["s2"]]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["outage:N", "design:N", alt.Tooltip("service_level:Q", format=".1%")],
        )
        .properties(height=max(200, 28 * data["outage"].nunique()))
    )
    return chart
