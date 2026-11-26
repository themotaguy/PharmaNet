"""End-to-end pipeline and report tests (small config, in a temp directory)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharmanet.config import load_config
from pharmanet.pipeline import config_fingerprint, run_all
from pharmanet.report import _money, load_results, render_markdown, resume_numbers, write_report


@pytest.fixture(scope="module")
def small_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full run-all on the small config, shared by the tests below."""
    root = tmp_path_factory.mktemp("runall")
    cfg_path = Path(__file__).parents[1] / "configs" / "small_test.yaml"
    cfg = load_config(cfg_path)
    cfg = cfg.model_copy(update={"run": cfg.run.model_copy(update={"output_dir": root})})
    out, _ = run_all(cfg, run_id="e2e", config_file=str(cfg_path), publish_to=root / "docs.md")
    return out


def test_run_all_writes_every_stage(small_run: Path) -> None:
    for stage in (
        "data",
        "forecast",
        "optimize",
        "simulate",
        "graph",
        "neo4j",
        "stochastic",
        "benchmark",
    ):
        assert (small_run / stage).is_dir(), stage
    assert (small_run / "results.md").exists()
    assert (small_run.parent / "docs.md").read_text() == (small_run / "results.md").read_text()


def test_manifest_records_provenance(small_run: Path) -> None:
    m = json.loads((small_run / "manifest.json").read_text())
    assert set(m["stage_seconds"]) == {
        "generate", "forecast", "optimize", "simulate", "graph", "neo4j", "stochastic", "benchmark"
    }  # fmt: skip
    assert len(m["config_sha256"]) == 64 and m["seed"] == 7
    assert m["package_version"]


def test_report_is_honest_and_complete(small_run: Path) -> None:
    md = (small_run / "results.md").read_text()
    assert "All data is synthetic" in md.splitlines()[2]
    for heading in ("## 1. Forecasting", "## 3. Network design (MILP)", "## 5. Simulation",
                    "## 6. Graph risk and mitigation", "## 7. Two-stage stochastic design",
                    "## Resume numbers"):  # fmt: skip
        assert heading in md
    assert "do not quote it" in md  # the in-sample forecast figure is flagged


def test_resume_numbers_cover_the_spec(small_run: Path) -> None:
    """CLAUDE.md lists five numbers the pipeline must report; each needs a caveat."""
    nums = resume_numbers(load_results(small_run))
    claims = " | ".join(n["claim"] for n in nums)
    for needle in (
        "greedy",
        "WAPE",
        "Service level",
        "Demand at risk",
        "instance size",
        "stochastic",
    ):
        assert needle in claims
    assert all(n["caveat"] and n["source"] for n in nums)


def test_report_regenerates_identically(small_run: Path) -> None:
    before = (small_run / "results.md").read_text()
    write_report(small_run)
    assert (small_run / "results.md").read_text() == before
    assert render_markdown(load_results(small_run)) == before


def test_report_fails_loudly_on_partial_run(tmp_path: Path) -> None:
    (tmp_path / "forecast").mkdir()
    (tmp_path / "forecast" / "summary.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="optimize"):
        load_results(tmp_path)


def test_config_fingerprint(small_config_path: Path) -> None:
    cfg = load_config(small_config_path)
    other = cfg.model_copy(update={"run": cfg.run.model_copy(update={"seed": 8})})
    assert config_fingerprint(cfg) == config_fingerprint(load_config(small_config_path))
    assert config_fingerprint(cfg) != config_fingerprint(other)


@pytest.mark.parametrize(
    ("x", "s"), [(4_187_530.0, "$4.19M"), (852_699.0, "$853k"), (999.0, "$999")]
)
def test_money_format(x: float, s: str) -> None:
    assert _money(x) == s


# ---------------------------------------------------------------------- viewer
def test_viewer_loads_run_and_builds_every_chart(small_run: Path) -> None:
    import altair as alt

    from pharmanet.viewer import charts
    from pharmanet.viewer.data import list_runs, load_run, network_frames, series_frame

    assert small_run in list_runs(small_run.parent)
    run = load_run(small_run)
    nodes, edges = network_frames(run)
    assert set(nodes["kind"]) >= {"Plant", "Open DC", "Region"}
    assert edges["flow"].sum() > 0 and edges[["x", "y", "x2", "y2"]].notna().all().all()
    region, product = run.demand["region_id"].iloc[0], run.demand["product_id"].iloc[0]
    sdf = series_frame(run, region, product)
    assert set(sdf["series"]) == {"Actual", "Forecast"}
    res = run.results
    t = run.stochastic_tables
    for mode in ("light", "dark"):
        built = [
            charts.wape_by_model(res["forecast"]["portfolio_wape_by_model"], mode),
            charts.forecast_series(sdf, mode),
            charts.network_map(nodes, edges, mode),
            charts.cost_histogram(run.monte_carlo, 1.0, mode),
            charts.ranked_bars(run.disruptions, "scenario_id", "unmet", "Unmet", mode),
            charts.bounds_chart(res["cross_check"], mode),
            charts.outage_service_comparison(
                t["sp_out_of_sample"], t["deterministic_out_of_sample"], mode
            ),
        ]
        for chart in built:
            assert isinstance(chart, alt.TopLevelMixin)
            assert chart.to_dict()  # valid Vega-Lite spec


def test_viewer_uses_mode_specific_palette() -> None:
    from pharmanet.viewer.charts import PALETTE

    assert PALETTE["light"]["s1"] != PALETTE["dark"]["s1"]  # dark is selected, not flipped


def test_streamlit_app_runs_without_errors(
    small_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PHARMANET_OUTPUT_DIR", str(small_run.parent))
    app = Path(__file__).parents[1] / "src" / "pharmanet" / "viewer" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, at.exception
    assert "synthetic" in at.caption[0].value
    assert len(at.tabs) == 6
    assert at.sidebar.selectbox[0].value == small_run.name


def test_streamlit_app_handles_no_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PHARMANET_OUTPUT_DIR", str(tmp_path))
    app = Path(__file__).parents[1] / "src" / "pharmanet" / "viewer" / "app.py"
    at = AppTest.from_file(str(app)).run()
    assert not at.exception
    assert "No finished runs" in at.info[0].value
