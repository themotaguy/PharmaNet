"""Neo4j bundle tests: the import script matches the CSVs, and the pandas twins of
the Cypher queries give hand-checkable answers on the regression instance."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pharmanet.config import load_config
from pharmanet.graph.neo4j_export import (
    IMPORT_CYPHER,
    QUERIES,
    answer_queries,
    csv_columns_referenced,
    queries_cypher,
    write_neo4j_bundle,
)
from pharmanet.graph.risk import exposure_detail
from pharmanet.optimization.solution import Solution
from pharmanet.simulation.evaluate import evaluate_design
from regression_instance import regression_instance


def _tables(sol: Solution) -> dict[str, pd.DataFrame]:
    """Minimal export tables built straight from the regression instance."""
    inst = regression_instance()
    rows = lambda d: [(*k, v, k[2] in inst.cold_products) for k, v in sorted(d.items())]  # noqa: E731
    cols = ["from_id", "to_id", "product_id", "flow", "is_cold"]
    return {
        "ships_inbound": pd.DataFrame(rows(sol.x), columns=cols),
        "ships_outbound": pd.DataFrame(rows(sol.z), columns=cols),
        "qualified_for": pd.DataFrame(
            [(p, k, inst.cap_pk[(p, k)]) for (p, k), q in inst.qualified.items() if q],
            columns=["plant_id", "product_id", "capacity"],
        ),
        "products": pd.DataFrame({"id": ["A", "C"], "is_cold": [False, True]}),
    }


@pytest.fixture
def sol(small_config_path: Path) -> Solution:
    inst = regression_instance()
    inst = replace(inst, c_in={**inst.c_in, ("P2", "D2", "A"): 2.5})  # unique optimum
    return evaluate_design(
        inst, frozenset({"D1", "D2"}), load_config(small_config_path).optimization
    )


def test_import_script_only_reads_existing_csv_columns(tmp_path: Path) -> None:
    from pharmanet.data.generate import annual_demand, generate
    from pharmanet.graph.neo4j_export import export_tables
    from pharmanet.optimization.instance import build_instance

    cfg = load_config(Path(__file__).parents[1] / "configs" / "small_test.yaml")
    ds = generate(cfg)
    dd = annual_demand(ds.demand, ["region_id", "product_id"]).rename("design_demand")
    inst = build_instance(ds, dd.reset_index())
    s = evaluate_design(inst, frozenset(inst.dcs), cfg.optimization)
    tables = export_tables(ds, inst, s)
    referenced = csv_columns_referenced(IMPORT_CYPHER)
    assert set(referenced) == set(tables)  # every table is loaded, nothing else
    for name, cols in referenced.items():
        assert cols <= set(tables[name].columns), (name, cols - set(tables[name].columns))
    answers = write_neo4j_bundle(tmp_path, tables)
    assert (tmp_path / "import.cypher").exists() and (tmp_path / "README.md").exists()
    assert {p.stem for p in (tmp_path / "expected_answers").glob("*.csv")} == set(answers)


def test_queries_script_has_every_query_and_parameter() -> None:
    script = queries_cypher("P1")
    assert ":param plant => 'P1';" in script
    assert all(q.cypher in script for q in QUERIES) and len(QUERIES) == 5


def test_single_plant_cold_regions_by_hand(sol: Solution) -> None:
    """Only P1 makes C, so every region depends on P1 for it."""
    a = answer_queries(_tables(sol), "P1")["single_plant_cold_regions"]
    assert a.to_dict(orient="list") == {
        "region": ["R1", "R2", "R3"], "product": ["C"] * 3, "sole_plant": ["P1"] * 3
    }  # fmt: skip


def test_single_qualified_products_by_hand(sol: Solution) -> None:
    a = answer_queries(_tables(sol), "P1")["single_qualified_products"]
    assert a.to_dict(orient="records") == [{"product": "C", "is_cold": True, "sole_plant": "P1"}]


def test_dc_exposure_and_single_dc_regions_by_hand(sol: Solution) -> None:
    """D1 serves R1 (A 100 + C 50 = 150); D2 serves R2, R3 (300). Each region has one DC."""
    ans = answer_queries(_tables(sol), "P1")
    q3 = ans["dc_exposure"].set_index("dc")
    assert q3.loc["D2", "flow"] == pytest.approx(300.0)
    assert q3.loc["D1", "share"] == pytest.approx(150 / 450)
    assert ans["single_dc_regions"].to_dict(orient="list") == {
        "region": ["R1", "R2", "R3"], "sole_dc": ["D1", "D2", "D2"]
    }  # fmt: skip


def test_plant_failure_query_matches_graph_module(sol: Solution) -> None:
    """The Cypher twin must agree with Phase 6's proportional flow tracing."""
    inst = regression_instance()
    q5 = answer_queries(_tables(sol), "P1")["plant_failure_impact"]
    ref = exposure_detail(inst, sol)
    ref = ref[ref["node"] == "P1"].set_index(["region_id", "product_id"])["exposed"]
    got = q5.set_index(["region", "product"])["exposed"]
    pd.testing.assert_series_equal(got.sort_index(), ref.sort_index(), check_names=False)
    assert got.sum() == pytest.approx(450.0)


# --------------------------------------------------------------- live verification
def test_parse_cypher_shell_plain() -> None:
    from pharmanet.graph.neo4j_export import parse_cypher_shell_plain

    text = 'product, is_cold, sole_plant\n"SKU08", TRUE, "PL02"\n"SKU01", FALSE, "PL03"\n'
    df = parse_cypher_shell_plain(text)
    assert list(df.columns) == ["product", "is_cold", "sole_plant"]
    assert df["is_cold"].tolist() == [True, False] and df["product"].tolist() == ["SKU08", "SKU01"]


def test_compare_answers_flags_mismatch(tmp_path: Path) -> None:
    from pharmanet.graph.neo4j_export import compare_answers

    pd.DataFrame({"dc": ["D1"], "flow": [10.0]}).to_csv(tmp_path / "q.csv", index=False)
    good = compare_answers({"q": pd.DataFrame({"dc": ["D1"], "flow": [10.0]})}, tmp_path)
    bad = compare_answers({"q": pd.DataFrame({"dc": ["D1"], "flow": [11.0]})}, tmp_path)
    assert good[0]["match"] and not bad[0]["match"] and bad[0]["detail"]


def test_plant_from_queries_file(tmp_path: Path) -> None:
    from pharmanet.graph.neo4j_export import plant_from_queries_file

    f = tmp_path / "queries.cypher"
    f.write_text(queries_cypher("PL07"))
    assert plant_from_queries_file(f) == "PL07"


@pytest.mark.neo4j
def test_live_neo4j_matches_pandas() -> None:
    """Integration: the bundle loaded in a live Neo4j answers exactly like pandas.

    Opt-in (skipped in CI): load a bundle into a running container per the bundle
    README, then set PHARMANET_NEO4J_CONTAINER, NEO4J_PASSWORD and optionally
    PHARMANET_NEO4J_BUNDLE (default outputs/base/neo4j).
    """
    import os

    from pharmanet.graph.neo4j_export import (
        compare_answers,
        plant_from_queries_file,
        run_queries_in_docker,
    )

    container = os.environ.get("PHARMANET_NEO4J_CONTAINER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (container and password):
        pytest.skip("no live Neo4j container configured")
    bundle = Path(os.environ.get("PHARMANET_NEO4J_BUNDLE", "outputs/base/neo4j"))
    got = run_queries_in_docker(container, password, plant_from_queries_file(
        bundle / "queries.cypher"))  # fmt: skip
    results = compare_answers(got, bundle / "expected_answers")
    assert all(r["match"] for r in results), results
