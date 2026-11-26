"""Neo4j export: CSVs + import script + business queries + independently computed answers.

Graph model
-----------
Nodes:          (:Plant {id, x, y}), (:DC {id, x, y, is_open, capacity, cold_capacity,
                fixed_cost}), (:Region {id, x, y, demand}), (:Product {id, is_cold})
Relationships:  (:Plant)-[:SHIPS {product_id, flow, is_cold}]->(:DC)
                (:DC)-[:SHIPS {product_id, flow, is_cold}]->(:Region)
                (:Plant)-[:QUALIFIED_FOR {capacity}]->(:Product)
                (:Region)-[:DEMANDS {units}]->(:Product)

Load with ``cypher-shell -f import.cypher`` (CSV files in Neo4j's import folder),
then run ``queries.cypher``. Every query has a pandas twin in ``answer_queries``;
its results are written to ``expected_answers/`` so a live database can be
checked against an independent implementation. No Python Neo4j driver is
needed (and none is a dependency).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pharmanet.data.generate import Dataset
from pharmanet.optimization.instance import NetworkInstance
from pharmanet.optimization.solution import Solution


@dataclass(frozen=True)
class CypherQuery:
    name: str
    question: str
    cypher: str


QUERIES: tuple[CypherQuery, ...] = (
    CypherQuery(
        "single_plant_cold_regions",
        "Which regions depend on a single plant for a cold-chain product?",
        """MATCH (p:Plant)-[a:SHIPS]->(d:DC)-[b:SHIPS]->(r:Region)
WHERE a.product_id = b.product_id AND b.is_cold
WITH r, b.product_id AS product, collect(DISTINCT p.id) AS plants
WHERE size(plants) = 1
RETURN r.id AS region, product, plants[0] AS sole_plant
ORDER BY region, product;""",
    ),
    CypherQuery(
        "single_qualified_products",
        "Which products can be made at only one approved plant?",
        """MATCH (p:Plant)-[:QUALIFIED_FOR]->(k:Product)
WITH k, collect(p.id) AS plants
WHERE size(plants) = 1
RETURN k.id AS product, k.is_cold AS is_cold, plants[0] AS sole_plant
ORDER BY product;""",
    ),
    CypherQuery(
        "dc_exposure",
        "How much demand flows through each DC, and what share of the total?",
        """MATCH (:DC)-[s:SHIPS]->(:Region)
WITH sum(s.flow) AS total
MATCH (d:DC)-[s:SHIPS]->(:Region)
RETURN d.id AS dc, sum(s.flow) AS flow, sum(s.flow) / total AS share
ORDER BY flow DESC;""",
    ),
    CypherQuery(
        "single_dc_regions",
        "Which regions are served by only one DC?",
        """MATCH (d:DC)-[:SHIPS]->(r:Region)
WITH r, collect(DISTINCT d.id) AS dcs
WHERE size(dcs) = 1
RETURN r.id AS region, dcs[0] AS sole_dc
ORDER BY region;""",
    ),
    CypherQuery(
        "plant_failure_impact",
        "If plant $plant fails, how much of each product does each region lose "
        "(proportional flow tracing)?",
        """MATCH (p:Plant {id: $plant})-[a:SHIPS]->(d:DC)
MATCH (:Plant)-[i:SHIPS {product_id: a.product_id}]->(d)
WITH a, d, sum(i.flow) AS inflow
MATCH (d)-[b:SHIPS {product_id: a.product_id}]->(r:Region)
RETURN r.id AS region, a.product_id AS product, sum(a.flow / inflow * b.flow) AS exposed
ORDER BY exposed DESC, region, product;""",
    ),
)


# ------------------------------------------------------------------------- tables
def export_tables(ds: Dataset, inst: NetworkInstance, sol: Solution) -> dict[str, pd.DataFrame]:
    """CSV-ready tables describing the network and the solution's flows."""
    cold = inst.cold_products
    dcs = ds.dcs.assign(is_open=ds.dcs["dc_id"].isin(sol.open_dcs))
    region_demand = {r: sum(inst.dem[(r, k)] for k in inst.products) for r in inst.regions}
    inbound = pd.DataFrame(
        [(p, d, k, v, k in cold) for (p, d, k), v in sorted(sol.x.items())],
        columns=["from_id", "to_id", "product_id", "flow", "is_cold"],
    )
    outbound = pd.DataFrame(
        [(d, r, k, v, k in cold) for (d, r, k), v in sorted(sol.z.items())],
        columns=["from_id", "to_id", "product_id", "flow", "is_cold"],
    )
    qualified = [
        (p, k, inst.cap_pk[(p, k)])
        for p in inst.plants
        for k in inst.products
        if inst.qualified[(p, k)] and inst.cap_pk[(p, k)] > 0
    ]
    return {
        "plants": ds.plants.rename(columns={"plant_id": "id"})[["id", "x", "y"]],
        "dcs": dcs.rename(columns={"dc_id": "id", "throughput_capacity": "capacity"})[
            ["id", "x", "y", "is_open", "capacity", "cold_capacity", "fixed_cost"]
        ],
        "regions": ds.regions.rename(columns={"region_id": "id"})[["id", "x", "y"]].assign(
            demand=lambda f: f["id"].map(region_demand)
        ),
        "products": pd.DataFrame(
            {"id": list(inst.products), "is_cold": [k in cold for k in inst.products]}
        ),
        "ships_inbound": inbound,
        "ships_outbound": outbound,
        "qualified_for": pd.DataFrame(qualified, columns=["plant_id", "product_id", "capacity"]),
        "demands": pd.DataFrame(
            [(r, k, v) for (r, k), v in sorted(inst.dem.items())],
            columns=["region_id", "product_id", "units"],
        ),
    }


IMPORT_CYPHER = """// PharmaNet import (synthetic data). Run: cypher-shell -f import.cypher
// The CSV files must be in Neo4j's import directory.
CREATE CONSTRAINT plant_id IF NOT EXISTS FOR (n:Plant) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT dc_id IF NOT EXISTS FOR (n:DC) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT region_id IF NOT EXISTS FOR (n:Region) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///plants.csv' AS row
MERGE (n:Plant {id: row.id}) SET n.x = toFloat(row.x), n.y = toFloat(row.y);

LOAD CSV WITH HEADERS FROM 'file:///dcs.csv' AS row
MERGE (n:DC {id: row.id})
SET n.x = toFloat(row.x), n.y = toFloat(row.y), n.is_open = toBoolean(row.is_open),
    n.capacity = toFloat(row.capacity), n.cold_capacity = toFloat(row.cold_capacity),
    n.fixed_cost = toFloat(row.fixed_cost);

LOAD CSV WITH HEADERS FROM 'file:///regions.csv' AS row
MERGE (n:Region {id: row.id})
SET n.x = toFloat(row.x), n.y = toFloat(row.y), n.demand = toFloat(row.demand);

LOAD CSV WITH HEADERS FROM 'file:///products.csv' AS row
MERGE (n:Product {id: row.id}) SET n.is_cold = toBoolean(row.is_cold);

LOAD CSV WITH HEADERS FROM 'file:///ships_inbound.csv' AS row
MATCH (a:Plant {id: row.from_id}), (b:DC {id: row.to_id})
CREATE (a)-[:SHIPS {product_id: row.product_id, flow: toFloat(row.flow),
                    is_cold: toBoolean(row.is_cold)}]->(b);

LOAD CSV WITH HEADERS FROM 'file:///ships_outbound.csv' AS row
MATCH (a:DC {id: row.from_id}), (b:Region {id: row.to_id})
CREATE (a)-[:SHIPS {product_id: row.product_id, flow: toFloat(row.flow),
                    is_cold: toBoolean(row.is_cold)}]->(b);

LOAD CSV WITH HEADERS FROM 'file:///qualified_for.csv' AS row
MATCH (p:Plant {id: row.plant_id}), (k:Product {id: row.product_id})
CREATE (p)-[:QUALIFIED_FOR {capacity: toFloat(row.capacity)}]->(k);

LOAD CSV WITH HEADERS FROM 'file:///demands.csv' AS row
MATCH (r:Region {id: row.region_id}), (k:Product {id: row.product_id})
CREATE (r)-[:DEMANDS {units: toFloat(row.units)}]->(k);
"""


def queries_cypher(plant: str) -> str:
    """All business queries as one cypher-shell script (with the $plant parameter set)."""
    parts = [f"// PharmaNet business queries (synthetic data)\n:param plant => '{plant}';\n"]
    for q in QUERIES:
        parts.append(f"// {q.name}: {q.question}\n{q.cypher}\n")
    return "\n".join(parts)


def csv_columns_referenced(script: str) -> dict[str, set[str]]:
    """{csv file: columns used as row.<col>} for each LOAD CSV block (consistency checks)."""
    out: dict[str, set[str]] = {}
    for block in script.split("LOAD CSV")[1:]:
        m = re.search(r"file:///(\w+)\.csv", block)
        if m:
            body = block.split(";")[0]
            out[m.group(1)] = set(re.findall(r"row\.(\w+)", body))
    return out


# ---------------------------------------------------------------- pandas answers
def answer_queries(tables: dict[str, pd.DataFrame], plant: str) -> dict[str, pd.DataFrame]:
    """Independent pandas implementation of every Cypher query (same columns, same order)."""
    inb, outb = tables["ships_inbound"], tables["ships_outbound"]
    paths = inb.merge(outb, left_on=["to_id", "product_id"], right_on=["from_id", "product_id"],
                      suffixes=("_in", "_out"))  # fmt: skip

    cold_paths = paths[paths["is_cold_out"]]
    q1 = (
        cold_paths.groupby(["to_id_out", "product_id"])["from_id_in"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    q1 = q1[q1["from_id_in"].map(len) == 1].assign(sole_plant=lambda f: f["from_id_in"].str[0])
    q1 = q1.rename(columns={"to_id_out": "region", "product_id": "product"})[
        ["region", "product", "sole_plant"]
    ].sort_values(["region", "product"])

    qf = tables["qualified_for"].groupby("product_id")["plant_id"].agg(list).reset_index()
    qf = qf[qf["plant_id"].map(len) == 1]
    is_cold = tables["products"].set_index("id")["is_cold"]
    q2 = pd.DataFrame(
        {"product": qf["product_id"], "is_cold": qf["product_id"].map(is_cold),
         "sole_plant": qf["plant_id"].str[0]}
    ).sort_values("product")  # fmt: skip

    total = outb["flow"].sum()
    q3 = outb.groupby("from_id")["flow"].sum().reset_index().rename(columns={"from_id": "dc"})
    q3 = q3.assign(share=q3["flow"] / total).sort_values("flow", ascending=False)

    q4 = outb.groupby("to_id")["from_id"].agg(lambda s: sorted(set(s))).reset_index()
    q4 = q4[q4["from_id"].map(len) == 1].assign(sole_dc=lambda f: f["from_id"].str[0])
    q4 = q4.rename(columns={"to_id": "region"})[["region", "sole_dc"]].sort_values("region")

    inflow = inb.groupby(["to_id", "product_id"])["flow"].sum().rename("inflow").reset_index()
    p = paths[paths["from_id_in"] == plant].merge(
        inflow, left_on=["to_id_in", "product_id"], right_on=["to_id", "product_id"]
    )
    p = p.assign(exposed=p["flow_in"] / p["inflow"] * p["flow_out"])
    q5 = (
        p.groupby(["to_id_out", "product_id"])["exposed"].sum().reset_index()
        .rename(columns={"to_id_out": "region", "product_id": "product"})
        .sort_values(["exposed", "region", "product"], ascending=[False, True, True])
    )  # fmt: skip

    answers = {
        "single_plant_cold_regions": q1,
        "single_qualified_products": q2,
        "dc_exposure": q3,
        "single_dc_regions": q4,
        "plant_failure_impact": q5,
    }
    return {k: v.reset_index(drop=True) for k, v in answers.items()}


def default_failure_plant(tables: dict[str, pd.DataFrame]) -> str:
    """The plant with the most outbound flow (the most interesting one to 'fail')."""
    inb = tables["ships_inbound"]
    return str(inb.groupby("from_id")["flow"].sum().idxmax())


BUNDLE_README = """# PharmaNet Neo4j bundle (synthetic data)

1. Start Neo4j 5 with this folder as its import directory, e.g.
   `docker run --rm -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<password> \\
      -v "$PWD":/import neo4j:5`
2. Load: `cypher-shell -u neo4j -p <password> -f import.cypher`
   (inside the container: `docker exec -i <id> cypher-shell -u neo4j -p <password> < import.cypher`)
3. Query: `cypher-shell -u neo4j -p <password> -f queries.cypher`
4. Compare with `expected_answers/*.csv`, computed independently in pandas.
"""


def write_neo4j_bundle(
    directory: Path, tables: dict[str, pd.DataFrame], plant: str | None = None
) -> dict[str, pd.DataFrame]:
    """Write CSVs, import.cypher, queries.cypher, expected answers, and a README."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(directory / f"{name}.csv", index=False)
    plant = plant or default_failure_plant(tables)
    (directory / "import.cypher").write_text(IMPORT_CYPHER)
    (directory / "queries.cypher").write_text(queries_cypher(plant))
    (directory / "README.md").write_text(BUNDLE_README)
    answers = answer_queries(tables, plant)
    adir = directory / "expected_answers"
    adir.mkdir(exist_ok=True)
    for name, df in answers.items():
        df.to_csv(adir / f"{name}.csv", index=False)
    return answers
