"""Build simulator fixtures from raw INEGI/OSM data.

Stages (each idempotent — skips work whose output already exists unless
--force is passed):

    denue   parse DENUE, filter to food-preparation SCIAN codes (7225*) in
            the 4 target municipalities, compute demand weight -> restaurants.parquet
    ageb    parse INEGI Census 2020 AGEB population, derive AGEB centroids
            from DENUE coordinates, aggregate to H3 cells -> population.parquet
    cells   build the H3 res-7 operating grid from the restaurant point cloud -> cells.parquet
    graph   download/cache the OSMnx drive graph for the operating polygon -> monterrey_graph.graphml
    matrix  precompute the cell-to-cell distance/time matrices -> travel_matrix.npz
    all     run every stage above, in dependency order

Usage:
    python scripts/build_fixtures.py denue
    python scripts/build_fixtures.py all --force

`graph` and `matrix` invoke OSMnx and can take minutes on first run (network
download + Dijkstra per cell); `denue`/`ageb`/`cells` only need the local
INEGI CSVs and run in seconds.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.world import geo, network  # noqa: E402

FIXTURES_DIR = PROJECT_ROOT / "fixtures"

DENUE_PATH = PROJECT_ROOT / "Docs/denue_19_csv/conjunto_de_datos/denue_inegi_19_.csv"
AGEB_PATH = PROJECT_ROOT / "Docs/ageb_mza_urbana_19_cpv2020/conjunto_de_datos/conjunto_de_datos_ageb_urbana_19_cpv2020.csv"

RESTAURANTS_OUTPUT = FIXTURES_DIR / "restaurants.parquet"
POPULATION_OUTPUT = FIXTURES_DIR / "population.parquet"

FOOD_SCIAN_PREFIX = "7225"

# Reference counts from prior manual inspection of the DENUE file (see
# engram observation #133). Informational only ("~" approximate in the
# task spec) — printed and compared, not asserted, since minor drift is
# expected if INEGI republishes the extract.
EXPECTED_FOOD_ROWS_STATEWIDE = 26_151
EXPECTED_RESTAURANT_COUNTS = {
    "039": 6808,  # Monterrey
    "026": 2863,  # Guadalupe
    "046": 2373,  # San Nicolás de los Garza
    "019": 899,  # San Pedro Garza García
}

# CALIBRATION KNOBS (not measured data): SCIAN 6-digit delivery propensity
# factors. Fast/to-go formats generate more delivery-app order volume per
# venue than sit-down/a-la-carta formats; there is no INEGI field for this,
# these are hand-picked priors to be tuned during calibration.
SCIAN_DELIVERY_FACTOR: dict[str, float] = {
    "722517": 1.5,  # pizzas / burgers / hot dogs, prepared to-go
    "722516": 1.4,  # self-service restaurants
    "722514": 1.3,  # tacos and tortas stands
    "722518": 1.2,  # other prepared-to-go food
    "722513": 1.1,  # antojitos (local snack food)
    "722519": 1.0,  # other food-preparation services
    "722515": 0.9,  # cafeterias
    "722511": 0.8,  # a-la-carta / fixed-menu restaurants
    "722512": 0.6,  # seafood restaurants (typically dine-in, less delivery)
}
DEFAULT_DELIVERY_FACTOR = 1.0

# CALIBRATION KNOBS (not measured data): midpoint headcount per INEGI
# `per_ocu` stratum, used only as the input to a SUBLINEAR (sqrt) weight.
# Linear weighting by these midpoints would let the ~130 largest venues
# dominate the demand field and collapse it spatially.
PER_OCU_MIDPOINT: dict[str, float] = {
    "0 a 5 personas": 2.5,
    "6 a 10 personas": 8.0,
    "11 a 30 personas": 20.5,
    "31 a 50 personas": 40.5,
    "51 a 100 personas": 75.5,
    "101 a 250 personas": 175.5,
    "251 y más personas": 300.0,  # open-ended bracket, capped for weighting only
}
DEFAULT_PER_OCU_MIDPOINT = PER_OCU_MIDPOINT["0 a 5 personas"]

# Data-integrity reference counts for AGEB-level population rows (AGEB !=
# '0000' and MZA == '000') in the 4 target municipalities. These are NOT
# calibration values — they are asserted against the real file so silent
# parsing regressions (wrong filter, wrong encoding) fail loudly.
EXPECTED_AGEB_STATS: dict[str, tuple[int, int]] = {
    "039": (491, 1_142_952),  # Monterrey
    "026": (227, 642_928),  # Guadalupe
    "046": (127, 412_199),  # San Nicolás de los Garza
    "019": (55, 132_128),  # San Pedro Garza García
}
EXPECTED_TOTAL_AGEBS = 900
EXPECTED_TOTAL_POPULATION = 2_330_207


# --- shared DENUE parsing ----------------------------------------------------


def _iter_denue_rows(path: Path):
    with path.open(encoding="latin-1", newline="") as f:
        yield from csv.DictReader(f)


def compute_restaurant_weight(per_ocu: str, codigo_act: str) -> float:
    """Sublinear (sqrt) demand weight: sqrt(midpoint(per_ocu)) * SCIAN delivery factor."""
    midpoint = PER_OCU_MIDPOINT.get(per_ocu, DEFAULT_PER_OCU_MIDPOINT)
    factor = SCIAN_DELIVERY_FACTOR.get(codigo_act, DEFAULT_DELIVERY_FACTOR)
    return math.sqrt(midpoint) * factor


# --- stage: denue -> restaurants.parquet ------------------------------------


def build_restaurants(force: bool = False) -> pd.DataFrame:
    if RESTAURANTS_OUTPUT.exists() and not force:
        print(f"[denue] {RESTAURANTS_OUTPUT} already exists, skipping (use --force to rebuild)")
        return pd.read_parquet(RESTAURANTS_OUTPUT)

    print(f"[denue] parsing {DENUE_PATH} (encoding=latin-1)...")
    rows: list[dict] = []
    total_food_rows_statewide = 0
    per_mun_counts: Counter[str] = Counter()

    for row in _iter_denue_rows(DENUE_PATH):
        codigo_act = row["codigo_act"]
        if not codigo_act.startswith(FOOD_SCIAN_PREFIX):
            continue
        total_food_rows_statewide += 1

        cve_mun = row["cve_mun"]
        if cve_mun not in geo.TARGET_MUNICIPALITIES:
            continue
        per_mun_counts[cve_mun] += 1

        lat = float(row["latitud"])
        lon = float(row["longitud"])
        rows.append(
            {
                "denue_id": row["id"],
                "nom_estab": row["nom_estab"],
                "codigo_act": codigo_act,
                "nombre_act": row["nombre_act"],
                "per_ocu": row["per_ocu"],
                "cve_mun": cve_mun,
                "municipio": geo.TARGET_MUNICIPALITIES[cve_mun],
                "cve_loc": row["cve_loc"],
                "ageb": row["ageb"],
                "manzana": row["manzana"],
                "cod_postal": row["cod_postal"],
                "lat": lat,
                "lon": lon,
                "weight": compute_restaurant_weight(row["per_ocu"], codigo_act),
            }
        )

    df = pd.DataFrame(rows)
    df["cell"] = [geo.latlon_to_cell(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]

    print(f"[denue] {total_food_rows_statewide} rows with SCIAN prefix '{FOOD_SCIAN_PREFIX}' statewide "
          f"(reference: {EXPECTED_FOOD_ROWS_STATEWIDE})")
    print(f"[denue] {len(df)} restaurants in the 4 target municipalities:")
    for cve_mun, name in geo.TARGET_MUNICIPALITIES.items():
        actual = per_mun_counts.get(cve_mun, 0)
        expected = EXPECTED_RESTAURANT_COUNTS.get(cve_mun)
        flag = "" if actual == expected else f"  (reference: {expected})"
        print(f"  {name} ({cve_mun}): {actual}{flag}")

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESTAURANTS_OUTPUT, index=False)
    print(f"[denue] wrote {RESTAURANTS_OUTPUT}")
    return df


# --- stage: ageb -> population.parquet --------------------------------------


def _iter_ageb_population_rows(path: Path):
    """AGEB-level population rows for the target municipalities.

    Row-level convention: AGEB == '0000' is an entity/municipality aggregate
    (excluded); AGEB != '0000' and MZA == '000' is the AGEB-level total
    (kept); both non-zero is an individual manzana, finer than needed (skipped).
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["MUN"] not in geo.TARGET_MUNICIPALITIES:
                continue
            if row["AGEB"] == "0000":
                continue
            if row["MZA"] != "000":
                continue
            yield row


def _build_ageb_centroids_from_denue(path: Path) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """(cve_mun, cve_loc, ageb) -> (mean_lat, mean_lon, n_establishments).

    Uses ALL DENUE establishments (any SCIAN code) in the target
    municipalities, not just restaurants, for the best possible AGEB
    coordinate coverage — the population file itself carries no coordinates.
    """
    sums: dict[tuple[str, str, str], list[float]] = {}
    for row in _iter_denue_rows(path):
        cve_mun = row["cve_mun"]
        if cve_mun not in geo.TARGET_MUNICIPALITIES:
            continue
        ageb = row["ageb"]
        if not ageb or ageb == "0000":
            continue
        try:
            lat = float(row["latitud"])
            lon = float(row["longitud"])
        except ValueError:
            continue
        key = (cve_mun, row["cve_loc"], ageb)
        acc = sums.setdefault(key, [0.0, 0.0, 0])
        acc[0] += lat
        acc[1] += lon
        acc[2] += 1
    return {key: (lat_sum / n, lon_sum / n, n) for key, (lat_sum, lon_sum, n) in sums.items()}


def build_population(force: bool = False) -> pd.DataFrame:
    if POPULATION_OUTPUT.exists() and not force:
        print(f"[ageb] {POPULATION_OUTPUT} already exists, skipping (use --force to rebuild)")
        return pd.read_parquet(POPULATION_OUTPUT)

    print(f"[ageb] parsing {AGEB_PATH} (encoding=utf-8-sig)...")
    ageb_rows = list(_iter_ageb_population_rows(AGEB_PATH))

    per_mun_agebs: Counter[str] = Counter()
    per_mun_pop: Counter[str] = Counter()
    for row in ageb_rows:
        per_mun_agebs[row["MUN"]] += 1
        per_mun_pop[row["MUN"]] += int(row["POBTOT"])

    print("[ageb] AGEB-level rows found per target municipality:")
    for cve_mun, name in geo.TARGET_MUNICIPALITIES.items():
        print(f"  {name} ({cve_mun}): {per_mun_agebs.get(cve_mun, 0)} AGEBs, population {per_mun_pop.get(cve_mun, 0)}")

    total_agebs = len(ageb_rows)
    total_population = sum(int(r["POBTOT"]) for r in ageb_rows)
    print(f"[ageb] total: {total_agebs} AGEBs, population {total_population}")

    # Data-integrity assertions: these MUST reproduce the documented reference
    # counts, or the parsing (filter/encoding) has silently regressed.
    for cve_mun, (expected_n, expected_pop) in EXPECTED_AGEB_STATS.items():
        actual_n = per_mun_agebs.get(cve_mun, 0)
        actual_pop = per_mun_pop.get(cve_mun, 0)
        name = geo.TARGET_MUNICIPALITIES[cve_mun]
        assert actual_n == expected_n, f"{name}: expected {expected_n} AGEBs, got {actual_n}"
        assert actual_pop == expected_pop, f"{name}: expected population {expected_pop}, got {actual_pop}"
    assert total_agebs == EXPECTED_TOTAL_AGEBS, f"expected {EXPECTED_TOTAL_AGEBS} AGEBs total, got {total_agebs}"
    assert total_population == EXPECTED_TOTAL_POPULATION, (
        f"expected total population {EXPECTED_TOTAL_POPULATION}, got {total_population}"
    )
    print("[ageb] data-integrity assertions passed against documented reference counts")

    print("[ageb] deriving AGEB centroids from DENUE establishment coordinates...")
    centroids = _build_ageb_centroids_from_denue(DENUE_PATH)

    # Fallback centroid per municipality: mean of that municipality's
    # DENUE-covered AGEB centroids. This is an approximation (true adjacency
    # would need AGEB polygons, which we don't have — see engram #133) but
    # keeps every AGEB in-municipality rather than borrowing from elsewhere.
    fallback_sums: dict[str, list[float]] = {}
    for (cve_mun, _cve_loc, _ageb), (lat, lon, _n) in centroids.items():
        acc = fallback_sums.setdefault(cve_mun, [0.0, 0.0, 0])
        acc[0] += lat
        acc[1] += lon
        acc[2] += 1
    fallback_centroid = {mun: (lat_sum / n, lon_sum / n) for mun, (lat_sum, lon_sum, n) in fallback_sums.items() if n > 0}

    resolved_rows: list[dict] = []
    n_direct = n_fallback = n_dropped = 0
    for row in ageb_rows:
        cve_mun = row["MUN"]
        key = (cve_mun, row["LOC"], row["AGEB"])
        if key in centroids:
            lat, lon, _n = centroids[key]
            source = "denue"
            n_direct += 1
        elif cve_mun in fallback_centroid:
            lat, lon = fallback_centroid[cve_mun]
            source = "fallback_same_municipality"
            n_fallback += 1
        else:
            n_dropped += 1
            continue
        resolved_rows.append(
            {
                "cve_mun": cve_mun,
                "municipio": geo.TARGET_MUNICIPALITIES[cve_mun],
                "cve_loc": row["LOC"],
                "ageb": row["AGEB"],
                "population": int(row["POBTOT"]),
                "lat": lat,
                "lon": lon,
                "centroid_source": source,
            }
        )

    coverage_pct = 100.0 * n_direct / total_agebs if total_agebs else 0.0
    print(
        f"[ageb] centroid coverage: {n_direct} direct from DENUE, {n_fallback} same-municipality fallback, "
        f"{n_dropped} dropped (no coverage) -> {coverage_pct:.1f}% direct coverage"
    )
    if n_dropped:
        print(f"[ageb] WARNING: dropped {n_dropped} AGEBs with no DENUE coverage in their municipality")

    df = pd.DataFrame(resolved_rows)
    df["cell"] = [geo.latlon_to_cell(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]

    cell_agg = df.groupby("cell", as_index=False).agg(population=("population", "sum"), n_agebs=("ageb", "count"))

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    cell_agg.to_parquet(POPULATION_OUTPUT, index=False)
    print(f"[ageb] aggregated to {len(cell_agg)} H3 cells, total population {int(cell_agg['population'].sum())}")
    print(f"[ageb] wrote {POPULATION_OUTPUT}")
    return cell_agg


# --- stage: cells -> cells.parquet -------------------------------------------


def build_cells(force: bool = False) -> pd.DataFrame:
    if geo.CELLS_FIXTURE_PATH.exists() and not force:
        print(f"[cells] {geo.CELLS_FIXTURE_PATH} already exists, skipping (use --force to rebuild)")
        return geo.load_cell_index()

    restaurants = build_restaurants(force=False)
    print(f"[cells] building H3 res-{geo.H3_RESOLUTION} grid from convex hull of {len(restaurants)} restaurants...")
    df = geo.get_or_build_cell_index(restaurants["lat"], restaurants["lon"], force=force)
    print(f"[cells] {len(df)} cells, wrote {geo.CELLS_FIXTURE_PATH}")
    return df


# --- stage: graph -> monterrey_graph.graphml --------------------------------


def build_graph(force: bool = False):
    cells = build_cells(force=False)
    polygon = geo.build_operating_polygon(cells["lat"], cells["lon"], buffer_km=network.GRAPH_BUFFER_KM)

    if network.GRAPH_FIXTURE_PATH.exists() and not force:
        print(f"[graph] {network.GRAPH_FIXTURE_PATH} already exists, loading cached graph")
    else:
        print("[graph] downloading/building OSMnx drive graph (this can take a few minutes on first run)...")

    graph = network.get_or_build_graph(polygon, force=force)
    print(f"[graph] {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph


# --- stage: matrix -> travel_matrix.npz --------------------------------------


def build_matrix(force: bool = False):
    if network.MATRIX_FIXTURE_PATH.exists() and not force:
        print(f"[matrix] {network.MATRIX_FIXTURE_PATH} already exists, skipping (use --force to rebuild)")
        return None

    cells = build_cells(force=False)
    graph = build_graph(force=False)
    print(f"[matrix] computing {len(cells)}x{len(cells)} distance(km)/time(min) matrices (one Dijkstra per origin)...")
    matrix = network.TravelMatrix.build(graph, cells)
    matrix.save()
    print(f"[matrix] wrote {network.MATRIX_FIXTURE_PATH}")
    return matrix


STAGE_FUNCS = {
    "denue": build_restaurants,
    "ageb": build_population,
    "cells": build_cells,
    "graph": build_graph,
    "matrix": build_matrix,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulator fixtures from raw INEGI/OSM data.")
    parser.add_argument(
        "stage",
        choices=[*STAGE_FUNCS.keys(), "all"],
        help="Which fixture stage to build. 'all' runs every stage in dependency order.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if the output fixture already exists.")
    args = parser.parse_args()

    stages = list(STAGE_FUNCS.keys()) if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"=== stage: {stage} ===")
        STAGE_FUNCS[stage](force=args.force)


if __name__ == "__main__":
    main()
