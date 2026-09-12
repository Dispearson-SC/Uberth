"""Characterization tests: on-disk fixture integrity.

These lock the shape of the committed data fixtures themselves (row counts,
sums, resolution) so a fixture regeneration (`scripts/build_fixtures.py`,
`scripts/derive_monterrey_traffic_profile.py`) that silently changes the
underlying data is caught immediately, before it can be blamed on the
simulator logic. All comparisons here are EXACT: these are counts and sums
over static, committed files, not statistical draws — any deviation, however
small, means the fixture itself changed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import PROJECT_ROOT

FIXTURES_DIR = PROJECT_ROOT / "fixtures"


def test_restaurants_fixture_row_count_and_municipality_breakdown():
    df = pd.read_parquet(FIXTURES_DIR / "restaurants.parquet")
    assert len(df) == 12_943

    counts = df["municipio"].value_counts().to_dict()
    assert counts == {
        "Monterrey": 6_808,
        "Guadalupe": 2_863,
        "San Nicolás de los Garza": 2_373,
        "San Pedro Garza García": 899,
    }


def test_population_fixture_cell_count_and_sums():
    df = pd.read_parquet(FIXTURES_DIR / "population.parquet")
    assert len(df) == 92
    assert int(df["population"].sum()) == 2_330_207
    assert int(df["n_agebs"].sum()) == 900


def test_workplaces_fixture_employment_and_establishment_totals():
    df = pd.read_parquet(FIXTURES_DIR / "workplaces.parquet")
    assert int(df["employment"].sum()) == 367_068
    assert int(df["n_establishments"].sum()) == 24_471


def test_cells_fixture_count_and_resolution():
    df = pd.read_parquet(FIXTURES_DIR / "cells.parquet")
    assert len(df) == 127
    assert set(df["resolution"].unique().tolist()) == {7}


def test_hourly_traffic_profile_free_flow_reference_and_peak():
    df = pd.read_csv(FIXTURES_DIR / "monterrey_hourly_profile.csv")
    assert len(df) == 24

    hour3 = df.loc[df["hour"] == 3, "congestion_multiplier"].iloc[0]
    # Exact: hour 3 is defined as the free-flow reference hour, so this
    # multiplier is 1.0 by construction, not an empirical coincidence.
    assert hour3 == pytest.approx(1.0, abs=1e-9)

    hour12 = df.loc[df["hour"] == 12, "congestion_multiplier"].iloc[0]
    assert hour12 == pytest.approx(1.679, abs=0.01)
