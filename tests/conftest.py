"""Shared fixtures for the characterization test suite.

CHARACTERIZATION, NOT UNIT TESTS: every assertion in `tests/` pins a number
this simulator currently produces for a fixed, hand-verified scenario. These
tests do not judge whether the calibration is "good" — several of the
constants they lock (surge's `k=2.0`, the courier time-of-day profile, the
fare coefficients...) are empirically tuned knobs, not derived truths, and
one documented window (the Day realism band, see
`tests/test_surge_realism_windows.py`) is asserted as a KNOWN LIMITATION,
not a target. The point of this suite is regression detection: if any of
these numbers drift, a constant changed somewhere and nobody noticed.

Reference scenario used throughout, unless a test states otherwise:
    seed = 42, date = 2026-07-10 (a Friday), shift = 840..1320 min
    (14:00-22:00), day_of_week = "Friday".
All target numbers in this suite were verified by hand against this exact
scenario.

Expensive artifacts (a ~27k-order stream takes ~3.3s to generate; the real
OSMnx drive-graph fixture takes ~20s to parse) are built ONCE per test
session/module via the fixtures below, not once per test.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world import demand as demand_mod  # noqa: E402
from src.world import events as events_mod  # noqa: E402
from src.world import surge as surge_mod  # noqa: E402
from src.world.demand import build_order_stream  # noqa: E402
from src.world.scenario import rng_streams  # noqa: E402

# --------------------------------------------------------------------------
# Reference scenario constants
# --------------------------------------------------------------------------

SEED = 42
DATE = date(2026, 7, 10)  # Friday
DAY_OF_WEEK = "Friday"
SHIFT_START_MIN = 840  # 14:00
SHIFT_END_MIN = 1320  # 22:00


def build_surge_field(
    seed: int = SEED,
    d: date = DATE,
    start: int = SHIFT_START_MIN,
    end: int = SHIFT_END_MIN,
    day_of_week: str = DAY_OF_WEEK,
    calibration: Mapping[str, float] | None = None,
):
    """Reconstruct the ground-truth `SurgeField` the same way `demand.py`
    does internally, without paying for full order generation (which also
    draws restaurant/destination/prep/tip randomness irrelevant to surge
    dynamics). ~1s instead of ~3.3s.

    `calibration` is forwarded to `surge.build_supply_and_surge` as a
    non-mutating override (see that function's `calibration` parameter) —
    this is how `test_surge.py`'s lag-emergence test zeroes the reaction lag
    without touching the module-level `SUPPLY_CALIBRATION` dict at all, so
    there is nothing to restore and no state-leak risk between tests.
    """
    minutes = list(range(start, end))
    n_min = len(minutes)
    restaurants = demand_mod._load_restaurants()
    population = demand_mod._load_population()
    workplaces = demand_mod._load_workplaces()
    rngs = rng_streams(seed)
    model = demand_mod._DemandModel(restaurants, population, workplaces, rngs["kitchen"])

    day_type = demand_mod.day_type_for(day_of_week)
    profile_arr = np.array([demand_mod.temporal_profile(m % 1440, day_type) for m in minutes])
    rate = demand_mod.DEMAND_CALIBRATION["orders_per_weight_unit_per_min"]
    demand_by_cell = {cell: model.cell_weight_sum[cell] * rate * profile_arr for cell in model.origin_cells}
    demand_full = {cell: demand_by_cell.get(cell, np.zeros(n_min)) for cell in model.full_cell_grid}

    return surge_mod.build_supply_and_surge(
        rng=rngs["competitors"],
        cells=model.full_cell_grid,
        minutes=minutes,
        demand_by_cell=demand_full,
        baseline_weights=model.baseline_weights,
        calibration=calibration,
    )


@pytest.fixture(scope="session")
def reference_orders():
    """The full reference-scenario order stream (~27k orders, ~3.3s to
    build). Session-scoped: every test file that needs the reference shift's
    order-level statistics (trip km, fare, surge_at_spawn distribution,
    determinism digest) shares this single build."""
    return build_order_stream(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN, day_of_week=DAY_OF_WEEK)


@pytest.fixture(scope="session")
def reference_surge_field():
    """The reference-scenario SurgeField alone (cells x minutes), for
    temporal/cross-sectional variance properties that don't need the actual
    sampled orders."""
    return build_surge_field()


@pytest.fixture(scope="module")
def loaded_graph():
    """Load the real ~120MB OSMnx drive-graph fixture ONCE for a test
    module, then monkeypatch `events._try_load_graph` so every
    `build_events_timeline` call inside that module reuses this in-memory
    graph instead of re-parsing the file from disk (~20s each) again.

    This does not change behaviour: `build_events_timeline` only ever reads
    from the object `_try_load_graph` returns, never re-derives anything
    from the file path itself.
    """
    graph = events_mod._try_load_graph()
    original = events_mod._try_load_graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph
    try:
        yield graph
    finally:
        events_mod._try_load_graph = original
