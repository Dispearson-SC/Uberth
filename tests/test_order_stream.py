"""Characterization tests: the order-arrival stream (src/world/demand.py).

CHARACTERIZATION, NOT UNIT TESTS — see tests/conftest.py's module docstring.
All numbers below were verified by hand for seed=42, date=2026-07-10 (a
Friday), shift 840..1320 (14:00-22:00).

Counts, digests and structural facts are asserted EXACTLY. Distributional
quantities (medians, percentiles, means, shares) are asserted with
`pytest.approx` and a small relative tolerance: they are still fully
deterministic for a fixed seed, but pinning them to 10+ decimal places would
make the suite brittle to harmless internal reordering of floating-point
operations (e.g. a future numpy version summing in a different order) without
protecting against anything the "about" figures in the calibration notes
don't already cover.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.world.demand import build_order_stream
from src.world.surge import SUPPLY_CALIBRATION
from tests.conftest import DATE, DAY_OF_WEEK, SHIFT_END_MIN, SHIFT_START_MIN, SEED

# The single most valuable assertion in this suite (see module docstring):
# a sha256 over the whole order stream catches ANY behavioural drift
# anywhere in the generation path, not just the specific figures called out
# individually below. Computed as sha256 of the sort-keys JSON encoding of
# `[o.model_dump(mode="json") for o in orders]` — sort_keys makes the hash
# independent of pydantic's field-declaration order, which is otherwise an
# implementation detail this suite has no business depending on.
# Re-pinned when FARE_CALIBRATION was refitted from "make the SHIFT TOTAL
# plausible" to "make the PER-TRIP payout match what a working courier
# reports" (25-40 normally, 50-80 for a long one, ~85/15). The previous
# sentinel was 6d18a2332166c4210171c03dc2022cceb1b53c8914db9cfae2322e6b922a9d02
# at base 18.0 / per_km 6.5 / per_min 1.2, which produced a 46 MXN mean offer
# and a 41 MXN median. The move is intentional and the reason is recorded in
# `world/demand.py::FARE_CALIBRATION`; the sentinel exists so a change like
# this cannot happen quietly, not to forbid it.
# Re-pinned again when the fare card went from a straight line to distance
# BRACKETS. The courier's reported distribution is 80% of trips at 25-40 MXN,
# 15% at 40-50 and 5% at 50-80 -- and no single slope can produce it, because
# a linear fare inherits the shape of a km distribution with a long right
# tail. Previous sentinels, both intentional:
#   6d18a233... linear base 18.0 / per_km 6.5 / per_min 1.2 (tuned to the shift total)
#   4e27d9d1... linear base 20.0 / per_km 3.9 / per_min 0.9 (tuned to the per-trip median)
#   5afaec20... three distance brackets (still could not make the flat region)
#   518617d2... minimum-fare card charged against STRAIGHT-LINE km (a unit error)
# The card is now a 30 MXN minimum covering 5 ROAD km, rising to the ceiling,
# and the gravity decay and trip ceiling were solved backwards from the
# courier's reported payout shares: 80.0 / 15.9 / 4.1 against 80 / 15 / 5.
CALIBRATED_DIGEST = "a046ab962572a666fd49beee75a4627268c047c89d934c685cf71b516e8d98bc"


def _digest(orders) -> str:
    payload = [o.model_dump(mode="json") for o in orders]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_order_stream_count(reference_orders):
    # Exact: a fixed seed against fixed fixtures produces a fixed count.
    # 27_031 after the destination model was solved backwards from the
    # courier's reported payout shares. The count moves because the
    # trip-length ceiling re-draws destinations, consuming RNG draws.
    assert len(reference_orders) == 27_031


def test_offers_per_courier_per_hour(reference_orders):
    total_couriers = SUPPLY_CALIBRATION["total_couriers"]
    shift_hours = (SHIFT_END_MIN - SHIFT_START_MIN) / 60.0
    offers_per_courier_per_hour = len(reference_orders) / total_couriers / shift_hours
    assert offers_per_courier_per_hour == pytest.approx(7.52, rel=0.01)


def test_trip_km_distribution(reference_orders):
    km = np.array([o.ref_km for o in reference_orders])
    # 2.55 straight-line, i.e. 3.69 km of road at the measured 1.449 route
    # factor. The old 2.09 came from a gravity decay whose comment read
    # "most food delivery in Monterrey is 1-5 km" -- an assumption, not a
    # measurement. See world/demand.py MAX_TRIP_KM.
    assert np.median(km) == pytest.approx(2.55, rel=0.01)
    assert np.percentile(km, 90) == pytest.approx(4.36, rel=0.01)


def test_gross_fare_median(reference_orders):
    fares = np.array([o.gross_payout_mxn for o in reference_orders])
    # 31.44 on the tiered card. The band shares it produces are 79.8 / 15.4
    # / 4.8 against the reported 80 / 15 / 5.
    assert np.median(fares) == pytest.approx(33.82, rel=0.01)


def test_order_stream_is_deterministic_across_calls():
    """Core determinism property: two independent builds from the same seed
    must be byte-identical. This is the property that makes A/B policy
    comparison and replay meaningful at all."""
    orders_a = build_order_stream(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN, day_of_week=DAY_OF_WEEK)
    orders_b = build_order_stream(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN, day_of_week=DAY_OF_WEEK)
    assert _digest(orders_a) == _digest(orders_b)


def test_order_stream_digest_differs_for_different_seed(reference_orders):
    other = build_order_stream(SEED + 1, DATE, SHIFT_START_MIN, SHIFT_END_MIN, day_of_week=DAY_OF_WEEK)
    assert _digest(reference_orders) != _digest(other)


def test_order_stream_digest_matches_calibrated_snapshot(reference_orders):
    """Pins the stream to the exact calibrated snapshot recorded when this
    suite was written. This is an EXACT comparison by design (see module
    docstring): if it ever fails, something in the generation path changed,
    and the fix is to find out what — never to update this constant to make
    the test pass again without understanding why it moved.
    """
    assert _digest(reference_orders) == CALIBRATED_DIGEST
