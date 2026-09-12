"""Characterization tests: the traffic/congestion field (src/world/traffic.py).

CHARACTERIZATION, NOT UNIT TESTS — see tests/conftest.py's module docstring.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.world.traffic import build_traffic_timeline, travel_time_minutes

FRIDAY = date(2026, 7, 10)
SATURDAY = date(2026, 7, 11)


def test_friday_reference_shift_tick_count_and_multiplier_range():
    ticks = build_traffic_timeline(FRIDAY, 840, 1320)
    # Exact: one tick per minute of a half-open [840, 1320) range.
    assert len(ticks) == 480

    city_multipliers = [t.city_multiplier for t in ticks]
    assert min(city_multipliers) == pytest.approx(1.492, rel=0.01)
    assert max(city_multipliers) == pytest.approx(1.656, rel=0.01)


def test_travel_time_never_faster_than_free_flow():
    """Hard invariant stated in traffic.py's module docstring: congestion
    only ever slows a leg down. Checked as a property over every tick of a
    whole shift and a representative set of cells (including one absent
    from any tick's override map, to also exercise the city_multiplier
    fallback path), rather than a single spot-check value.
    """
    ticks = build_traffic_timeline(FRIDAY, 840, 1320)
    sample_cells = list(ticks[0].cell_multipliers.keys())[:5] + ["not_a_real_cell"]
    base_minutes_samples = (0.0, 1.0, 5.0, 12.3, 100.0)

    for tick in ticks:
        for cell in sample_cells:
            for base_minutes in base_minutes_samples:
                assert travel_time_minutes(base_minutes, tick, cell) >= base_minutes - 1e-9


def test_weekday_traffic_differs_from_saturday():
    """Weekday multipliers are 100% measured Monterrey data; weekend
    multipliers are that same weekday curve reshaped by a transferred
    Mexico City weekend/weekday ratio (see traffic.py's module docstring).
    The two must not be identical."""
    friday_ticks = build_traffic_timeline(FRIDAY, 840, 1320)
    saturday_ticks = build_traffic_timeline(SATURDAY, 840, 1320)

    friday_mults = [t.city_multiplier for t in friday_ticks]
    saturday_mults = [t.city_multiplier for t in saturday_ticks]
    assert friday_mults != saturday_mults
