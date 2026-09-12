"""Characterization tests: the weather timeline (src/world/weather.py).

CHARACTERIZATION, NOT UNIT TESTS — see tests/conftest.py's module docstring.
All three dates are real Open-Meteo archive observations for Monterrey
(`fixtures/raw/openmeteo_mty_2026-06-01_2026-09-05.json`), not synthetic
data, verified by hand for the default 14:00-22:00 shift.
"""

from __future__ import annotations

import math

import pytest

from src.world.weather import (
    DEMO_DATE_BASELINE,
    DEMO_DATE_EXTREME_HEAT,
    DEMO_DATE_RAIN,
    DEMO_SHIFT_END_MIN,
    DEMO_SHIFT_START_MIN,
    build_weather_timeline,
)


def test_rain_demo_date_precipitation_and_temperature():
    ticks = build_weather_timeline(DEMO_DATE_RAIN, DEMO_SHIFT_START_MIN, DEMO_SHIFT_END_MIN)
    assert len(ticks) == 480

    total_precip = sum(t.precip_mm for t in ticks)
    assert total_precip == pytest.approx(6.6, rel=0.02)

    max_temp = max(t.temp_c for t in ticks)
    assert max_temp == pytest.approx(32.5, abs=0.1)


def test_extreme_heat_demo_date_triggers_flag():
    ticks = build_weather_timeline(DEMO_DATE_EXTREME_HEAT, DEMO_SHIFT_START_MIN, DEMO_SHIFT_END_MIN)
    max_apparent = max(t.apparent_c for t in ticks)
    assert max_apparent == pytest.approx(40.6, abs=0.1)
    # Exact: is_extreme_heat is a hard >= 40.0 threshold (see WeatherTick),
    # so at least one tick on this date must trip it.
    assert any(t.is_extreme_heat for t in ticks)


def test_baseline_demo_date_precipitation_and_temperature():
    ticks = build_weather_timeline(DEMO_DATE_BASELINE, DEMO_SHIFT_START_MIN, DEMO_SHIFT_END_MIN)
    total_precip = sum(t.precip_mm for t in ticks)
    assert total_precip == pytest.approx(1.0, abs=0.05)

    max_apparent = max(t.apparent_c for t in ticks)
    assert max_apparent == pytest.approx(34.3, abs=0.1)


@pytest.mark.parametrize("demo_date", [DEMO_DATE_RAIN, DEMO_DATE_EXTREME_HEAT, DEMO_DATE_BASELINE])
def test_no_nans_in_any_weather_series(demo_date):
    ticks = build_weather_timeline(demo_date, DEMO_SHIFT_START_MIN, DEMO_SHIFT_END_MIN)
    for tick in ticks:
        for value in (tick.temp_c, tick.apparent_c, tick.precip_mm, tick.wind_kmh, tick.humidity_pct):
            assert not math.isnan(value)
