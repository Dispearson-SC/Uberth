"""Ground truth: per-minute weather timeline for the operating area.

This module is ground truth. It turns real archived Open-Meteo observations
for Monterrey (`fixtures/raw/openmeteo_mty_2026-06-01_2026-09-05.json`) into
a per-minute `WeatherTick` series for a simulated shift. Weather is a pure
function of `(date, shift_start_min, shift_end_min)` and the fixture data: it
never depends on courier actions, and it is deterministic (no RNG is used at
all, since the source is real historical observations, not a stochastic
model). `src/agent/` must never import this module directly.

The interpolated fields (`temp_c`, `apparent_c`, `wind_kmh`,
`humidity_pct`, `weather_code`) are measured Open-Meteo data, not a
calibration knob. The *derived impact factors* at the bottom of this module
(`speed_factor`, `demand_factor`, `courier_supply_factor`, `risk_factor`)
are a different thing entirely: there is no public Monterrey dataset linking
rain intensity to courier speed, demand, supply or risk, so those mappings
are explicit, tunable calibration dicts. Say so out loud whenever presenting
results built on them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

from src.world.timeline import WeatherTick

# --------------------------------------------------------------------------
# Fixture location
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEATHER_FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "raw" / "openmeteo_mty_2026-06-01_2026-09-05.json"

MINUTES_PER_HOUR = 60

# --------------------------------------------------------------------------
# Verified demo dates (real Open-Meteo data, shift 14:00-22:00 local)
# --------------------------------------------------------------------------

# Friday, rain scenario: 6.6 mm accumulated over the 14:00-22:00 shift,
# max temperature 32.5 C. Verified against the raw fixture.
DEMO_DATE_RAIN: Date = Date(2026, 6, 12)

# Friday, extreme heat scenario: apparent temperature peaks at 40.6 C
# (real 36.6 C) right at shift start (14:00). Verified against the raw
# fixture.
DEMO_DATE_EXTREME_HEAT: Date = Date(2026, 6, 19)

# Friday, baseline scenario: 1.0 mm accumulated over the shift, 34.3 C
# apparent temperature peak. Verified against the raw fixture.
DEMO_DATE_BASELINE: Date = Date(2026, 7, 10)

# All three demo dates use the default Friday 14:00-22:00 shift (see
# `src.world.scenario.DEFAULT_SHIFT_START_MIN` / `DEFAULT_SHIFT_END_MIN`),
# repeated here as named constants so this module has no import-time
# dependency on scenario.py for its own demo wiring.
DEMO_SHIFT_START_MIN = 840
DEMO_SHIFT_END_MIN = 1320


# --------------------------------------------------------------------------
# Fixture loading (cached; the raw JSON is parsed once per process)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _HourlyRecord:
    """One hourly Open-Meteo observation, parsed to native types."""

    dt: DateTime
    temp_c: float
    apparent_c: float
    precip_mm: float
    wind_kmh: float
    humidity_pct: float
    weather_code: int


@lru_cache(maxsize=1)
def _load_hourly_records(fixture_path: str) -> tuple[_HourlyRecord, ...]:
    """Parse the raw Open-Meteo archive JSON into hourly records.

    Cached because the fixture is ~2300 hourly rows and `build_weather_timeline`
    is called once per shift per scenario build; re-parsing per call would be
    wasted work with no benefit (the fixture file never changes at runtime).
    """
    with open(fixture_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    hourly = raw["hourly"]
    records = []
    for i, iso_time in enumerate(hourly["time"]):
        records.append(
            _HourlyRecord(
                dt=DateTime.fromisoformat(iso_time),
                temp_c=float(hourly["temperature_2m"][i]),
                apparent_c=float(hourly["apparent_temperature"][i]),
                precip_mm=float(hourly["precipitation"][i]),
                wind_kmh=float(hourly["wind_speed_10m"][i]),
                humidity_pct=float(hourly["relative_humidity_2m"][i]),
                weather_code=int(hourly["weather_code"][i]),
            )
        )
    return tuple(records)


@lru_cache(maxsize=1)
def _hourly_index(fixture_path: str) -> dict[DateTime, int]:
    """Map an exact on-the-hour timestamp to its record index."""
    records = _load_hourly_records(fixture_path)
    return {record.dt: i for i, record in enumerate(records)}


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between `a` and `b` at fraction `t` in [0, 1]."""
    return a + (b - a) * t


# --------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------


def build_weather_timeline(date: Date, shift_start_min: int, shift_end_min: int) -> list[WeatherTick]:
    """Build one `WeatherTick` per simulated minute of the shift.

    `shift_start_min`/`shift_end_min` are minutes since local midnight on
    `date` (e.g. 840 = 14:00, 1320 = 22:00), matching `Scenario.shift_start_min`
    / `shift_end_min`. The output has exactly `shift_end_min - shift_start_min`
    ticks, minute-indexed and half-open (`shift_end_min` itself excluded, same
    convention as `range()`).

    Interpolation rules, per field kind:
    - Continuous fields (`temp_c`, `apparent_c`, `wind_kmh`, `humidity_pct`):
      linearly interpolated between the two bracketing hourly observations.
    - `precip_mm`: Open-Meteo's `precipitation` is an *hourly accumulation*,
      not an instantaneous reading. Linearly interpolating an accumulated
      value would fabricate a smooth ramp across the hour that the source
      data says nothing about. Instead the hour's total is spread evenly
      across its 60 minutes, so summing `precip_mm` over any full hour of
      ticks reproduces the original hourly total exactly.
    - `weather_code`: categorical (WMO code), so it is held constant (step
      function) across the hour it belongs to rather than interpolated.

    At the trailing edge of the fixture's date range, if the next hour's
    record is unavailable, the current hour's values are held flat for the
    remainder of the interpolation window instead of raising.
    """
    fixture_key = str(WEATHER_FIXTURE_PATH)
    records = _load_hourly_records(fixture_key)
    index = _hourly_index(fixture_key)

    midnight = DateTime.combine(date, DateTime.min.time())
    ticks: list[WeatherTick] = []

    for minute in range(shift_start_min, shift_end_min):
        abs_dt = midnight + timedelta(minutes=minute)
        hour_floor = abs_dt.replace(minute=0, second=0, microsecond=0)
        next_hour = hour_floor + timedelta(hours=1)

        idx0 = index.get(hour_floor)
        if idx0 is None:
            raise ValueError(
                f"No Open-Meteo observation for {hour_floor.isoformat()}. "
                f"Fixture covers {records[0].dt.isoformat()}..{records[-1].dt.isoformat()}."
            )
        idx1 = index.get(next_hour, idx0)  # flat extrapolation at the data edge

        r0, r1 = records[idx0], records[idx1]
        frac = (abs_dt - hour_floor).total_seconds() / 3600.0

        temp_c = _lerp(r0.temp_c, r1.temp_c, frac)
        apparent_c = _lerp(r0.apparent_c, r1.apparent_c, frac)
        wind_kmh = _lerp(r0.wind_kmh, r1.wind_kmh, frac)
        humidity_pct = _lerp(r0.humidity_pct, r1.humidity_pct, frac)

        # Hourly accumulation spread evenly across the 60 minutes it covers.
        precip_mm = r0.precip_mm / MINUTES_PER_HOUR

        # Categorical: step, never interpolate.
        weather_code = r0.weather_code

        ticks.append(
            WeatherTick(
                minute=minute,
                temp_c=temp_c,
                apparent_c=apparent_c,
                precip_mm=max(precip_mm, 0.0),
                wind_kmh=max(wind_kmh, 0.0),
                humidity_pct=min(max(humidity_pct, 0.0), 100.0),
                weather_code=weather_code,
            )
        )

    return ticks


def _hourly_rate_mm(tick: WeatherTick) -> float:
    """Reconstruct the equivalent hourly precipitation rate (mm/h) from a
    per-minute tick, for use as the intensity axis in the calibration
    mappings below. `tick.precip_mm` is already the per-minute share of the
    hour's accumulation (see `build_weather_timeline`), so this just
    reverses that split."""
    return tick.precip_mm * MINUTES_PER_HOUR


def _ramp(intensity_mm_per_hour: float, reference_mm_per_hour: float) -> float:
    """Linear ramp from 0.0 at zero intensity to 1.0 at (or beyond) the
    reference intensity. Shared shape for every rain-driven factor below;
    only the reference point and the magnitude applied to it differ."""
    if reference_mm_per_hour <= 0:
        return 0.0
    return min(max(intensity_mm_per_hour, 0.0) / reference_mm_per_hour, 1.0)


# --------------------------------------------------------------------------
# Derived impact factors
#
# None of these mappings are measured Monterrey data — there is no public
# dataset linking local rain intensity to courier speed, demand, supply or
# risk. Each is an explicit, tunable calibration dict, deliberately kept
# separate from the measured `WeatherTick` fields above.
# --------------------------------------------------------------------------

# CALIBRATION VALUE, not measured data. Published studies on rain and urban
# traffic speed (e.g. Tsapakis et al. 2013; Datla & Sharma 2008) report
# speed reductions roughly in the 10-25% range for moderate-to-heavy rain
# versus dry conditions. Modeled as a linear ramp in rain intensity, capped
# at `max_speed_reduction` once intensity reaches `reference_mm_per_hour`.
SPEED_FACTOR_CALIBRATION: dict[str, float] = {
    "reference_mm_per_hour": 10.0,
    "max_speed_reduction": 0.22,
}

# CALIBRATION VALUE, not measured data. Food-delivery platforms and industry
# commentary widely report a demand bump on rainy days (people cook/go out
# less, order in more), but no public Monterrey-specific figure exists.
# Modeled as a linear ramp, capped at `max_demand_boost`.
DEMAND_FACTOR_CALIBRATION: dict[str, float] = {
    "reference_mm_per_hour": 10.0,
    "max_demand_boost": 0.6,
}

# CALIBRATION VALUE, not measured data. Couriers on bikes/motorcycles are
# widely reported (industry anecdote, not a published Monterrey study) to
# log off or decline shifts in heavy rain at a materially higher rate than
# the demand bump above — this is the supply side of why rain drives surge
# so hard, not just the demand side. Modeled as a linear ramp, capped at
# `max_supply_reduction`.
COURIER_SUPPLY_FACTOR_CALIBRATION: dict[str, float] = {
    "reference_mm_per_hour": 10.0,
    "max_supply_reduction": 0.4,
}

# CALIBRATION VALUE, not measured data. Two independent contributors: wet
# roads (rain, ramped by intensity like the factors above) and extreme heat
# (a flat add once apparent temperature crosses the 40 C threshold that
# `WeatherTick.is_extreme_heat` encodes — the exact condition the challenge
# brief calls out explicitly).
RISK_FACTOR_CALIBRATION: dict[str, float] = {
    "rain_reference_mm_per_hour": 10.0,
    "rain_max_risk_add": 0.5,
    "extreme_heat_risk_add": 0.4,
}


def speed_factor(tick: WeatherTick) -> float:
    """Multiplier on travel *speed* from precipitation intensity.

    Apply as `effective_speed = free_flow_speed * speed_factor(tick)` (so
    travel *time* for a fixed distance scales by `1 / speed_factor(tick)`).
    Ranges from 1.0 (no rain, no slowdown) down to
    `1 - SPEED_FACTOR_CALIBRATION["max_speed_reduction"]` at or above the
    calibration's reference rain intensity. See
    `SPEED_FACTOR_CALIBRATION` for the calibration note.
    """
    cal = SPEED_FACTOR_CALIBRATION
    ramp = _ramp(_hourly_rate_mm(tick), cal["reference_mm_per_hour"])
    return 1.0 - ramp * cal["max_speed_reduction"]


def demand_factor(tick: WeatherTick) -> float:
    """Multiplier on order demand from precipitation intensity.

    1.0 = no effect; rises with rain intensity up to
    `1 + DEMAND_FACTOR_CALIBRATION["max_demand_boost"]`. See
    `DEMAND_FACTOR_CALIBRATION` for the calibration note.
    """
    cal = DEMAND_FACTOR_CALIBRATION
    ramp = _ramp(_hourly_rate_mm(tick), cal["reference_mm_per_hour"])
    return 1.0 + ramp * cal["max_demand_boost"]


def courier_supply_factor(tick: WeatherTick) -> float:
    """Multiplier on the number of couriers willing to work, from
    precipitation intensity.

    1.0 = no effect; falls below 1.0 as rain intensity rises, down to
    `1 - COURIER_SUPPLY_FACTOR_CALIBRATION["max_supply_reduction"]`. This is
    the supply-side half of the rain -> surge mechanism (see
    `COURIER_SUPPLY_FACTOR_CALIBRATION` for the calibration note); the
    demand-side half is `demand_factor`.
    """
    cal = COURIER_SUPPLY_FACTOR_CALIBRATION
    ramp = _ramp(_hourly_rate_mm(tick), cal["reference_mm_per_hour"])
    return 1.0 - ramp * cal["max_supply_reduction"]


def risk_factor(tick: WeatherTick) -> float:
    """Multiplier on incident/accident risk from wet roads and extreme heat.

    1.0 = no added risk. Rain contributes a ramped term up to
    `rain_max_risk_add`; extreme heat (`tick.is_extreme_heat`, apparent
    temperature >= 40 C) contributes a flat `extreme_heat_risk_add` on top.
    The two contributions are independent and additive (a hot, rainy minute
    gets both). See `RISK_FACTOR_CALIBRATION` for the calibration note.
    """
    cal = RISK_FACTOR_CALIBRATION
    rain_component = _ramp(_hourly_rate_mm(tick), cal["rain_reference_mm_per_hour"]) * cal["rain_max_risk_add"]
    heat_component = cal["extreme_heat_risk_add"] if tick.is_extreme_heat else 0.0
    return 1.0 + rain_component + heat_component
