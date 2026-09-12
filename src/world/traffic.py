"""Ground truth: exogenous city-wide and per-cell congestion field.

This module is ground truth. `src/agent/` must never import this module,
directly or indirectly: a policy only ever sees whatever noisy, reduced
signal `src/platform/`/`src/enrichment/` (later slices) choose to surface.
The courier cannot influence traffic and must not be able to read the exact
multipliers it experiences from this file.

HONESTY NOTE / DATA PROVENANCE (read before trusting any number below):
there is no free, public per-hour or per-edge traffic dataset for
Monterrey. TomTom's free hourly Traffic Index downloads cover only 11
cities worldwide (Dublin, Los Angeles, London, Chicago, Berlin, Bangkok,
Tokyo, Sydney, Paris, New York City, Mexico City) — Monterrey is not one
of them, and Monterrey-specific data would require TomTom's paid Area
Analytics product, which this project does not use. Uber Movement, the
other historical source for this kind of data, is effectively
discontinued. Nothing in this module may ever be labeled "Monterrey
traffic data" — it is not that.

What `HOURLY_CONGESTION_MULTIPLIER` actually is: the *timing/shape* of the
24h curve (when the peaks and troughs fall, and their relative size) is
derived from a real, measured dataset — `fixtures/raw/tomtom_mexico_city_hourly.csv`,
TomTom's free hourly series for Mexico City (4,344 hourly rows, 2025-01-01
through 2025-06-30). The reasoning for transferring *timing* from Mexico
City to Monterrey: both are Mexican metros on the same work/school/meal
schedule (comida hour, morning/evening commute windows), so the hours at
which congestion peaks and troughs are far more transferable across
Mexican cities than the *magnitude* of the congestion itself. The
*magnitude* is explicitly NOT transferred: Mexico City is structurally far
more congested than Monterrey, so the derived shape is re-scaled down to
plausible Monterrey levels by one explicit, clearly-labeled calibration
knob, `MONTERREY_CONGESTION_SCALE` — pinned by simulator sanity checks
(plausible courier earnings/hour and deliveries/hour), NOT by any measured
Monterrey traffic figure. See `derive_hourly_multiplier_table_from_csv`
for the exact, reproducible derivation of the baked table below.

Two exogenous inputs combine multiplicatively into one congestion field:

1. A city-wide, time-of-day/day-type baseline (`HOURLY_CONGESTION_MULTIPLIER`),
   whose timing/shape is derived from the real Mexico City TomTom series
   described above (morning peak, a midday dip, a more pronounced evening
   peak, a quiet night, and a flatter weekend) and whose magnitude is
   re-anchored to Monterrey via `MONTERREY_CONGESTION_SCALE`.
2. A per-H3-cell spatial "zone factor" derived from a real, auditable
   density proxy (road-network edge density from the OSMnx drive graph
   when available, otherwise commercial/DENUE restaurant density) — never
   from hardcoded "downtown"/"San Pedro" coordinates. Commercial and local
   road density are used as proxies for vehicle trip generation, which is
   the standard traffic-engineering justification for treating them as a
   congestion proxy in the absence of a real per-edge feed.

An optional, duck-typed weather signal can further scale congestion upward
when it is raining (`RAIN_CONGESTION_MULTIPLIER`). This module never
imports `weather.py` — weather producers may not exist yet, and coupling
here is deliberately loose (attribute/sequence duck-typing) so this file
never depends on that module's concrete type.

Hard invariant: every multiplier produced by this module is >= 1.0.
Congestion only ever slows a leg down relative to the free-flow time
computed in `network.py`; it never speeds one up. `travel_time_minutes` is
the single place that invariant is enforced against actual usage.

Determinism: `build_traffic_timeline` is a pure function of its arguments
plus the fixed calibration tables and the fixture-derived zone factors. It
uses no randomness and no hidden global/mutable state (the zone-factor
cache is a memoized pure computation over on-disk fixtures, not run state).
"""

from __future__ import annotations

import logging
import math
from datetime import date as Date
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.world.geo import latlon_to_cell, load_cell_index
from src.world.network import GRAPH_FIXTURE_PATH, HIGHWAY_SPEED_KMH
from src.world.timeline import TrafficTick

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
RESTAURANTS_FIXTURE_PATH = FIXTURES_DIR / "restaurants.parquet"

# Hard invariant: no multiplier produced anywhere in this module may go
# below free flow.
MIN_MULTIPLIER = 1.0


def _clamp_min(value: float) -> float:
    """Enforce the hard invariant: congestion multipliers are never < 1.0."""
    return value if value >= MIN_MULTIPLIER else MIN_MULTIPLIER


# ----------------------------------------------------------------------------
# Day type
# ----------------------------------------------------------------------------


class DayType(StrEnum):
    """Weekday vs weekend, derived from the calendar date. `Scenario` does
    carry a `day_of_week` string (e.g. "Friday"), but the weekday/weekend
    split only needs the date itself, so it is derived here rather than
    parsing that string."""

    WEEKDAY = "weekday"
    WEEKEND = "weekend"


def day_type_for_date(date: Date) -> DayType:
    """Saturday/Sunday are weekend; everything else is a weekday."""
    return DayType.WEEKEND if date.weekday() >= 5 else DayType.WEEKDAY


# ----------------------------------------------------------------------------
# A. Calibration table: city-wide baseline by (day_type, hour)
# ----------------------------------------------------------------------------

# Real, measured source series for the *timing/shape* of the daily curve.
# TomTom's free hourly Traffic Index download; Mexico City, not Monterrey
# (Monterrey is not one of the 11 cities TomTom publishes for free — see
# the module docstring). Used only to source WHEN congestion peaks/dips,
# never as a claim about Monterrey's own congestion level.
TOMTOM_MEXICO_CITY_CSV_PATH = FIXTURES_DIR / "raw" / "tomtom_mexico_city_hourly.csv"

# CALIBRATION / TUNING KNOB — NOT MEASURED DATA. Re-scales the dimensionless
# Mexico City shape down to a plausible Monterrey magnitude. Mexico City is
# structurally far more congested than Monterrey, so the raw normalized
# curve (which peaks above 2x its own series mean) is not usable directly.
# 0.40 was picked so the weekday evening peak lands at ~1.84x free flow and
# deep night sits at ~1.0x — pinned by simulator sanity checks (plausible
# courier earnings/hour and deliveries/hour for an 8h Monterrey shift), NOT
# by any measured Monterrey congestion figure.
MONTERREY_CONGESTION_SCALE = 0.40


def derive_hourly_multiplier_table_from_csv(
    csv_path: Path = TOMTOM_MEXICO_CITY_CSV_PATH,
    scale: float = MONTERREY_CONGESTION_SCALE,
) -> dict[tuple[DayType, int], float]:
    """Reproducible derivation of `HOURLY_CONGESTION_MULTIPLIER` from the
    real TomTom Mexico City hourly series.

    Method (auditable, no hand-tuned per-hour numbers):
      1. Group `Congestion level [%]` by (day_type, hour-of-day) and take
         the mean of each bucket, using the real 2025-01-01..2025-06-30
         hourly series.
      2. Normalize every bucket against the *series' own overall mean*,
         producing a dimensionless shape curve (1.0 = an average hour in
         the series; > 1.0 = more congested than average; < 1.0 = less).
      3. Anchor magnitude to Monterrey: `multiplier = 1.0 + scale *
         normalized_shape`. At `normalized_shape == 0` this is exactly free
         flow (1.0); `scale` controls how strongly the real timing/shape
         swings the multiplier away from free flow.

    This function is not called by `build_traffic_timeline` at runtime (the
    result is baked into `HOURLY_CONGESTION_MULTIPLIER` below so this
    module has no hard runtime dependency on `fixtures/raw/`) — it exists
    so the baked table is independently reproducible and auditable. Re-run
    it if the source CSV is ever regenerated.
    """
    frame = pd.read_csv(csv_path, parse_dates=["Time"])
    frame["hour"] = frame["Time"].dt.hour
    frame["day_type"] = frame["Time"].dt.weekday.apply(lambda d: DayType.WEEKEND if d >= 5 else DayType.WEEKDAY)

    overall_mean = frame["Congestion level [%]"].mean()
    bucket_means = frame.groupby(["day_type", "hour"])["Congestion level [%]"].mean()

    table: dict[tuple[DayType, int], float] = {}
    for (day_type, hour), bucket_mean in bucket_means.items():
        normalized_shape = bucket_mean / overall_mean
        table[(day_type, hour)] = _clamp_min(1.0 + scale * normalized_shape)
    return table


# BAKED, DERIVED TABLE — produced by calling
# `derive_hourly_multiplier_table_from_csv()` once against
# `fixtures/raw/tomtom_mexico_city_hourly.csv` (see that function for the
# exact method). NOT hand-invented: every value below is
# `round(1.0 + MONTERREY_CONGESTION_SCALE * normalized_shape, 3)` where
# `normalized_shape` is the real Mexico City (day_type, hour) mean
# congestion level divided by the real series' overall mean (42.398%).
# Shape sanity-checked against the source series: weekday trough ~1.00x at
# 03:00-04:00, weekday morning peak ~1.80x at 08:00, weekday midday dip
# ~1.52x-1.74x across 11:00-16:00 (lower than either peak), weekday evening
# peak ~1.84x at 18:00 (higher than the morning peak), tapering to ~1.13x
# by 23:00; weekend curve flatter and generally lower (max ~1.57x, at
# 14:00 — an afternoon rather than a sharp commute peak). 1.0 = free flow;
# e.g. 1.65 means a leg takes 65% longer than free flow.
HOURLY_CONGESTION_MULTIPLIER: dict[tuple[DayType, int], float] = {
    # --- Weekday ---
    (DayType.WEEKDAY, 0): 1.047,
    (DayType.WEEKDAY, 1): 1.017,
    (DayType.WEEKDAY, 2): 1.005,
    (DayType.WEEKDAY, 3): 1.002,
    (DayType.WEEKDAY, 4): 1.002,
    (DayType.WEEKDAY, 5): 1.130,
    (DayType.WEEKDAY, 6): 1.412,
    (DayType.WEEKDAY, 7): 1.683,  # morning peak ramp
    (DayType.WEEKDAY, 8): 1.801,  # morning peak
    (DayType.WEEKDAY, 9): 1.682,
    (DayType.WEEKDAY, 10): 1.560,
    (DayType.WEEKDAY, 11): 1.524,  # midday dip
    (DayType.WEEKDAY, 12): 1.537,
    (DayType.WEEKDAY, 13): 1.616,
    (DayType.WEEKDAY, 14): 1.736,
    (DayType.WEEKDAY, 15): 1.744,
    (DayType.WEEKDAY, 16): 1.671,  # midday dip ends
    (DayType.WEEKDAY, 17): 1.703,
    (DayType.WEEKDAY, 18): 1.842,  # evening peak (higher than morning)
    (DayType.WEEKDAY, 19): 1.833,
    (DayType.WEEKDAY, 20): 1.606,
    (DayType.WEEKDAY, 21): 1.388,
    (DayType.WEEKDAY, 22): 1.235,
    (DayType.WEEKDAY, 23): 1.128,
    # --- Weekend: flatter, generally lower, no sharp commute peaks ---
    (DayType.WEEKEND, 0): 1.112,
    (DayType.WEEKEND, 1): 1.068,
    (DayType.WEEKEND, 2): 1.037,
    (DayType.WEEKEND, 3): 1.013,
    (DayType.WEEKEND, 4): 1.003,
    (DayType.WEEKEND, 5): 1.024,
    (DayType.WEEKEND, 6): 1.072,
    (DayType.WEEKEND, 7): 1.129,
    (DayType.WEEKEND, 8): 1.208,
    (DayType.WEEKEND, 9): 1.281,
    (DayType.WEEKEND, 10): 1.341,
    (DayType.WEEKEND, 11): 1.383,
    (DayType.WEEKEND, 12): 1.440,
    (DayType.WEEKEND, 13): 1.522,
    (DayType.WEEKEND, 14): 1.571,  # modest afternoon social bump
    (DayType.WEEKEND, 15): 1.529,
    (DayType.WEEKEND, 16): 1.445,
    (DayType.WEEKEND, 17): 1.394,
    (DayType.WEEKEND, 18): 1.392,
    (DayType.WEEKEND, 19): 1.412,
    (DayType.WEEKEND, 20): 1.357,
    (DayType.WEEKEND, 21): 1.278,
    (DayType.WEEKEND, 22): 1.212,
    (DayType.WEEKEND, 23): 1.142,
}


def _baseline_multiplier(day_type: DayType, minute_of_day: int) -> float:
    """City-wide baseline for one minute, linearly interpolated between the
    containing hour's calibration value and the next hour's, so the curve
    does not jump discontinuously at hour boundaries. Each hour's tabulated
    value is treated as the value exactly at :00 of that hour."""
    hour = (minute_of_day // 60) % 24
    minute_in_hour = minute_of_day % 60
    next_hour = (hour + 1) % 24
    current = HOURLY_CONGESTION_MULTIPLIER[(day_type, hour)]
    following = HOURLY_CONGESTION_MULTIPLIER[(day_type, next_hour)]
    fraction = minute_in_hour / 60.0
    interpolated = current * (1.0 - fraction) + following * fraction
    return _clamp_min(interpolated)


# ----------------------------------------------------------------------------
# B. Spatial variation: per-cell zone factor from a density proxy
# ----------------------------------------------------------------------------
#
# CALIBRATION / TUNING KNOB — NOT MEASURED DATA. Controls how much a
# maximally-dense cell's congestion is amplified relative to city baseline.
ZONE_DENSITY_SPAN = 0.40

# Road classes treated as "local/arterial" for the road-network density
# proxy (built on top of `network.HIGHWAY_SPEED_KMH`, never redefined
# here). Motorway/trunk classes are excluded: they are through-traffic
# corridors with controlled access, not the surface-street density that
# drives local congestion and vehicle trip generation.
LOCAL_ROAD_CLASSES: frozenset[str] = frozenset(
    highway_class
    for highway_class in HIGHWAY_SPEED_KMH
    if highway_class not in {"motorway", "motorway_link", "trunk", "trunk_link", "default"}
)


def _normalized_log_density(raw_counts: dict[str, float]) -> dict[str, float]:
    """Map raw per-cell counts to a smooth [0, 1] density score via
    log1p-normalization against the densest cell. log1p (rather than a
    plain linear ratio) keeps a handful of extreme outlier cells (e.g. a
    single very dense commercial hub) from flattening every other cell's
    score toward zero."""
    log_counts = {cell: math.log1p(count) for cell, count in raw_counts.items()}
    max_log = max(log_counts.values(), default=0.0)
    if max_log <= 0.0:
        return {cell: 0.0 for cell in raw_counts}
    return {cell: log_value / max_log for cell, log_value in log_counts.items()}


def _restaurant_density_zone_factors(cell_order: list[str]) -> dict[str, float]:
    """Fallback (and primary, since the drive graph may not exist yet)
    spatial proxy: DENUE restaurant/commercial establishment density per
    H3 cell. Commercial density correlates with vehicle trip generation
    (deliveries, customer traffic, parking turnover), which is the
    standard justification for using it as a congestion proxy absent a
    real per-edge traffic feed."""
    establishments = pd.read_parquet(RESTAURANTS_FIXTURE_PATH)
    counts_by_cell = establishments["cell"].value_counts().to_dict()
    raw_counts = {cell: float(counts_by_cell.get(cell, 0.0)) for cell in cell_order}
    normalized = _normalized_log_density(raw_counts)
    return {cell: _clamp_min(1.0 + ZONE_DENSITY_SPAN * score) for cell, score in normalized.items()}


def _graph_edge_density_zone_factors(cell_order: list[str]) -> dict[str, float] | None:
    """Optional spatial proxy: local-road edge density per H3 cell from the
    real OSMnx drive graph, when the fixture is present and loadable.
    Returns None (never raises) if the graph fixture is absent or fails to
    load/parse for any reason (e.g. a concurrently-running fixture-build
    job has not finished writing it yet) — callers must fall back to the
    restaurant-density proxy in that case."""
    if not GRAPH_FIXTURE_PATH.exists():
        return None
    try:
        import osmnx as ox  # local import: only pay this cost when usable

        graph = ox.io.load_graphml(GRAPH_FIXTURE_PATH)
        cell_set = set(cell_order)
        raw_counts: dict[str, float] = {cell: 0.0 for cell in cell_order}
        for u, _v, data in graph.edges(data=True):
            highway = data.get("highway", "unclassified")
            highway_class = highway[0] if isinstance(highway, list) else highway
            if highway_class not in LOCAL_ROAD_CLASSES:
                continue
            node = graph.nodes[u]
            cell = latlon_to_cell(node["y"], node["x"])
            if cell in cell_set:
                raw_counts[cell] += 1.0
    except Exception:
        logger.warning(
            "traffic.py: failed to load/parse graph fixture at %s; "
            "falling back to restaurant-density spatial proxy.",
            GRAPH_FIXTURE_PATH,
            exc_info=True,
        )
        return None

    normalized = _normalized_log_density(raw_counts)
    return {cell: _clamp_min(1.0 + ZONE_DENSITY_SPAN * score) for cell, score in normalized.items()}


@lru_cache(maxsize=1)
def _zone_factors() -> dict[str, float]:
    """Per-cell zone factor, memoized: a pure function of the on-disk
    fixtures (cell catalog, drive graph if present, restaurant catalog),
    not of any run/scenario state. Prefers the road-network edge-density
    proxy when the graph fixture is available and loadable; otherwise
    falls back to DENUE restaurant density. Which path was taken is always
    logged, so it is auditable at runtime."""
    cell_order = list(load_cell_index()["cell"])

    graph_based = _graph_edge_density_zone_factors(cell_order)
    if graph_based is not None:
        logger.info(
            "traffic.py: spatial congestion proxy = road-network edge density "
            "from %s (%d cells).",
            GRAPH_FIXTURE_PATH,
            len(cell_order),
        )
        return graph_based

    logger.info(
        "traffic.py: spatial congestion proxy = DENUE restaurant/commercial "
        "density from %s (%d cells); drive-graph fixture at %s was absent "
        "or unusable.",
        RESTAURANTS_FIXTURE_PATH,
        len(cell_order),
        GRAPH_FIXTURE_PATH,
    )
    return _restaurant_density_zone_factors(cell_order)


# ----------------------------------------------------------------------------
# C. Weather coupling (duck-typed; never imports weather.py)
# ----------------------------------------------------------------------------

# CALIBRATION / TUNING KNOB — NOT MEASURED DATA. Multiplicative bump applied
# on top of baseline congestion whenever precipitation is present.
RAIN_CONGESTION_MULTIPLIER = 1.15


def _lookup_precip_mm(weather_timeline: Any, minute_of_day: int, tick_index: int) -> float:
    """Best-effort, duck-typed precipitation lookup. Deliberately never
    imports `weather.py` (it may not exist yet / may be written
    concurrently by another agent) and never assumes a concrete type.

    Accepted shapes, checked defensively in order:
      - `None` -> no rain signal (the safe default: 0.0, no adjustment).
      - an object exposing a callable `precip_at(minute)` -> float.
      - a sequence indexable by absolute minute-of-day, then (if that index
        is out of range) by tick offset within the shift. Each element may
        be a plain number, or an object exposing a `precip_mm` attribute,
        or an object exposing `is_raining` (bool or zero-arg callable).
    Any lookup that does not cleanly match one of these shapes is treated
    as "no rain signal" rather than raising.
    """
    if weather_timeline is None:
        return 0.0

    precip_at = getattr(weather_timeline, "precip_at", None)
    if callable(precip_at):
        try:
            return float(precip_at(minute_of_day))
        except Exception:
            pass  # fall through to sequence-style access below

    for index in (minute_of_day, tick_index):
        try:
            entry = weather_timeline[index]
        except (IndexError, TypeError, KeyError):
            continue
        if entry is None:
            continue
        if isinstance(entry, (int, float)):
            return float(entry)

        precip_mm = getattr(entry, "precip_mm", None)
        if precip_mm is not None:
            try:
                return float(precip_mm)
            except (TypeError, ValueError):
                pass

        is_raining = getattr(entry, "is_raining", None)
        if callable(is_raining):
            try:
                return 1.0 if is_raining() else 0.0
            except Exception:
                pass
        elif isinstance(is_raining, bool):
            return 1.0 if is_raining else 0.0

    return 0.0


# ----------------------------------------------------------------------------
# D. Main builder
# ----------------------------------------------------------------------------

# Minimum multiplicative gap from the city baseline for a cell to be worth
# recording as an override in `TrafficTick.cell_multipliers`. Below this,
# the cell is indistinguishable from baseline for gameplay/analysis
# purposes and is simply omitted (per `TrafficTick`'s own contract: a cell
# absent from the override map uses `city_multiplier`).
CELL_OVERRIDE_DEVIATION_THRESHOLD = 0.02


def build_traffic_timeline(
    date: Date,
    shift_start_min: int,
    shift_end_min: int,
    weather_timeline: Any = None,
) -> list[TrafficTick]:
    """Build one `TrafficTick` per minute in `[shift_start_min, shift_end_min)`.

    Pure function of `(date, shift_start_min, shift_end_min,
    weather_timeline)` plus the fixed calibration tables and the
    fixture-derived per-cell zone factors — no randomness, no hidden
    global/mutable run state. There is no sibling `build_*_timeline`
    function elsewhere in the codebase yet to establish an RNG-argument
    convention, and no randomness is needed here, so this function takes
    no `Scenario`/RNG argument.

    `weather_timeline` is optional and loosely typed on purpose (see
    `_lookup_precip_mm`): this module must never import `weather.py`.
    """
    day_type = day_type_for_date(date)
    zone_factors = _zone_factors()

    ticks: list[TrafficTick] = []
    for tick_index, minute in enumerate(range(shift_start_min, shift_end_min)):
        minute_of_day = minute % 1440
        baseline = _baseline_multiplier(day_type, minute_of_day)

        precip_mm = _lookup_precip_mm(weather_timeline, minute_of_day, tick_index)
        rain_factor = RAIN_CONGESTION_MULTIPLIER if precip_mm > 0.0 else 1.0

        city_multiplier = _clamp_min(baseline * rain_factor)

        cell_multipliers: dict[str, float] = {}
        for cell, zone_factor in zone_factors.items():
            effective = _clamp_min(baseline * zone_factor * rain_factor)
            if abs(effective - city_multiplier) > CELL_OVERRIDE_DEVIATION_THRESHOLD:
                cell_multipliers[cell] = effective

        ticks.append(
            TrafficTick(
                minute=minute,
                city_multiplier=city_multiplier,
                cell_multipliers=cell_multipliers,
            )
        )

    return ticks


# ----------------------------------------------------------------------------
# E. Travel time application
# ----------------------------------------------------------------------------


def travel_time_minutes(base_minutes: float, tick: TrafficTick, cell: str) -> float:
    """The single place congestion is applied to a free-flow travel time.

    Looks up the effective multiplier for `cell` (its override in `tick`,
    else `tick.city_multiplier`) and scales `base_minutes` by it. Hard
    invariant, enforced here regardless of what produced `tick`: the
    result is never less than `base_minutes` (multipliers are clamped to
    >= 1.0). Only ever touches minutes — never distances; km and minutes
    stay strictly separate, as everywhere else in this simulator.
    """
    multiplier = _clamp_min(tick.multiplier_for(cell))
    return base_minutes * multiplier
