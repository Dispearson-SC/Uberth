"""Ground truth: exogenous city-wide and per-cell congestion field.

This module is ground truth. `src/agent/` must never import this module,
directly or indirectly: a policy only ever sees whatever noisy, reduced
signal `src/platform/`/`src/enrichment/` (later slices) choose to surface.
The courier cannot influence traffic and must not be able to read the exact
multipliers it experiences from this file.

HONESTY NOTE / DATA PROVENANCE (read before trusting any number below):
this module used to borrow its temporal shape from Mexico City because no
Monterrey-specific traffic dataset existed. That is no longer true for
weekdays. We purchased TomTom's paid Traffic Stats Area Analysis product
for our exact operating-area polygon (Monterrey, San Pedro, San Nicolas,
Guadalupe), July 2026, 24 weekday hourly time sets (`WD-00`..`WD-23`,
Monday-Friday). That raw fixture
(`fixtures/raw/tomtom_monterrey_areaanalysis.json`, ~2.63 GB, never
committed to git and never read at simulator runtime) was reduced by
`scripts/derive_monterrey_traffic_profile.py` — a single streaming
(`ijson`) pass computing, per hour, the distance-weighted harmonic-mean
network speed across ~358k real Monterrey road segments — into the
compact, committed, human-auditable artifact this module actually loads:
`fixtures/monterrey_hourly_profile.csv` (24 rows). See that script's
docstring for the exact aggregation method, the low-sample-size filter,
and the empirical free-flow reference-hour choice.

What is and is not measured, precisely:

- WEEKDAY shape AND magnitude: 100% measured Monterrey data (the CSV
  above). Nothing about the weekday curve is transferred from Mexico City
  any more, and there is no re-scaling knob on it: the congestion
  multiplier is directly `free_flow_hour_speed / hour_speed`, both real
  network speeds.
- WEEKEND shape: still TRANSFERRED, not measured. TomTom's Area Analysis
  trial only covered 24 time sets, and we spent all 24 on the weekday
  curve (the shape that matters most for a courier simulator, since most
  shifts are weekdays). The weekend curve is therefore built by taking the
  real Monterrey weekday curve and reshaping it with the
  weekend-vs-weekday RATIO measured in Mexico City's free hourly TomTom
  series (`fixtures/raw/tomtom_mexico_city_hourly.csv`, 4,344 hourly rows,
  2025-01-01 through 2025-06-30): `derive_weekend_weekday_ratio_table_from_csv`
  computes, per hour, how much less (or more) congested Mexico City is on
  a weekend vs. a weekday at that same hour, and that ratio is applied to
  how far *above free flow* the real Monterrey weekday number sits. The
  reasoning for transferring this specific ratio: both are Mexican metros
  on the same work/school/meal schedule, so *how much weekend traffic
  eases off relative to a weekday* is far more transferable across Mexican
  cities than an absolute congestion level would be. Nothing in this
  module may be labeled "measured Monterrey weekend data" — it is not
  that; it is real Monterrey weekday data reshaped by a transferred ratio.

Two exogenous inputs combine multiplicatively into one congestion field:

1. A city-wide, time-of-day/day-type baseline (`HOURLY_CONGESTION_MULTIPLIER`),
   whose weekday numbers are measured Monterrey network speeds and whose
   weekend numbers are those same measured weekday numbers reshaped by the
   Mexico City weekend/weekday ratio described above.
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
cache is a memoized pure computation over on-disk fixtures, not run state;
`HOURLY_CONGESTION_MULTIPLIER` is likewise a pure, deterministic function
of the two small on-disk source CSVs, loaded once at import time).
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

# Compact, committed artifact produced by
# `scripts/derive_monterrey_traffic_profile.py` from the raw ~2.63 GB TomTom
# Area Analysis fixture (see module docstring). This is the ONLY traffic
# input read at runtime — the raw fixture is never opened outside that
# offline derivation script. 24 rows: hour, harmonic_speed_kmh,
# congestion_multiplier, sample_size, total_distance_km, segment_count.
MONTERREY_HOURLY_PROFILE_CSV_PATH = FIXTURES_DIR / "monterrey_hourly_profile.csv"

# Real, measured source series used ONLY to derive the weekend-vs-weekday
# RATIO (see module docstring — the weekday curve above needs no help from
# this any more). TomTom's free hourly Traffic Index download for Mexico
# City, not Monterrey (Monterrey is not one of the 11 cities TomTom
# publishes for free).
TOMTOM_MEXICO_CITY_CSV_PATH = FIXTURES_DIR / "raw" / "tomtom_mexico_city_hourly.csv"


def load_monterrey_weekday_multiplier_table(
    csv_path: Path = MONTERREY_HOURLY_PROFILE_CSV_PATH,
) -> dict[int, float]:
    """Load the real, measured Monterrey weekday congestion multiplier per
    hour straight from the compact derived CSV — no re-scaling, no
    transformation. `congestion_multiplier` in that file is already
    `free_flow_hour_speed / hour_speed` computed from real network speeds
    by `scripts/derive_monterrey_traffic_profile.py`."""
    frame = pd.read_csv(csv_path)
    table = {int(row.hour): _clamp_min(float(row.congestion_multiplier)) for row in frame.itertuples()}
    if sorted(table) != list(range(24)):
        raise ValueError(f"Expected hours 0..23 in {csv_path}, got {sorted(table)}")
    return table


# CALIBRATION / NUMERICAL-STABILITY KNOB — NOT a magnitude-anchoring knob
# like the retired `MONTERREY_CONGESTION_SCALE`. Caps the raw Mexico City
# weekend/weekday ratio from above only. At true overnight hours (roughly
# 00:00-04:00) Mexico City's own WEEKDAY congestion level is itself close to
# 0% (a near-empty road network), so dividing weekend-by-weekday there is a
# division of two small, noisy numbers: the raw ratio spikes as high as
# ~7.7x (see `derive_weekend_weekday_ratio_table_from_csv` docstring for the
# measured per-hour ratios). That spike is a real, measured Mexico City
# phenomenon (weekend nightlife keeps its own overnight roads busier, in
# *relative* terms, than a dead-quiet weekday night) — but transferring an
# uncapped 7.7x multiplicatively onto the real Monterrey weekday curve,
# where the same overnight hours already carry a non-trivial measured extra
# (unlike Mexico City's near-zero weekday baseline there), would inflate
# Monterrey's weekend early-morning multiplier close to or above its own
# measured weekday midday peak — implausible for a simulator whose whole
# point is plausible courier economics. The low end of the ratio (as low as
# ~0.18 during the weekday-only commute peak, where weekend traffic
# genuinely collapses because nobody commutes) is a believable signal, not
# noise, and is intentionally left uncapped.
MAX_TRANSFERABLE_WEEKEND_RATIO = 2.0


def derive_weekend_weekday_ratio_table_from_csv(
    csv_path: Path = TOMTOM_MEXICO_CITY_CSV_PATH,
    max_ratio: float = MAX_TRANSFERABLE_WEEKEND_RATIO,
) -> dict[int, float]:
    """Reproducible derivation of the weekend/weekday congestion RATIO per
    hour from the real TomTom Mexico City hourly series. This ratio is the
    only thing Mexico City still contributes to this module (see the
    module docstring's honesty note): TomTom's Area Analysis trial for
    Monterrey only covered 24 weekday time sets, so there is no measured
    Monterrey weekend curve to load.

    Method (auditable, no hand-tuned per-hour numbers): group real
    `Congestion level [%]` by (day_type, hour-of-day), take each bucket's
    mean over the full 2025-01-01..2025-06-30 series, and divide the
    weekend mean by the weekday mean at that same hour, capped above at
    `max_ratio` (see that constant's comment for why only the upper bound
    is capped). A ratio of 0.85 at some hour means Mexico City's weekend
    congestion at that hour runs 85% of its weekday congestion; that same
    0.85 is what `build_hourly_congestion_multiplier_table` applies to the
    real Monterrey weekday-over-free-flow amount to produce a weekend
    number.
    """
    frame = pd.read_csv(csv_path, parse_dates=["Time"])
    frame["hour"] = frame["Time"].dt.hour
    frame["day_type"] = frame["Time"].dt.weekday.apply(lambda d: DayType.WEEKEND if d >= 5 else DayType.WEEKDAY)
    bucket_means = frame.groupby(["day_type", "hour"])["Congestion level [%]"].mean()

    ratio_table: dict[int, float] = {}
    for hour in range(24):
        weekday_pct = bucket_means[(DayType.WEEKDAY, hour)]
        weekend_pct = bucket_means[(DayType.WEEKEND, hour)]
        ratio_table[hour] = min(weekend_pct / weekday_pct, max_ratio)
    return ratio_table


def build_hourly_congestion_multiplier_table() -> dict[tuple[DayType, int], float]:
    """Build the full (day_type, hour) -> multiplier table used at runtime.

    - Weekday: the real Monterrey number, unmodified
      (`load_monterrey_weekday_multiplier_table`).
    - Weekend: the real Monterrey weekday number reshaped by the Mexico
      City weekend/weekday ratio, applied to the *amount above free flow*
      rather than the raw multiplier (so a ratio of 1.0 exactly reproduces
      the weekday number, and a ratio < 1.0 relaxes the multiplier back
      toward free flow rather than toward zero):

          weekend(h) = 1.0 + (weekday(h) - 1.0) * ratio(h)

    Loaded fresh (not baked into a hardcoded literal) so this table always
    reflects whatever is currently on disk in the two small, committed
    source CSVs — both are cheap to read (24 and 4,344 rows respectively),
    so there is no performance reason to bake a snapshot, and baking one
    would risk silently drifting from the CSVs if either is regenerated.
    """
    weekday_table = load_monterrey_weekday_multiplier_table()
    weekend_ratio_table = derive_weekend_weekday_ratio_table_from_csv()

    table: dict[tuple[DayType, int], float] = {}
    for hour in range(24):
        weekday_multiplier = weekday_table[hour]
        table[(DayType.WEEKDAY, hour)] = weekday_multiplier
        weekend_multiplier = 1.0 + (weekday_multiplier - 1.0) * weekend_ratio_table[hour]
        table[(DayType.WEEKEND, hour)] = _clamp_min(weekend_multiplier)
    return table


# Loaded once at import time from the two small source CSVs above (see
# `build_hourly_congestion_multiplier_table`). Weekday values are real
# measured Monterrey network speeds; weekend values are those same real
# weekday values reshaped by the Mexico City weekend/weekday ratio. 1.0 =
# free flow; e.g. 1.65 means a leg takes 65% longer than free flow.
HOURLY_CONGESTION_MULTIPLIER: dict[tuple[DayType, int], float] = build_hourly_congestion_multiplier_table()


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
