"""Ground truth: exogenous disruptive-event timeline (crashes, closures,
checkpoints, surge shocks, mass events, kitchen backlogs).

This module is ground truth. It is one of the exogenous producers written
against the coordination contract declared in `timeline.py` (`Event`,
`EventType`). `src/agent/` must never import this module: a policy only
ever sees a reduced, possibly-noisy view of these events surfaced through
`src/platform/`/`src/enrichment/` (later slices), never this ground-truth
timeline directly.

Determinism, spelled out because it is load-bearing: every random draw in
this module comes from exactly one RNG, the `"incidents"` stream returned
by `src.world.scenario.rng_streams(seed)`. Never `random`, never
`np.random.seed`, never any other named stream. Given the same
`(scenario_seed, date, shift_start_min, shift_end_min)` and the same
on-disk graph fixture state, `build_events_timeline` returns byte-identical
output every time; a different seed produces a different timeline.

Calibration disclosure: there is no reliable public incident dataset for
Monterrey (no per-street crash feed, no closure registry, no checkpoint
schedule API). Every rate, duration, and effect magnitude below is a
calibration knob tuned by hand for a plausible demo, collected in
module-level dicts/tuples so they are auditable and easy to retune. They
are explicitly NOT measured data. Two exceptions worth calling out:

  * CHECKPOINT timing (Thursday-Sunday, roughly 22:00-04:00, on major
    avenues) is modelled from the widely reported, publicly known pattern
    of Nuevo Leon "alcoholimetro" checkpoints. No dataset backs this
    either -- it is a hand-authored schedule approximation, not a fitted
    distribution.
  * Stadium coordinates (Estadio BBVA, Estadio Universitario) are real,
    approximate public landmark coordinates, not surveyed points.

Graph degradation policy: `fixtures/monterrey_graph.graphml` may not exist
yet (it is built by a separate background job). This module never builds
it and never blocks waiting for it -- it only ever attempts to *load* an
existing fixture (see `_try_load_graph`). When the graph is present,
CRASH and STREET_CLOSURE events use real `(u, v)` node-id edges sampled
from it, which is what lets `network.py`'s `TravelMatrix.close_streets`
do a real partial recompute. When the graph is absent, those same event
types degrade to a `cells`-scoped locator (a real H3 cell, either from the
cached cell catalog or a deterministic city-bounding-box sample) instead
of a fabricated edge id: `Event` documents "exactly one of edges, cells,
or point_radius_km" as its locator convention, and fabricating a
plausible-looking-but-fake edge id would silently violate that contract
for any downstream code that branches on `event.edges`. The degraded
event's `event_id` carries an explicit `-no_graph_fallback` suffix and the
degradation path is logged via the standard `logging` module, so which
path was taken is always visible.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

from src.world import geo
from src.world.network import GRAPH_FIXTURE_PATH, TravelMatrix
from src.world.scenario import rng_streams
from src.world.timeline import Event, EventType

logger = logging.getLogger(__name__)

# ============================================================================
# CALIBRATION KNOBS -- hand-tuned, NOT measured. See module docstring.
# ============================================================================

# --- CRASH -------------------------------------------------------------
CRASH_RATE_PER_HOUR = 0.15
CRASH_DURATION_MIN_RANGE = (20, 90)
CRASH_SPEED_MULT_RANGE = (0.15, 0.25)  # "around 0.2"
CRASH_EDGE_COUNT_RANGE = (1, 3)
CRASH_DETECT_OFFSET_MIN_RANGE = (4, 10)  # detected LATE
CRASH_DETECT_RADIUS_KM = 2.0
CRASH_CONFIDENCE = 0.7  # crowd-sourced-style report, not a verified feed

# Congestion propagation to edges adjacent to the crash. `Event` carries a
# single `speed_mult` per event, so a graded effect (severe on the blocked
# edge, lighter on the edges feeding into it) is modelled as a *second*,
# separate spillover Event sharing the same time window, rather than
# stretching one Event's single multiplier across two different severities.
CRASH_SPILLOVER_EDGE_COUNT_RANGE = (0, 2)
CRASH_SPILLOVER_SPEED_MULT_RANGE = (0.5, 0.75)
CRASH_SPILLOVER_DURATION_FRACTION = 0.5  # spillover clears before the crash itself

# --- STREET_CLOSURE ------------------------------------------------------
STREET_CLOSURE_RATE_PER_HOUR = 0.08
STREET_CLOSURE_DURATION_MIN_RANGE = (30, 240)  # 30 min .. 4 h
STREET_CLOSURE_EDGE_COUNT_RANGE = (1, 2)
# Two flavours, chosen per occurrence.
STREET_CLOSURE_ANNOUNCED_PROBABILITY = 0.6  # scheduled roadworks are more common
STREET_CLOSURE_ANNOUNCED_DETECT_OFFSET_MIN_RANGE = (-90, -30)  # announced ahead of time
STREET_CLOSURE_ANNOUNCED_CONFIDENCE = 0.9
STREET_CLOSURE_SUDDEN_DETECT_OFFSET_MIN_RANGE = (2, 10)  # sudden blockage / protest
STREET_CLOSURE_SUDDEN_CONFIDENCE = 0.6
STREET_CLOSURE_DETECT_RADIUS_KM = 3.0

# --- CHECKPOINT (reten / alcoholimetro) ---------------------------------
# Modelled from the publicly known pattern, NOT from a dataset (see module
# docstring): Thursday through Sunday, roughly 22:00-04:00, on major
# avenues. Weekday numbering follows `date.weekday()`: Monday=0 .. Sunday=6.
CHECKPOINT_WEEKDAYS: frozenset[int] = frozenset({3, 4, 5, 6})  # Thu, Fri, Sat, Sun
CHECKPOINT_WINDOW_START_MIN_OF_DAY = 22 * 60  # 22:00
CHECKPOINT_WINDOW_DURATION_MIN = 6 * 60  # through 04:00 the next day
CHECKPOINT_RATE_PER_NIGHT = 2.5  # expected checkpoints citywide, per applicable night
CHECKPOINT_SETUP_DURATION_MIN_RANGE = (45, 180)  # how long one checkpoint stays in place
CHECKPOINT_FIXED_DELAY_MIN_RANGE = (5, 20)  # hard stop, not a speed reduction
CHECKPOINT_RADIUS_KM = 0.3  # footprint used as the event's own point_radius_km
CHECKPOINT_DETECT_RADIUS_KM = 0.5  # perceived only when practically on top of it
CHECKPOINT_CONFIDENCE = 1.0  # unambiguous once you are there

# Major avenues where checkpoints commonly appear, per public knowledge.
# (name, lat, lon) -- approximate landmark coordinates, not surveyed.
MAJOR_AVENUES: list[tuple[str, float, float]] = [
    ("Av. Constitucion", 25.6690, -100.3098),
    ("Av. Gonzalitos", 25.6961, -100.3564),
    ("Av. Lazaro Cardenas", 25.6280, -100.3350),
    ("Av. Eugenio Garza Sada", 25.6484, -100.2907),
    ("Av. Revolucion", 25.6889, -100.3465),
    ("Av. Ruiz Cortines", 25.6395, -100.2780),
    ("Av. Vasconcelos", 25.6520, -100.4030),
    ("Av. Morones Prieto", 25.6690, -100.3350),
]

# --- SURGE_WINDOW (exogenous demand shock) ------------------------------
# NOTE: `surge.py` (owned by another agent) is the mechanism-based
# demand/supply surge model. This event is an exogenous *shock* layered on
# top of it -- a sudden localized spike, not a replacement for the
# mechanism. Do not import `surge.py` here.
SURGE_WINDOW_RATE_PER_HOUR = 0.10
SURGE_WINDOW_DURATION_MIN_RANGE = (20, 60)
SURGE_WINDOW_DEMAND_MULT_RANGE = (1.5, 3.0)
SURGE_WINDOW_PAYOUT_MULT_RANGE = (1.2, 2.0)
SURGE_WINDOW_CELL_RING_RANGE = (1, 2)  # k-ring around a randomly seeded cell
SURGE_WINDOW_DETECT_OFFSET_MIN = 0
SURGE_WINDOW_DETECT_RADIUS_KM = 5.0
SURGE_WINDOW_CONFIDENCE = 0.8

# --- MASS_EVENT (stadium / concert) --------------------------------------
# Real, approximate public landmark coordinates (NOT surveyed).
MASS_EVENT_VENUES: list[tuple[str, float, float]] = [
    ("Estadio BBVA", 25.6692, -100.2453),  # Guadalupe, NL
    ("Estadio Universitario", 25.7256, -100.3147),  # San Nicolas de los Garza, NL
]
MASS_EVENT_PROBABILITY_PER_SHIFT = 0.05  # rare: most shifts have none at all
MASS_EVENT_DURATION_MIN_RANGE = (180, 300)  # pre-event buildup through post-event egress
MASS_EVENT_DEMAND_MULT_RANGE = (2.0, 4.0)
MASS_EVENT_SPEED_MULT_RANGE = (0.3, 0.5)  # severe congestion around the venue
MASS_EVENT_RADIUS_KM = 2.5
MASS_EVENT_DETECT_OFFSET_MIN_RANGE = (-4320, -1440)  # announced 1-3 days ahead

# --- KITCHEN_BACKLOG -------------------------------------------------------
KITCHEN_BACKLOG_RATE_PER_HOUR = 0.10
KITCHEN_BACKLOG_DURATION_MIN_RANGE = (30, 120)
KITCHEN_BACKLOG_FIXED_DELAY_MIN_RANGE = (8, 25)  # added prep wait
KITCHEN_BACKLOG_CLUSTER_RING = 1  # neighboring cells swept into the same cluster
KITCHEN_BACKLOG_DETECT_RADIUS_KM = 0.15  # discovered only on arrival at the restaurant

# --- RAIN_ONSET / EXTREME_HEAT (opt-in stubs only; weather.py owns these) --
# Emitted ONLY when explicitly requested via `include_rain_events` /
# `include_heat_events`, both default False. `weather.py` derives real
# per-minute weather from measured Open-Meteo archive data; emitting these
# by default here would double-count that signal. These stubs exist only so
# an early caller without `weather.py` wired up yet can exercise the event
# type end to end.
WEATHER_STUB_RAIN_RATE_PER_HOUR = 0.05
WEATHER_STUB_RAIN_DURATION_MIN_RANGE = (15, 60)
WEATHER_STUB_RAIN_SPEED_MULT = 0.7
WEATHER_STUB_HEAT_RATE_PER_HOUR = 0.02
WEATHER_STUB_HEAT_DURATION_MIN_RANGE = (60, 180)
WEATHER_STUB_HEAT_SUPPLY_MULT = 0.8  # fewer couriers willing to ride in extreme heat

# --- Fallback cell sampling (used when no graph, and for cell-scoped types) -
# Rough bounding box over the four operating municipalities, used only if
# even the cached cell catalog (`fixtures/cells.parquet`) is unavailable.
FALLBACK_BBOX_LAT_RANGE = (25.55, 25.80)
FALLBACK_BBOX_LON_RANGE = (-100.45, -100.15)

# Downtown Monterrey (Macroplaza) -- real, approximate landmark coordinate
# used to anchor the hand-authored demo timeline.
DEMO_DOWNTOWN_LAT = 25.6714
DEMO_DOWNTOWN_LON = -100.3095


# ============================================================================
# Small RNG helpers (all draws go through these, and only ever consume the
# `incidents` stream passed in by the caller).
# ============================================================================


def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    """Random integer in [low, high], both ends inclusive."""
    return int(rng.integers(low, high + 1))


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _sample_random_cell(rng: np.random.Generator) -> str:
    """A real H3 cell: from the cached catalog if it exists, else a
    deterministic sample within the operating area's rough bounding box."""
    try:
        cell_index = geo.load_cell_index()
        idx = int(rng.integers(0, len(cell_index)))
        return str(cell_index.iloc[idx]["cell"])
    except FileNotFoundError:
        lat = _uniform(rng, *FALLBACK_BBOX_LAT_RANGE)
        lon = _uniform(rng, *FALLBACK_BBOX_LON_RANGE)
        return geo.latlon_to_cell(lat, lon)


# ============================================================================
# Graph loading -- read-only. Never builds, never blocks on the background
# fixture job; see the module docstring's "Graph degradation policy".
# ============================================================================


def _try_load_graph(path: Path = GRAPH_FIXTURE_PATH) -> nx.MultiDiGraph | None:
    """Load the cached OSMnx drive graph if the fixture already exists.

    Never triggers a build (that is `network.get_or_build_graph`'s job and
    requires a live OSM download). Returns None -- and logs which path was
    taken -- when the fixture is missing or fails to load.
    """
    if not path.exists():
        logger.info(
            "events.py: graph fixture not found at %s; degrading CRASH/"
            "STREET_CLOSURE to cell-scoped fallback events.",
            path,
        )
        return None
    try:
        graph = ox.io.load_graphml(path)
    except Exception:
        logger.warning(
            "events.py: failed to load graph fixture at %s; degrading to "
            "cell-scoped fallback events.",
            path,
            exc_info=True,
        )
        return None
    logger.info(
        "events.py: loaded graph fixture (%d nodes, %d edges) from %s; "
        "using real edge-scoped events.",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        path,
    )
    return graph


def _node_latlon(graph: nx.MultiDiGraph, node: int) -> tuple[float, float]:
    data = graph.nodes[node]
    return float(data["y"]), float(data["x"])


# ============================================================================
# Per-type generators. Each takes the shared `incidents` RNG and appends its
# own events; call order inside `build_events_timeline` is fixed so the RNG
# stream is consumed identically on every run for a given seed.
# ============================================================================


def _generate_crashes(
    rng: np.random.Generator,
    shift_start_min: int,
    shift_end_min: int,
    graph: nx.MultiDiGraph | None,
) -> list[Event]:
    hours = (shift_end_min - shift_start_min) / 60.0
    count = int(rng.poisson(CRASH_RATE_PER_HOUR * hours))
    edge_list = list(graph.edges(keys=True)) if graph is not None else []

    events: list[Event] = []
    for i in range(count):
        start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
        duration_min = _randint_inclusive(rng, *CRASH_DURATION_MIN_RANGE)
        speed_mult = _uniform(rng, *CRASH_SPEED_MULT_RANGE)
        detect_offset = _randint_inclusive(rng, *CRASH_DETECT_OFFSET_MIN_RANGE)
        n_primary = _randint_inclusive(rng, *CRASH_EDGE_COUNT_RANGE)

        if edge_list:
            idx = rng.choice(len(edge_list), size=min(n_primary, len(edge_list)), replace=False)
            primary_edges = [(edge_list[j][0], edge_list[j][1]) for j in np.atleast_1d(idx)]
            events.append(
                Event(
                    event_id=f"crash-{i:04d}",
                    type=EventType.CRASH,
                    start_min=start_min,
                    duration_min=duration_min,
                    edges=primary_edges,
                    speed_mult=speed_mult,
                    detect_offset_min=detect_offset,
                    detect_radius_km=CRASH_DETECT_RADIUS_KM,
                    confidence=CRASH_CONFIDENCE,
                )
            )

            n_spillover = _randint_inclusive(rng, *CRASH_SPILLOVER_EDGE_COUNT_RANGE)
            if n_spillover > 0:
                touched_nodes = {n for edge in primary_edges for n in edge}
                candidate_edges = [
                    (u, v)
                    for u, v, _k in edge_list
                    if (u in touched_nodes or v in touched_nodes) and (u, v) not in primary_edges
                ]
                if candidate_edges:
                    idx2 = rng.choice(
                        len(candidate_edges), size=min(n_spillover, len(candidate_edges)), replace=False
                    )
                    spillover_edges = [candidate_edges[j] for j in np.atleast_1d(idx2)]
                    events.append(
                        Event(
                            event_id=f"crash-{i:04d}-spillover",
                            type=EventType.CRASH,
                            start_min=start_min,
                            duration_min=max(5, int(duration_min * CRASH_SPILLOVER_DURATION_FRACTION)),
                            edges=spillover_edges,
                            speed_mult=_uniform(rng, *CRASH_SPILLOVER_SPEED_MULT_RANGE),
                            detect_offset_min=detect_offset,
                            detect_radius_km=CRASH_DETECT_RADIUS_KM,
                            confidence=CRASH_CONFIDENCE,
                        )
                    )
        else:
            cell = _sample_random_cell(rng)
            events.append(
                Event(
                    event_id=f"crash-{i:04d}-no_graph_fallback",
                    type=EventType.CRASH,
                    start_min=start_min,
                    duration_min=duration_min,
                    cells=[cell],
                    speed_mult=speed_mult,
                    detect_offset_min=detect_offset,
                    detect_radius_km=CRASH_DETECT_RADIUS_KM,
                    confidence=CRASH_CONFIDENCE,
                )
            )
    return events


def _generate_street_closures(
    rng: np.random.Generator,
    shift_start_min: int,
    shift_end_min: int,
    graph: nx.MultiDiGraph | None,
) -> list[Event]:
    hours = (shift_end_min - shift_start_min) / 60.0
    count = int(rng.poisson(STREET_CLOSURE_RATE_PER_HOUR * hours))
    edge_list = list(graph.edges(keys=True)) if graph is not None else []

    events: list[Event] = []
    for i in range(count):
        start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
        duration_min = _randint_inclusive(rng, *STREET_CLOSURE_DURATION_MIN_RANGE)
        n_edges = _randint_inclusive(rng, *STREET_CLOSURE_EDGE_COUNT_RANGE)
        announced = bool(rng.uniform() < STREET_CLOSURE_ANNOUNCED_PROBABILITY)

        if announced:
            flavor = "roadworks"
            detect_offset = _randint_inclusive(rng, *STREET_CLOSURE_ANNOUNCED_DETECT_OFFSET_MIN_RANGE)
            confidence = STREET_CLOSURE_ANNOUNCED_CONFIDENCE
        else:
            flavor = "sudden_blockage"
            detect_offset = _randint_inclusive(rng, *STREET_CLOSURE_SUDDEN_DETECT_OFFSET_MIN_RANGE)
            confidence = STREET_CLOSURE_SUDDEN_CONFIDENCE

        if edge_list:
            idx = rng.choice(len(edge_list), size=min(n_edges, len(edge_list)), replace=False)
            edges = [(edge_list[j][0], edge_list[j][1]) for j in np.atleast_1d(idx)]
            events.append(
                Event(
                    event_id=f"street_closure-{i:04d}-{flavor}",
                    type=EventType.STREET_CLOSURE,
                    start_min=start_min,
                    duration_min=duration_min,
                    edges=edges,
                    close_edges=True,
                    detect_offset_min=detect_offset,
                    detect_radius_km=STREET_CLOSURE_DETECT_RADIUS_KM,
                    confidence=confidence,
                )
            )
        else:
            cell = _sample_random_cell(rng)
            events.append(
                Event(
                    event_id=f"street_closure-{i:04d}-{flavor}-no_graph_fallback",
                    type=EventType.STREET_CLOSURE,
                    start_min=start_min,
                    duration_min=duration_min,
                    cells=[cell],
                    close_edges=True,
                    detect_offset_min=detect_offset,
                    detect_radius_km=STREET_CLOSURE_DETECT_RADIUS_KM,
                    confidence=confidence,
                )
            )
    return events


def _generate_checkpoints(
    rng: np.random.Generator,
    date: Date,
    shift_start_min: int,
    shift_end_min: int,
) -> list[Event]:
    """See module docstring: this schedule is modelled from the publicly
    known Thursday-Sunday, ~22:00-04:00 pattern of Nuevo Leon checkpoints
    ("reten"/"alcoholimetro"), NOT from any dataset."""
    events: list[Event] = []
    counter = 0

    first_day = shift_start_min // 1440
    last_day = (shift_end_min - 1) // 1440
    for day_index in range(first_day, last_day + 1):
        day_date = date + timedelta(days=day_index)
        if day_date.weekday() not in CHECKPOINT_WEEKDAYS:
            continue

        window_start = day_index * 1440 + CHECKPOINT_WINDOW_START_MIN_OF_DAY
        window_end = window_start + CHECKPOINT_WINDOW_DURATION_MIN
        overlap_start = max(window_start, shift_start_min)
        overlap_end = min(window_end, shift_end_min)
        if overlap_start >= overlap_end:
            continue

        count = int(rng.poisson(CHECKPOINT_RATE_PER_NIGHT))
        for _ in range(count):
            start_min = _randint_inclusive(rng, overlap_start, overlap_end - 1)
            duration_min = _randint_inclusive(rng, *CHECKPOINT_SETUP_DURATION_MIN_RANGE)
            _name, lat, lon = MAJOR_AVENUES[int(rng.integers(0, len(MAJOR_AVENUES)))]
            fixed_delay = _uniform(rng, *CHECKPOINT_FIXED_DELAY_MIN_RANGE)
            events.append(
                Event(
                    event_id=f"checkpoint-{counter:04d}",
                    type=EventType.CHECKPOINT,
                    start_min=start_min,
                    duration_min=duration_min,
                    point_lat=lat,
                    point_lon=lon,
                    point_radius_km=CHECKPOINT_RADIUS_KM,
                    fixed_delay_min=fixed_delay,
                    detect_offset_min=0,
                    detect_radius_km=CHECKPOINT_DETECT_RADIUS_KM,
                    confidence=CHECKPOINT_CONFIDENCE,
                )
            )
            counter += 1
    return events


def _generate_surge_windows(
    rng: np.random.Generator,
    shift_start_min: int,
    shift_end_min: int,
) -> list[Event]:
    hours = (shift_end_min - shift_start_min) / 60.0
    count = int(rng.poisson(SURGE_WINDOW_RATE_PER_HOUR * hours))

    events: list[Event] = []
    for i in range(count):
        start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
        duration_min = _randint_inclusive(rng, *SURGE_WINDOW_DURATION_MIN_RANGE)
        seed_cell = _sample_random_cell(rng)
        ring = _randint_inclusive(rng, *SURGE_WINDOW_CELL_RING_RANGE)
        cells = geo.cell_neighbors(seed_cell, k=ring, include_self=True)
        events.append(
            Event(
                event_id=f"surge_window-{i:04d}",
                type=EventType.SURGE_WINDOW,
                start_min=start_min,
                duration_min=duration_min,
                cells=cells,
                demand_mult=_uniform(rng, *SURGE_WINDOW_DEMAND_MULT_RANGE),
                payout_mult=_uniform(rng, *SURGE_WINDOW_PAYOUT_MULT_RANGE),
                detect_offset_min=SURGE_WINDOW_DETECT_OFFSET_MIN,
                detect_radius_km=SURGE_WINDOW_DETECT_RADIUS_KM,
                confidence=SURGE_WINDOW_CONFIDENCE,
            )
        )
    return events


def _generate_mass_events(
    rng: np.random.Generator,
    shift_start_min: int,
    shift_end_min: int,
) -> list[Event]:
    # One Bernoulli draw per shift (mass events are rare enough that a
    # per-shift probability is a simpler, equally defensible calibration
    # choice than a per-hour Poisson rate here).
    if rng.uniform() >= MASS_EVENT_PROBABILITY_PER_SHIFT:
        return []

    name, lat, lon = MASS_EVENT_VENUES[int(rng.integers(0, len(MASS_EVENT_VENUES)))]
    start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
    duration_min = _randint_inclusive(rng, *MASS_EVENT_DURATION_MIN_RANGE)
    detect_offset = _randint_inclusive(rng, *MASS_EVENT_DETECT_OFFSET_MIN_RANGE)
    slug = name.lower().replace(" ", "_").replace(".", "")
    return [
        Event(
            event_id=f"mass_event-{slug}",
            type=EventType.MASS_EVENT,
            start_min=start_min,
            duration_min=duration_min,
            point_lat=lat,
            point_lon=lon,
            point_radius_km=MASS_EVENT_RADIUS_KM,
            demand_mult=_uniform(rng, *MASS_EVENT_DEMAND_MULT_RANGE),
            speed_mult=_uniform(rng, *MASS_EVENT_SPEED_MULT_RANGE),
            detect_offset_min=detect_offset,
            detect_radius_km=MASS_EVENT_RADIUS_KM * 2,
            confidence=0.95,
        )
    ]


def _generate_kitchen_backlogs(
    rng: np.random.Generator,
    shift_start_min: int,
    shift_end_min: int,
) -> list[Event]:
    hours = (shift_end_min - shift_start_min) / 60.0
    count = int(rng.poisson(KITCHEN_BACKLOG_RATE_PER_HOUR * hours))

    events: list[Event] = []
    for i in range(count):
        start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
        duration_min = _randint_inclusive(rng, *KITCHEN_BACKLOG_DURATION_MIN_RANGE)
        seed_cell = _sample_random_cell(rng)
        cluster = geo.cell_neighbors(seed_cell, k=KITCHEN_BACKLOG_CLUSTER_RING, include_self=True)
        events.append(
            Event(
                event_id=f"kitchen_backlog-{i:04d}",
                type=EventType.KITCHEN_BACKLOG,
                start_min=start_min,
                duration_min=duration_min,
                cells=cluster,
                fixed_delay_min=_uniform(rng, *KITCHEN_BACKLOG_FIXED_DELAY_MIN_RANGE),
                # Discovered only on arrival at the restaurant: no early
                # warning, and the detection radius is essentially "at the
                # door".
                detect_offset_min=0,
                detect_radius_km=KITCHEN_BACKLOG_DETECT_RADIUS_KM,
                confidence=1.0,
            )
        )
    return events


def _generate_weather_stub_events(
    rng: np.random.Generator,
    shift_start_min: int,
    shift_end_min: int,
    include_rain_events: bool,
    include_heat_events: bool,
) -> list[Event]:
    """Opt-in only; see module docstring and `WEATHER_STUB_*` knobs. Off by
    default so `weather.py`'s real signal is never double-counted."""
    events: list[Event] = []
    hours = (shift_end_min - shift_start_min) / 60.0

    if include_rain_events:
        count = int(rng.poisson(WEATHER_STUB_RAIN_RATE_PER_HOUR * hours))
        for i in range(count):
            start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
            duration_min = _randint_inclusive(rng, *WEATHER_STUB_RAIN_DURATION_MIN_RANGE)
            events.append(
                Event(
                    event_id=f"rain_onset_stub-{i:04d}",
                    type=EventType.RAIN_ONSET,
                    start_min=start_min,
                    duration_min=duration_min,
                    cells=[_sample_random_cell(rng)],
                    speed_mult=WEATHER_STUB_RAIN_SPEED_MULT,
                    detect_offset_min=0,
                    detect_radius_km=10.0,
                    confidence=1.0,
                )
            )

    if include_heat_events:
        count = int(rng.poisson(WEATHER_STUB_HEAT_RATE_PER_HOUR * hours))
        for i in range(count):
            start_min = _randint_inclusive(rng, shift_start_min, shift_end_min - 1)
            duration_min = _randint_inclusive(rng, *WEATHER_STUB_HEAT_DURATION_MIN_RANGE)
            events.append(
                Event(
                    event_id=f"extreme_heat_stub-{i:04d}",
                    type=EventType.EXTREME_HEAT,
                    start_min=start_min,
                    duration_min=duration_min,
                    cells=[_sample_random_cell(rng)],
                    courier_supply_mult=WEATHER_STUB_HEAT_SUPPLY_MULT,
                    detect_offset_min=0,
                    detect_radius_km=10.0,
                    confidence=1.0,
                )
            )
    return events


# ============================================================================
# Public API
# ============================================================================


def build_events_timeline(
    scenario_seed: int,
    date: Date,
    shift_start_min: int,
    shift_end_min: int,
    *,
    include_rain_events: bool = False,
    include_heat_events: bool = False,
    graph_path: Path = GRAPH_FIXTURE_PATH,
) -> list[Event]:
    """Build the full exogenous disruptive-event timeline for one shift.

    Pure function of `(scenario_seed, date, shift_start_min, shift_end_min)`
    plus whichever graph fixture state is on disk at call time (see the
    module docstring's graph degradation policy) -- never of any courier
    action. Draws exclusively from the `"incidents"` RNG stream.

    `include_rain_events`/`include_heat_events` default to False: weather is
    `weather.py`'s job, and turning these on is only for early integration
    testing before that module is wired in (see `WEATHER_STUB_*` knobs).
    """
    if shift_end_min <= shift_start_min:
        raise ValueError("shift_end_min must be strictly after shift_start_min")

    rng = rng_streams(scenario_seed)["incidents"]
    graph = _try_load_graph(graph_path)

    events: list[Event] = []
    events += _generate_crashes(rng, shift_start_min, shift_end_min, graph)
    events += _generate_street_closures(rng, shift_start_min, shift_end_min, graph)
    events += _generate_checkpoints(rng, date, shift_start_min, shift_end_min)
    events += _generate_surge_windows(rng, shift_start_min, shift_end_min)
    events += _generate_mass_events(rng, shift_start_min, shift_end_min)
    events += _generate_kitchen_backlogs(rng, shift_start_min, shift_end_min)
    events += _generate_weather_stub_events(
        rng, shift_start_min, shift_end_min, include_rain_events, include_heat_events
    )

    events.sort(key=lambda e: e.start_min)
    return events


def active_events(events: list[Event], minute: int) -> list[Event]:
    """Events actually in effect at `minute` (ground truth -- not filtered
    by whether a courier could know about them; see `perceivable_events`)."""
    return [e for e in events if e.is_active(minute)]


def _event_anchor_latlon(event: Event, graph: nx.MultiDiGraph | None) -> tuple[float, float] | None:
    """Best-effort (lat, lon) to measure detection distance from. Returns
    None only for an edge-scoped event when no graph is loaded to resolve
    node coordinates from -- callers should fail open in that case (see
    `perceivable_events`), since this only happens on the degraded path
    where CRASH/STREET_CLOSURE never carry real graph edges anyway."""
    if event.point_lat is not None and event.point_lon is not None:
        return event.point_lat, event.point_lon
    if event.cells:
        return geo.cell_centroid(event.cells[0])
    if event.edges and graph is not None:
        u, _v = event.edges[0]
        if graph.has_node(u):
            return _node_latlon(graph, u)
    return None


def perceivable_events(
    events: list[Event],
    minute: int,
    courier_lat: float,
    courier_lon: float,
    graph: nx.MultiDiGraph | None = None,
) -> list[Event]:
    """Events a courier standing at (courier_lat, courier_lon) could
    ACTUALLY know about at `minute`.

    This is the honesty check: a courier can never react to an event before
    `event.detectable_from_min` (which is exactly `start_min +
    detect_offset_min` -- negative for an announced event, positive for one
    learned about late), and never to one further away than
    `detect_radius_km` (a checkpoint's tiny radius is what forces "you find
    out only when you are already on top of it"). Perception stops once the
    event has ended.

    `graph` is optional and only used to resolve a coordinate for an
    edge-scoped event; pass the loaded graph (e.g. from
    `_try_load_graph`/`network.get_or_build_graph`) when you have one.
    """
    visible: list[Event] = []
    for event in events:
        if minute < event.detectable_from_min or minute >= event.end_min:
            continue
        distance_km = _event_distance_km(event, graph, courier_lat, courier_lon)
        if distance_km is None:
            # No spatial gate can be applied (degraded edge-scoped event
            # with no graph loaded) -- fail open on timing alone rather
            # than silently hiding it.
            visible.append(event)
            continue
        if distance_km <= event.detect_radius_km:
            visible.append(event)
    return visible


# Per-event coordinate arrays, keyed by (graph identity, event id). Building
# one costs a pass over the event's edges, and `perceivable_events` is called
# once per courier per minute -- without the cache a 900-edge corridor was
# re-walked 480 times a shift for an answer that never changes.
_EVENT_POINTS_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _event_points(event: Event, graph: nx.MultiDiGraph | None) -> np.ndarray | None:
    """Every point the event occupies, as an (n, 2) array of (lat, lon).

    For a point event that is one row; for an edge-scoped event it is every
    node of every closed edge.
    """
    if event.point_lat is not None and event.point_lon is not None:
        return np.array([[event.point_lat, event.point_lon]], dtype=float)
    if event.cells:
        return np.array([geo.cell_centroid(c) for c in event.cells], dtype=float)
    if not event.edges or graph is None:
        return None

    key = (id(graph), event.event_id)
    cached = _EVENT_POINTS_CACHE.get(key)
    if cached is not None:
        return cached
    nodes: set[int] = set()
    for u, v in event.edges:
        nodes.add(u)
        nodes.add(v)
    points = [_node_latlon(graph, n) for n in sorted(nodes) if graph.has_node(n)]
    if not points:
        return None
    array = np.array(points, dtype=float)
    _EVENT_POINTS_CACHE[key] = array
    return array


def _event_distance_km(
    event: Event, graph: nx.MultiDiGraph | None, lat: float, lon: float
) -> float | None:
    """Distance from (lat, lon) to the NEAREST part of `event`, in km.

    Measured against the whole event, not against one representative point,
    and that is the entire reason this function exists. `_event_anchor_latlon`
    returns the first node of the first closed edge, which is fine for a crash
    (a crash IS a point) and badly wrong for a corridor closure: measured on
    the chaotic-day build, three closures placed on the smart courier's own
    corridor were perceived by nobody, because the courier riding INTO the
    closure was several kilometres from the far end the anchor happened to sit
    on. A closure you cannot see while standing on it is not a detection
    radius, it is a bug.

    Returns None when no spatial gate can be applied at all.
    """
    points = _event_points(event, graph)
    if points is None or not len(points):
        return None
    # Equirectangular approximation, which is accurate well inside a percent
    # over a city and avoids a Python-level loop over hundreds of nodes.
    lat_rad = math.radians(lat)
    dy = (points[:, 0] - lat) * 110.574
    dx = (points[:, 1] - lon) * 111.320 * math.cos(lat_rad)
    return float(np.min(np.hypot(dy, dx)))


def make_event(event_id: str, type: EventType, start_min: int, duration_min: int, **effect_kwargs) -> Event:
    """Thin convenience wrapper over `Event(...)` for hand-authoring (see
    `demo_events`): keeps a curated timeline reading like a short list of
    one-liners instead of repeating the full constructor call shape."""
    return Event(event_id=event_id, type=type, start_min=start_min, duration_min=duration_min, **effect_kwargs)


def demo_events(
    scenario_seed: int,
    date: Date,
    shift_start_min: int,
    shift_end_min: int,
) -> list[Event]:
    """Hand-authored timeline for the live demo.

    Deliberately NOT derived from the `incidents` RNG stream: the entire
    point of a serialisable Scenario is being able to place a street
    closure at exactly minute 143 of the shift, right when the courier is
    mid-route, instead of hoping a random draw lands there. Edit this
    function directly to script a different demo beat -- every event below
    is placed at an exact, hand-picked minute.

    `scenario_seed`/`date` are accepted only for signature symmetry with
    `build_events_timeline`, so this function is a drop-in swap at the call
    site; they are not used to derive anything.
    """
    del scenario_seed, date  # hand-authored, not RNG- or date-derived (see docstring)

    downtown_cell = geo.latlon_to_cell(DEMO_DOWNTOWN_LAT, DEMO_DOWNTOWN_LON)
    _avenue_name, checkpoint_lat, checkpoint_lon = MAJOR_AVENUES[0]

    events = [
        make_event(
            "demo-street_closure-mid_route",
            EventType.STREET_CLOSURE,
            start_min=shift_start_min + 143,
            duration_min=45,
            cells=[downtown_cell],
            close_edges=True,
            detect_offset_min=5,  # sudden blockage, discovered a few minutes late
            detect_radius_km=3.0,
            confidence=0.9,
        ),
        make_event(
            "demo-surge-downtown",
            EventType.SURGE_WINDOW,
            start_min=shift_start_min + 60,
            duration_min=40,
            cells=geo.cell_neighbors(downtown_cell, k=1, include_self=True),
            demand_mult=2.5,
            payout_mult=1.6,
            detect_offset_min=0,
            detect_radius_km=5.0,
            confidence=1.0,
        ),
        make_event(
            "demo-checkpoint-evening",
            EventType.CHECKPOINT,
            start_min=shift_start_min + 300,
            duration_min=90,
            point_lat=checkpoint_lat,
            point_lon=checkpoint_lon,
            point_radius_km=CHECKPOINT_RADIUS_KM,
            fixed_delay_min=12.0,
            detect_offset_min=0,
            detect_radius_km=CHECKPOINT_DETECT_RADIUS_KM,
            confidence=1.0,
        ),
    ]
    return sorted(events, key=lambda e: e.start_min)


# --------------------------------------------------------------------------
# Hand-placed corridor closure (demo scripting)
# --------------------------------------------------------------------------

# Starting protection radius, in hops, around every cell-centroid node.
# This alone is NOT sufficient and must never be trusted on its own -- see
# `corridor_closure`'s connectivity repair. Measured: protecting two hops
# around all 127 centroids still left 125 cell pairs unroutable on the real
# Monterrey graph, because in the sparse parts of the city a centroid's only
# viable artery lies further than two hops out. A uniform lattice hides
# this completely, which is exactly how it got past a passing test suite.
CORRIDOR_PROTECT_HOPS = 2

# How much to widen an offending centroid's protection each repair round,
# and how many rounds before giving up. Widening is monotone -- protection
# only ever grows -- so the loop terminates: at a large enough radius every
# edge near that centroid is protected and it cannot be cut off.
CORRIDOR_REPAIR_HOP_STEP = 2
CORRIDOR_REPAIR_MAX_ROUNDS = 8

# A closure this small is not a road closure, it is a rounding error, and
# silently building one is how the demo ends up with a fork that changes
# nothing. Raise instead.
CORRIDOR_MIN_EDGES = 24


def _protected_nodes(
    undirected: nx.Graph,
    cell_to_node: Mapping[str, int],
    cell_order: Sequence[str],
    radius_by_cell: Mapping[str, int],
) -> set[int]:
    """Every node within its cell's protection radius of a cell centroid.

    Takes an already-undirected view, and does a depth-limited BFS rather
    than `nx.ego_graph`. Not a style preference: `ego_graph(...,
    undirected=True)` converts the WHOLE graph to undirected on every call,
    so protecting 127 centroids converted the 95k-node Monterrey graph 127
    times and the repair loop never finished. Direction is deliberately
    ignored here -- protection is about which roads are off limits, and a
    one-way street is just as much a road.
    """
    protected: set[int] = set()
    for cell in cell_order:
        node = cell_to_node.get(cell)
        if node is None or not undirected.has_node(node):
            continue
        protected.update(
            nx.single_source_shortest_path_length(undirected, node, cutoff=radius_by_cell[cell])
        )
    return protected


def _corridor_candidate_edges(
    matrix: TravelMatrix,
    pairs: Sequence[tuple[str, str]],
    protected: set[int],
) -> set[tuple[int, int]]:
    cell_index = {cell: i for i, cell in enumerate(matrix.cell_order)}
    edges: set[tuple[int, int]] = set()
    for origin, dest in pairs:
        i, j = cell_index.get(origin), cell_index.get(dest)
        if i is None or j is None:
            continue
        path = matrix._paths.get((i, j))
        if not path:
            # Not every pair is cached (measured: 15,878 of 16,129). A
            # missing path is not an error, just one fewer road to close.
            continue
        for u, v in zip(path[:-1], path[1:]):
            if u not in protected and v not in protected:
                edges.add((u, v))
    return edges


def _reachable_cells(graph: nx.MultiDiGraph, matrix: TravelMatrix) -> set[str]:
    """Cells whose centroid node sits in the graph's largest strongly
    connected component. Strong connectivity is the right test and weak is
    not: a courier has to be able to drive out AND back, and one-way
    streets make those two different questions."""
    largest: set[int] = max(nx.strongly_connected_components(graph), key=len, default=set())
    return {cell for cell in matrix.cell_order if matrix.cell_to_node.get(cell) in largest}


def corridor_closure(
    event_id: str,
    start_min: int,
    duration_min: int,
    corridor_cells: Sequence[str],
    matrix: TravelMatrix,
    *,
    protect_hops: int = CORRIDOR_PROTECT_HOPS,
    detect_offset_min: int = 5,
    detect_radius_km: float = 3.0,
    confidence: float = 0.9,
) -> Event:
    """A STREET_CLOSURE scoped to the roads a courier ACTUALLY drives, and
    guaranteed not to cut any cell off the road network.

    `corridor_cells` is the courier's own operating corridor, busiest cell
    first -- derive it from a recorded run (see `corridor_from_occupancy`),
    never from a landmark. A closure anchored on a fixed downtown cell is a
    coin flip: on most seeds the courier's route never touches it, the fork
    comes out identical to the reference run, and the demo beat is a lie.

    What it closes: the node paths linking the busiest cell to every other
    cell in the corridor, in both directions, minus every edge near a cell
    centroid. Both directions matter because one-way streets and
    limited-access ramps mean the road out of a zone is not the road back
    into it.

    THE LOAD-BEARING GUARANTEE, and the reason this is not just a path
    slice: closing a road must make a trip LONGER, never impossible. An
    unroutable pair does not take a detour -- it falls through to
    `NetworkTravelOracle`'s synthetic fallback constants, so the simulator
    silently stops simulating the street network it claims to simulate.

    Hop-based protection alone does NOT deliver that guarantee. Measured on
    the real Monterrey graph, protecting two hops around all 127 centroids
    still left 125 pairs unroutable, because in the sparse parts of the city
    a centroid's only artery is further out than that. A uniform synthetic
    lattice hides the failure entirely. So this function does not assume: it
    removes the candidate edges from a scratch copy of the graph, recomputes
    strong connectivity, widens the protection around any centroid that fell
    out of the largest component, and repeats until none do. Only cells that
    were reachable to begin with are held to the standard -- the fixture
    already carries 251 unroutable pairs and this function is not
    responsible for those.

    `detect_offset_min=5` means the courier learns about it five minutes
    late: a sudden blockage, not an announced roadwork. That is what makes
    "it did not see this coming, and it re-planned" an honest sentence.

    Raises ValueError if the corridor is too short, if connectivity cannot be
    repaired within `CORRIDOR_REPAIR_MAX_ROUNDS` rounds, or if fewer than
    `CORRIDOR_MIN_EDGES` edges survive -- never a closure too small to bite.
    """
    if len(corridor_cells) < 2:
        raise ValueError(
            f"corridor_cells needs at least an origin and one partner, got {list(corridor_cells)}"
        )

    graph: nx.MultiDiGraph = matrix.graph
    hot, partners = corridor_cells[0], list(corridor_cells[1:])
    pairs = [(hot, other) for other in partners] + [(other, hot) for other in partners]

    reachable_before = _reachable_cells(graph, matrix)
    # Built once, reused every repair round. See `_protected_nodes`.
    undirected = graph.to_undirected(as_view=False, reciprocal=False)
    radius_by_cell: dict[str, int] = {cell: protect_hops for cell in matrix.cell_order}

    edges: set[tuple[int, int]] = set()
    for round_number in range(1, CORRIDOR_REPAIR_MAX_ROUNDS + 1):
        protected = _protected_nodes(undirected, matrix.cell_to_node, matrix.cell_order, radius_by_cell)
        edges = _corridor_candidate_edges(matrix, pairs, protected)
        if not edges:
            break

        # A read-only view, NOT a copy. Copying a 95k-node / 242k-edge
        # MultiDiGraph once per repair round dominated the whole build;
        # `restricted_view` hides the edges in O(1) and strong connectivity
        # reads it exactly the same way.
        hidden = [
            (u, v, key) for u, v in edges for key in (graph.get_edge_data(u, v) or {})
        ]
        scratch = nx.restricted_view(graph, [], hidden)
        cut_off = sorted(reachable_before - _reachable_cells(scratch, matrix))
        if not cut_off:
            logger.info(
                "corridor_closure %r: %d edges closed across %d corridor pair(s) after %d "
                "connectivity round(s); no cell was cut off the network.",
                event_id, len(edges), len(pairs), round_number,
            )
            break

        logger.info(
            "corridor_closure %r: round %d cut %d cell(s) off the network (%s); widening their "
            "protection by %d hop(s) and retrying.",
            event_id, round_number, len(cut_off), ", ".join(cut_off[:5]), CORRIDOR_REPAIR_HOP_STEP,
        )
        for cell in cut_off:
            radius_by_cell[cell] += CORRIDOR_REPAIR_HOP_STEP
    else:
        raise ValueError(
            f"corridor_closure could not place a closure on corridor {list(corridor_cells)} "
            f"without cutting a cell off the road network, after {CORRIDOR_REPAIR_MAX_ROUNDS} "
            f"widening rounds. Pick a different corridor rather than shipping a closure that "
            f"makes trips impossible instead of longer."
        )

    if len(edges) < CORRIDOR_MIN_EDGES:
        raise ValueError(
            f"corridor_closure resolved only {len(edges)} closable edge(s) for corridor "
            f"{list(corridor_cells)} (minimum {CORRIDOR_MIN_EDGES}). Either the corridor is too "
            f"short or the connectivity repair has protected all of it; a closure this small "
            f"would leave the forked run identical to the reference run."
        )

    return Event(
        event_id=event_id,
        type=EventType.STREET_CLOSURE,
        start_min=start_min,
        duration_min=duration_min,
        edges=sorted(edges),
        close_edges=True,
        detect_offset_min=detect_offset_min,
        detect_radius_km=detect_radius_km,
        confidence=confidence,
    )


def corridor_from_occupancy(
    cell_minutes: Mapping[str, int], limit: int = 7
) -> list[str]:
    """The courier's operating corridor, busiest cell first.

    `cell_minutes` is how many minutes of the shift the courier spent in
    each cell -- count it off a recorded run's ticks. Measured on the
    reference seed, the smart courier spent 203 of 480 minutes in a single
    cell and never entered more than seven, which is why a closure placed
    anywhere else physically cannot touch it.
    """
    ranked = sorted(cell_minutes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [cell for cell, minutes in ranked[:limit] if minutes > 0]


# The block around a closed road, in km. A closure is not only impassable
# along its own length: the traffic it displaces congeals on the streets
# feeding it, and that halo is what a courier actually rides into. Kept
# deliberately SHORT -- this claims the surrounding block, not the district.
# Anything larger would be a statement about congestion propagation that
# nothing in this world model measures.
CLOSURE_JAM_RADIUS_KM = 0.35


def closure_segments(
    event: Event, graph: nx.MultiDiGraph | None
) -> list[list[tuple[float, float]]]:
    """The closed roads of `event`, as drawable (lat, lon) polylines.

    `Event.edges` holds OSM node-id PAIRS, which is the right storage for a
    routing engine and useless to a map: rendering a closure from
    `_event_anchor_latlon` drew the whole thing as one pin at the first
    edge's `u` node, under-reporting a multi-kilometre corridor by most of
    its length.

    Uses the real OSM edge geometry where the graph carries one, so a curved
    avenue draws as a curve rather than a chord across the blocks it bends
    around; falls back to the straight node-to-node line otherwise. Returns
    an empty list for an event that is genuinely a point, or when no graph
    is loaded to resolve node ids against.
    """
    if not event.edges or graph is None:
        return []
    segments: list[list[tuple[float, float]]] = []
    for u, v in event.edges:
        if not (graph.has_node(u) and graph.has_node(v)):
            continue
        line = None
        for _key, data in (graph.get_edge_data(u, v) or {}).items():
            geometry = data.get("geometry")
            if geometry is not None:
                # OSMnx stores edge geometry as a shapely LineString in
                # (x, y) = (lon, lat) order.
                line = [(float(y), float(x)) for x, y in geometry.coords]
                break
        if line is None:
            line = [_node_latlon(graph, u), _node_latlon(graph, v)]
        if len(line) >= 2:
            segments.append(line)
    return _merge_chains(segments)


def _merge_chains(
    segments: list[list[tuple[float, float]]]
) -> list[list[tuple[float, float]]]:
    """Join polylines that share an endpoint into continuous runs.

    Identical ink, far fewer objects. A corridor closure covers hundreds of
    road edges -- measured on the chaotic day, 484 to 893 apiece -- and each
    one handed to the map separately becomes its own vector, twice over once
    the jam halo is drawn under it. Five such closures live at once is several
    thousand SVG paths for what the eye reads as a handful of closed avenues.

    This is a rendering concern only: nothing is dropped, simplified or
    approximated, so a closure still draws over exactly the roads it shuts.
    """
    if not segments:
        return []
    # Endpoints are float pairs straight off the graph, so they compare
    # exactly when they come from the same node -- no tolerance needed, and a
    # tolerance would risk welding two roads that merely pass close.
    remaining = [list(seg) for seg in segments]
    by_start: dict[tuple[float, float], list[list[tuple[float, float]]]] = {}
    for seg in remaining:
        by_start.setdefault(seg[0], []).append(seg)

    used: set[int] = set()
    chains: list[list[tuple[float, float]]] = []
    for seg in remaining:
        if id(seg) in used:
            continue
        used.add(id(seg))
        chain = list(seg)
        # Walk forward while exactly one unused segment continues this one.
        while True:
            candidates = [c for c in by_start.get(chain[-1], []) if id(c) not in used]
            if not candidates:
                break
            nxt = candidates[0]
            used.add(id(nxt))
            chain.extend(nxt[1:])
        chains.append(chain)
    return chains
