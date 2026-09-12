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
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox

from src.world import geo
from src.world.network import GRAPH_FIXTURE_PATH
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
        anchor = _event_anchor_latlon(event, graph)
        if anchor is None:
            # No spatial gate can be applied (degraded edge-scoped event
            # with no graph loaded) -- fail open on timing alone rather
            # than silently hiding it.
            visible.append(event)
            continue
        distance_km = geo.great_circle_km(courier_lat, courier_lon, anchor[0], anchor[1])
        if distance_km <= event.detect_radius_km:
            visible.append(event)
    return visible


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
