"""TravelOracle backed by the real Monterrey street network (ground truth).

Implements `src.core.ports.TravelOracle`: `travel(from_cell, to_cell, minute)
-> (km, minutes)`, km and minutes always returned separately. This is the
engine's own oracle — backed by the real network and true conditions, never
to be confused with the agent's own forward model (same protocol, different
implementation, per the port's docstring).

Congestion is applied to MINUTES ONLY: `distance_km` never changes with
traffic. A street closure is the one thing that changes both, because it is
a real detour (`network.TravelMatrix.close_streets`), not more of the same
road driven slower.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.engine.calibration import TRAVEL_CALIBRATION
from src.world import geo
from src.world.network import GRAPH_FIXTURE_PATH, TravelMatrix, get_or_build_graph
from src.world.timeline import Event, TrafficTick
from src.world.traffic import travel_time_minutes

logger = logging.getLogger(__name__)


@dataclass
class NetworkTravelOracle:
    """Ground-truth `TravelOracle` over the precomputed cell-to-cell matrix.

    Holds the FULL `TravelMatrix` (graph + per-pair node paths), not just the
    fast `distance_km`/`time_min` arrays, because two things this engine
    needs both require it:

      - `apply_closure`: `TravelMatrix.close_streets` only knows which rows a
        closure affects by checking which cached paths used the removed
        edges — the fast-array-only load (`TravelMatrix.load_matrices`)
        never has that path cache.
      - `route_polyline`: real street geometry for `ShiftResult.routes`,
        which needs the graph and the cached node paths too.

    KNOWN SIMPLIFICATION: `TravelMatrix` exposes no "reopen a street"
    primitive — `close_streets` mutates the working graph and the affected
    rows in place, permanently, for the life of the object. A `STREET_CLOSURE`
    event is therefore applied once, at its start minute, and never undone
    even after `event.end_min`. Over an 8-hour shift with a rare (~0-1 per
    shift) closure whose duration can run up to 4 hours, this understates how
    quickly the road reopens rather than overstating disruption.

    Two mitigations against that accumulation actually mattering (see
    `calibration.TRAVEL_CALIBRATION` for the numbers and the full reasoning):
      1. `apply_closure` refuses further closures once
         `max_simultaneous_closures` are already active — bounded damage
         instead of the graph degrading toward permanent demolition over an
         8-hour shift. Cheap (no rebuild) on purpose: reopening for real
         means rebuilding the whole matrix from a pristine baseline, which
         costs one Dijkstra per operating cell over the full drive graph
         (~100+ seconds measured here) — too slow to pay mid-shift.
      2. `travel()` NEVER raises on an unroutable pair. A courier in the real
         world always finds some way around; it does not teleport or refuse
         the trip. An unroutable pair falls back to the PRE-CLOSURE
         ("pristine") baseline for that exact pair, penalised by an explicit
         detour multiplier on both km and minutes; if even the pristine pair
         has no route (a base-graph gap, nothing to do with any closure),
         it falls back once more to a straight-line estimate. Every fallback
         is logged.
    """

    matrix: TravelMatrix
    traffic_by_minute: dict[int, TrafficTick]
    _cell_row: dict[str, int] = field(default_factory=dict, repr=False)
    _active_closures: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict, repr=False)
    _pristine_distance_km: np.ndarray | None = field(default=None, repr=False)
    _pristine_time_min: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self._cell_row:
            self._cell_row = {cell: i for i, cell in enumerate(self.matrix.cell_order)}
        # Snapshot the pre-closure baseline ONCE, at construction — this is
        # what `_unroutable_fallback` falls back to, and it must be taken
        # before this oracle ever applies a closure of its own.
        if self._pristine_distance_km is None:
            self._pristine_distance_km = self.matrix.distance_km.copy()
        if self._pristine_time_min is None:
            self._pristine_time_min = self.matrix.time_min.copy()

    @classmethod
    def from_fixtures(cls, traffic_timeline: list[TrafficTick]) -> "NetworkTravelOracle":
        """Build the full travel matrix from the on-disk graph + cell
        fixtures (see `src.world.network`), and index the given traffic
        timeline by minute for O(1) lookup during `travel()`.

        This is an expensive, one-time call (one Dijkstra per operating
        cell over the full drive graph) — build one oracle and reuse it for
        every shift run against the same scenario/date rather than
        rebuilding per run.
        """
        cell_index = geo.load_cell_index()
        polygon = geo.build_operating_polygon(cell_index["lat"], cell_index["lon"])
        graph = get_or_build_graph(polygon, cache_path=GRAPH_FIXTURE_PATH)
        matrix = TravelMatrix.build(graph, cell_index)
        return cls(matrix=matrix, traffic_by_minute={tick.minute: tick for tick in traffic_timeline})

    # -- TravelOracle protocol --------------------------------------------

    def travel(self, from_cell: str, to_cell: str, minute: int) -> tuple[float, float]:
        """Return (km, minutes). Congestion (this minute's `TrafficTick`,
        evaluated at the ORIGIN cell) scales minutes only; km always comes
        straight from the matrix, unaffected by traffic.

        NEVER raises for an unroutable pair — see `_unroutable_fallback` and
        `calibration.TRAVEL_CALIBRATION`. A courier does not get to refuse a
        trip because the fastest-path graph momentarily disconnected."""
        i = self._cell_row.get(from_cell)
        j = self._cell_row.get(to_cell)
        if i is None or j is None:
            raise KeyError(f"Unknown cell in travel query: {from_cell!r} -> {to_cell!r}")
        base_km = float(self.matrix.distance_km[i, j])
        base_minutes = float(self.matrix.time_min[i, j])
        if base_km != base_km or base_minutes != base_minutes:  # NaN check, no numpy import needed
            base_km, base_minutes = self._unroutable_fallback(i, j, from_cell, to_cell)
        tick = self.traffic_by_minute.get(minute)
        if tick is None:
            return base_km, base_minutes
        return base_km, travel_time_minutes(base_minutes, tick, from_cell)

    def _unroutable_fallback(self, i: int, j: int, from_cell: str, to_cell: str) -> tuple[float, float]:
        """(i, j) has no route through the CURRENT (post-closure) working
        graph. Fall back to the pre-closure ("pristine") baseline for this
        exact pair, penalised by `TRAVEL_CALIBRATION`'s detour multipliers —
        a real courier takes a real, if here unmodelled, detour rather than
        refusing the trip. If even the pristine pair has no route (a gap in
        the base graph itself, unrelated to any closure this oracle ever
        applied), fall back once more to a straight-line estimate. This
        method never raises; every fallback is logged so it stays auditable.
        """
        cal = TRAVEL_CALIBRATION
        pristine_km = float(self._pristine_distance_km[i, j])
        pristine_minutes = float(self._pristine_time_min[i, j])
        if pristine_km == pristine_km and pristine_minutes == pristine_minutes:  # not NaN
            logger.warning(
                "NetworkTravelOracle: %r -> %r is unroutable through the current "
                "(post-closure) working graph; falling back to the pre-closure "
                "baseline (%.2f km, %.2f min) with a detour penalty.",
                from_cell, to_cell, pristine_km, pristine_minutes,
            )
            return (
                pristine_km * cal["unroutable_detour_km_multiplier"],
                pristine_minutes * cal["unroutable_detour_minutes_multiplier"],
            )
        lat1, lon1 = geo.cell_centroid(from_cell)
        lat2, lon2 = geo.cell_centroid(to_cell)
        km = geo.great_circle_km(lat1, lon1, lat2, lon2) * cal["unroutable_detour_km_multiplier"]
        minutes = km / cal["fallback_speed_kmh"] * 60.0 * cal["unroutable_detour_minutes_multiplier"]
        logger.warning(
            "NetworkTravelOracle: %r -> %r has no route even in the pristine "
            "pre-closure baseline; falling back to a straight-line estimate "
            "(%.2f km, %.2f min).",
            from_cell, to_cell, km, minutes,
        )
        return km, minutes

    # -- street closures (ground truth only; never called by src/agent/) --

    def apply_closure(self, event: Event) -> list[int]:
        """Route a STREET_CLOSURE event through `TravelMatrix.close_streets`
        so a blockage produces both more km (a real detour through the
        remaining street graph) and more minutes. Idempotent per
        `event_id`. A no-op for a degraded, cell-scoped event (no real
        graph edges to close — see `events.py`'s graph degradation policy);
        such an event still reaches the courier only through whatever the
        platform/enrichment layers choose to surface about it.

        Refuses to apply once `TRAVEL_CALIBRATION["max_simultaneous_closures"]`
        are already active (logged, not applied) — closures accumulate for
        the rest of the shift (see the class docstring), so left unbounded
        an unlucky run could eventually disconnect part of the operating
        area. This bounds the damage cheaply; `travel()`'s own fallback is
        what guarantees correctness even so.
        """
        if event.event_id in self._active_closures or not event.edges:
            return []
        max_active = int(TRAVEL_CALIBRATION["max_simultaneous_closures"])
        if len(self._active_closures) >= max_active:
            logger.warning(
                "NetworkTravelOracle: refusing to apply closure %r — already at the "
                "%d simultaneous-closure cap; the network stays as degraded as it "
                "currently is, not more so.",
                event.event_id, max_active,
            )
            return []
        edge_keys = self._resolve_edge_keys(event.edges)
        if not edge_keys:
            return []
        affected_rows = self.matrix.close_streets(edge_keys)
        self._active_closures[event.event_id] = edge_keys
        return affected_rows

    def _resolve_edge_keys(self, edges: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        """Every parallel key between each (u, v) pair, so a closure blocks
        the road rather than just one of several parallel lanes."""
        graph: nx.MultiDiGraph = self.matrix.graph
        keys: list[tuple[int, int, int]] = []
        for u, v in edges:
            if not graph.has_edge(u, v):
                continue
            for key in graph.get_edge_data(u, v):
                keys.append((u, v, key))
        return keys

    # -- route geometry for the replay/evaluation layer --------------------

    def route_polyline(self, from_cell: str, to_cell: str) -> list[tuple[float, float]]:
        """Real (lat, lon) street geometry for one leg, for `ShiftResult.routes`."""
        return self.matrix.route_polyline(from_cell, to_cell)
