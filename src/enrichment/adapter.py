"""The `EnrichmentPort` implementation: the courier's own external tools.

Wires together the five estimate builders in this package
(`weather_estimate`, `traffic_estimate`, `events_perception`,
`kitchen_memory`, `demand_sense`) plus untouched self-knowledge into one
`Observation` per minute. This is the ONLY class in this package meant to be
constructed by callers outside it.

Determinism contract, spelled out because it is load-bearing: `observe()`
must be called at most once per minute, in non-decreasing minute order, for
a given instance — exactly how the engine ticks a shift forward. All noise
is drawn from the single `observation_noise` RNG stream passed in (or
derived from the scenario seed), consumed in the same fixed order every
call, so the same scenario plus the same sequence of `observe()` calls
reproduces byte-identical observations on replay.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.core.ports import CourierSnapshot, Observation
from src.enrichment.demand_sense import estimate_demand
from src.enrichment.events_perception import build_perceived_events
from src.enrichment.kitchen_memory import KitchenMemory
from src.enrichment.traffic_estimate import estimate_traffic
from src.enrichment.weather_estimate import estimate_weather
from src.world import geo
from src.world.demand import day_type_for
from src.world.scenario import Scenario, rng_streams
from src.world.timeline import Event, TrafficTick, WeatherTick


class EnrichmentAdapter:
    """The courier's own external tools: weather app, traffic app, memory of
    kitchens, and a feel for the day's rhythm. Implements `EnrichmentPort`.

    Constructed once per scenario/shift. `graph` is optional and only used
    to resolve a coordinate for an edge-scoped event (see
    `src.world.events.perceivable_events`); omit it to use the same
    cell/point-scoped degradation the ground-truth event generator itself
    falls back to when no drive graph is available.

    It also fills `Observation.cell_coords`: a centroid for every cell named
    in `traffic_by_cell` or `demand_by_cell`. That is not a leak — a courier
    reading the zones off their own app plainly knows where those zones are
    — and withholding it made both belief maps unusable, because a cell id
    the policy cannot place tells it nothing about whether that zone is on
    its way or across the city.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        graph: nx.MultiDiGraph | None = None,
        noise_rng: np.random.Generator | None = None,
    ) -> None:
        self._shift_start_min = scenario.shift_start_min
        self._weather_by_minute: dict[int, WeatherTick] = {t.minute: t for t in scenario.weather_timeline}
        self._traffic_by_minute: dict[int, TrafficTick] = {t.minute: t for t in scenario.traffic_timeline}
        # Held only to be handed to `perceivable_events` — never inspected
        # directly by this class or by anything downstream of it. See
        # `events_perception.build_perceived_events`.
        self._events: list[Event] = list(scenario.events_timeline)
        self._graph = graph
        self._day_type = day_type_for(scenario.day_of_week)
        self._cell_order: list[str] = sorted(geo.load_cell_index()["cell"].tolist())
        # Coordinates for every cell the courier will be told about. A
        # courier looking at their own app plainly knows where the zones on
        # it are, so handing these over leaks nothing — and withholding them
        # was worse than useless: it handed the policy cell IDS it could not
        # place on a map, so it could not tell a believed-busy zone on its
        # way from one across the city, and both the traffic and the demand
        # belief were dead weight.
        #
        # Seeded from the cell catalog (the fine grid `estimate_demand`
        # keys on) and extended lazily in `_coords_for`, because the
        # courier's traffic app also reports on a few cells just outside
        # the operating grid.
        self._cell_coords: dict[str, tuple[float, float]] = {
            cell: geo.cell_centroid(cell) for cell in self._cell_order
        }

        self._rng: np.random.Generator = (
            noise_rng if noise_rng is not None else rng_streams(scenario.seed)["observation_noise"]
        )
        self._kitchen_memory = KitchenMemory()

    def record_kitchen_visit(self, denue_id: str, observed_prep_minutes: float, minute: int) -> None:
        """Called by the engine exactly once per pickup, with the actual
        prep time the courier just experienced. This is how kitchen memory
        is earned rather than given (see `kitchen_memory.py`)."""
        self._kitchen_memory.record_visit(denue_id, observed_prep_minutes, minute)

    def observe(self, minute: int, courier: CourierSnapshot) -> Observation:
        weather_tick = self._weather_by_minute.get(minute) or self._weather_by_minute[
            min(self._weather_by_minute)
        ]
        temp_c, apparent_c, precip_mm_per_hour = estimate_weather(weather_tick, self._rng)

        traffic_by_cell = estimate_traffic(
            minute=minute,
            courier_lat=courier.lat,
            courier_lon=courier.lon,
            courier_cell=courier.cell,
            traffic_by_minute=self._traffic_by_minute,
            shift_start_min=self._shift_start_min,
            rng=self._rng,
        )

        perceived_events = build_perceived_events(
            events=self._events,
            minute=minute,
            courier_lat=courier.lat,
            courier_lon=courier.lon,
            graph=self._graph,
            rng=self._rng,
        )

        demand_by_cell = estimate_demand(
            minute=minute,
            day_type=self._day_type,
            cell_order=self._cell_order,
            perceived_events=perceived_events,
            rng=self._rng,
        )

        km_to_home = geo.great_circle_km(courier.lat, courier.lon, courier.home_lat, courier.home_lon)

        # Exactly the cells named in the two belief maps, plus the one the
        # courier is standing in — coordinates for what their tools just
        # told them about, not a gazetteer of the city. Without this the
        # policy is handed cell IDS it cannot place on a map, so it cannot
        # tell a believed-busy zone on its way from one across town, and
        # both belief maps are dead weight.
        cell_coords = self._coords_for(
            traffic_by_cell, demand_by_cell, extra_cell=courier.cell
        )

        return Observation(
            minute=minute,
            at_lat=courier.lat,
            at_lon=courier.lon,
            at_cell=courier.cell,
            temp_c=temp_c,
            apparent_c=apparent_c,
            precip_mm_per_hour=precip_mm_per_hour,
            traffic_by_cell=traffic_by_cell,
            perceived_events=perceived_events,
            kitchen_minutes_by_denue_id=self._kitchen_memory.snapshot(minute),
            demand_by_cell=demand_by_cell,
            minutes_left_in_shift=courier.minutes_left_in_shift,
            km_to_home=km_to_home,
            fuel_minutes_remaining=courier.fuel_minutes_remaining,
            cell_coords=cell_coords,
        )

    def _coords_for(self, *cell_maps: dict, extra_cell: str = "") -> dict[str, tuple[float, float]]:
        """Centroid per cell named in any of `cell_maps`.

        Memoised on `self._cell_coords`: `geo.cell_centroid` is a pure
        function of the cell id, so a cell resolved once never needs
        resolving again, and the traffic app only ever adds a handful of
        cells per minute to the catalog seeded at construction.
        """
        out: dict[str, tuple[float, float]] = {}
        for cell_map in cell_maps:
            for cell in cell_map:
                if not cell:
                    continue
                coords = self._cell_coords.get(cell)
                if coords is None:
                    coords = geo.cell_centroid(cell)
                    self._cell_coords[cell] = coords
                out[cell] = coords
        if extra_cell:
            coords = self._cell_coords.get(extra_cell)
            if coords is None:
                coords = geo.cell_centroid(extra_cell)
                self._cell_coords[extra_cell] = coords
            out[extra_cell] = coords
        return out
