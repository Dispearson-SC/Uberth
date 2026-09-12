"""The `RawSourcePort` implementation: sources the agent QUERIES, not pushes.

This is the portable replacement for `EnrichmentAdapter`. The difference is
not cosmetic and it is not about performance:

  - `EnrichmentAdapter.observe()` builds a finished `Observation` and hands
    it over, which means the ENGINE decides what the agent needs to know.
    That is the simulator author's judgement baked into the agent's
    perception, and "where does the agent get that traffic estimate?"
    answers "we gave it to him", which is not an answer.
  - Worse, it cannot be exported. Drop the agent in Guadalajara and there is
    no engine to push it beliefs. It does not degrade; it does not start.

In production this class is replaced by reality, and reality pushes nothing.
It answers questions, if you know which to ask. So the agent asks.

THE PORTABILITY RULE, which is what every method here is shaped around:
every query must be answerable from a pair of coordinates and a phone
screen. A weather service answers any lat/lon. OpenStreetMap covers the
planet. The app feed is the same six fields in every market. The agent's own
history is empty on arrival everywhere. Nothing in this interface names a
country-specific dataset, and `tests/agent/test_portability.py` fails the
build if one appears.

The IMPLEMENTATION behind the interface is this simulator's stand-in for
reality, and the simulator IS the world — it is allowed Mexico-only sources
exactly as reality is allowed to be Mexican. The line falls at the
interface, and that is not a matter of taste: the agent using a Mexico-only
source would be cheating, because in Guadalajara it would not have one.

WHAT IS REUSED UNCHANGED, because it is good and it is measured: the noise,
lag and confidence machinery in `weather_estimate`, `traffic_estimate`,
`events_perception` and `kitchen_memory`. Traffic MAE runs 0.198 at ring 1
rising to 0.405 at ring 3, with zero exact truth matches in 1,099 estimates.
What changed is the SHAPE of the interface, not the error model.

THE DETECTABILITY RULE STAYS EXACTLY WHERE IT WAS. `perceived_disruptions`
is built ONLY from `src.world.events.perceivable_events`, via
`events_perception.build_perceived_events`. Never `active_events`, never the
raw timeline, never `Event.is_active` or `Event.start_min`. A crash starting
at minute 143 and detectable from 149 must not exist at 145, and that is the
property that keeps the agent from being clairvoyant.

DETERMINISM, spelled out because it is load-bearing. All noise comes from
the single `observation_noise` RNG stream, consumed once per minute in one
fixed order — weather, then traffic, then events, then POI density — by
`_refresh`, which the engine triggers exactly once per tick. Query order
within a minute therefore cannot affect any draw: the first query of a
minute refreshes everything and the rest read the cache. That is also why
migrating from `observe()` to these queries did not re-roll the simulator's
noise: the draw count and order are identical to what `observe()` consumed.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.core.ports import Estimate, PerceivedEvent
from src.enrichment.calibration import POI_DENSITY_CALIBRATION, TRAVEL_SOURCE_CALIBRATION
from src.enrichment.demand_sense import restaurant_weight_by_cell
from src.enrichment.events_perception import build_perceived_events
from src.enrichment.history import CourierHistory, TripRecord
from src.enrichment.osm_travel import TravelSkeleton, build_travel_skeleton, poi_coordinates
from src.enrichment.traffic_estimate import estimate_traffic
from src.enrichment.weather_estimate import estimate_weather
from src.world import geo
from src.world.scenario import Scenario, rng_streams
from src.world.timeline import Event, TrafficTick, WeatherTick


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class RawSourceAdapter:
    """`RawSourcePort` over one scenario. Constructed once per shift.

    `history` is the courier's accumulated experience and is the ONE thing
    meant to outlive the scenario: pass the previous shift's history in to
    run consecutive shifts as the same courier, or leave it out for a
    courier who has never worked this city. `skeleton` is the OSM travel
    matrix (see `osm_travel.py`); it is pure geometry, so it is shared
    freely across shifts, seeds and policies.

    `graph` is optional and only used to resolve a coordinate for an
    edge-scoped event, exactly as in `EnrichmentAdapter`.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        graph: nx.MultiDiGraph | None = None,
        noise_rng: np.random.Generator | None = None,
        history: CourierHistory | None = None,
        skeleton: TravelSkeleton | None = None,
    ) -> None:
        self._shift_start_min = scenario.shift_start_min
        self._weather_by_minute: dict[int, WeatherTick] = {
            t.minute: t for t in scenario.weather_timeline
        }
        self._traffic_by_minute: dict[int, TrafficTick] = {
            t.minute: t for t in scenario.traffic_timeline
        }
        # Held only to be handed to `perceivable_events`, never inspected
        # here. See the module docstring's detectability paragraph.
        self._events: list[Event] = list(scenario.events_timeline)
        self._graph = graph
        self._cell_order: list[str] = sorted(geo.load_cell_index()["cell"].tolist())
        self._rng: np.random.Generator = (
            noise_rng if noise_rng is not None else rng_streams(scenario.seed)["observation_noise"]
        )
        self.history = history if history is not None else CourierHistory()
        self._skeleton = skeleton if skeleton is not None else build_travel_skeleton(*poi_coordinates())

        self._cell_coords: dict[str, tuple[float, float]] = {
            cell: geo.cell_centroid(cell) for cell in self._cell_order
        }
        # Route factor and scooter speed per cell pair. Pure geometry, so it
        # is valid for the life of the adapter — see `_corridor`.
        self._corridor_cache: dict[tuple[str, str], tuple[float, float]] = {}

        # Per-minute cache, rebuilt by `_refresh`.
        self._minute: int | None = None
        self._weather: dict[str, Estimate] = {}
        self._congestion: dict[str, Estimate] = {}
        self._disruptions: tuple[PerceivedEvent, ...] = ()
        self._poi_density: dict[str, Estimate] = {}

    # ------------------------------------------------------------------
    # The one-per-minute refresh. Every draw this adapter makes is here.
    # ------------------------------------------------------------------

    def _refresh(self, minute: int, lat: float, lon: float) -> None:
        """Take one full reading of every source, from where the courier is.

        Called at most once per minute per instance, in non-decreasing
        minute order — exactly how the engine ticks a shift forward. The
        fixed order below is the whole determinism contract: change it and
        every downstream estimate in the simulator re-rolls.
        """
        if self._minute == minute:
            return
        self._minute = minute

        weather_tick = self._weather_by_minute.get(minute) or self._weather_by_minute[
            min(self._weather_by_minute)
        ]
        temp_c, apparent_c, precip_per_hour = estimate_weather(weather_tick, self._rng)
        self._weather = {
            "temp_c": temp_c,
            "apparent_c": apparent_c,
            "precip_mm_per_hour": precip_per_hour,
        }

        self._congestion = estimate_traffic(
            minute=minute,
            courier_lat=lat,
            courier_lon=lon,
            courier_cell=geo.latlon_to_cell(lat, lon),
            traffic_by_minute=self._traffic_by_minute,
            shift_start_min=self._shift_start_min,
            rng=self._rng,
        )

        self._disruptions = build_perceived_events(
            events=self._events,
            minute=minute,
            courier_lat=lat,
            courier_lon=lon,
            graph=self._graph,
            rng=self._rng,
        )

        self._poi_density = self._read_poi_density()

    def _read_poi_density(self) -> dict[str, Estimate]:
        """Food-commerce density per cell, as the courier's map app reports it.

        The VALUE is static geography — a POI count, normalised by the
        densest cell. The noise is the app's own jitter in what it reports:
        a POI table is never a perfect census of what is open and
        delivering today.

        Redrawn each time the courier looks, and drawn with the same two
        std-defined normals per cell in the same `cell_order` as the sensed
        demand this replaced. That is deliberate: it is what let the
        interface change from push to pull without re-rolling every other
        estimate in the simulator as a side effect, so the before/after
        comparison measures one change rather than two.
        """
        cal = POI_DENSITY_CALIBRATION
        weight_by_cell = restaurant_weight_by_cell()
        max_weight = max(weight_by_cell.values(), default=0.0) or 1.0
        out: dict[str, Estimate] = {}
        for cell in self._cell_order:
            density = weight_by_cell.get(cell, 0.0) / max_weight
            value_noise = float(self._rng.normal(0.0, cal["value_noise_std"]))
            confidence_noise = float(self._rng.normal(0.0, cal["confidence_noise_std"]))
            out[cell] = Estimate(
                value=_clamp(density + value_noise, 0.0, 1.0),
                confidence=_clamp(cal["base_confidence"] + confidence_noise, 0.0, 1.0),
                age_minutes=cal["age_minutes"],
            )
        return out

    def _within(self, mapping: dict[str, Estimate], lat: float, lon: float, radius_km: float):
        out: dict[str, Estimate] = {}
        for cell, estimate in mapping.items():
            coords = self._coords(cell)
            if geo.great_circle_km(lat, lon, coords[0], coords[1]) <= radius_km:
                out[cell] = estimate
        return out

    def _coords(self, cell: str) -> tuple[float, float]:
        coords = self._cell_coords.get(cell)
        if coords is None:
            coords = geo.cell_centroid(cell)
            self._cell_coords[cell] = coords
        return coords

    # ------------------------------------------------------------------
    # RawSourcePort: weather, by coordinate. Works in any city.
    # ------------------------------------------------------------------

    def weather_at(self, lat: float, lon: float, minute: int) -> dict[str, Estimate]:
        self._refresh(minute, lat, lon)
        return dict(self._weather)

    # ------------------------------------------------------------------
    # RawSourcePort: the road network, two layers and nothing else
    # ------------------------------------------------------------------

    def travel_estimate(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, minute: int
    ) -> tuple[Estimate, Estimate]:
        """(km, minutes) as SEPARATE estimates, from exactly two layers:

            matrix   free-flow skeleton over the OSM drive graph, built once
                     offline and cached (see `osm_travel.py`)
            learned  correction by zone and hour, from the agent's OWN
                     completed trips (see `history.py`)

        There is no third layer and deliberately no external traffic feed: a
        commercial feed measures car probes, and a courier on a motorcycle
        is a different vehicle doing a different job. The agent's own trips
        are the correct measurement, not a cheap substitute for one.

        km and minutes never collapse into one number. The skeleton is where
        they decouple — a leg along a fast corridor and a leg of the same
        length through a residential grid are not the same trip.
        """
        cal = TRAVEL_SOURCE_CALIBRATION
        straight_km = geo.great_circle_km(from_lat, from_lon, to_lat, to_lon)
        if straight_km < cal["same_place_km"]:
            return (
                Estimate(value=0.0, confidence=1.0, age_minutes=0.0),
                Estimate(value=0.0, confidence=1.0, age_minutes=0.0),
            )

        to_cell = self._skeleton.nearest_cell(to_lat, to_lon)
        km, structural_minutes = self._structural_leg(
            from_lat, from_lon, to_cell, straight_km
        )

        correction = self.history.recall_travel_correction(to_cell, minute)
        if correction is None:
            minutes = structural_minutes
            minutes_confidence = cal["structural_minutes_confidence"]
        else:
            minutes = structural_minutes * correction.value
            minutes_confidence = correction.confidence

        return (
            Estimate(value=km, confidence=cal["km_confidence"], age_minutes=0.0),
            Estimate(
                value=max(cal["min_leg_minutes"], minutes),
                confidence=minutes_confidence,
                age_minutes=0.0,
            ),
        )

    def _structural_leg(
        self, from_lat: float, from_lon: float, to_cell: str, straight_km: float
    ) -> tuple[float, float]:
        """The skeleton's own answer, before anything is learned.

        The matrix is CELL-resolution — centroid to centroid — so what is
        read off it is two SCALE-FREE properties of the corridor and never
        an absolute distance:

            route factor    how much longer the road is than the crow flies
            corridor speed  how fast that road is, from the OSM road class

        Both are then applied to the ACTUAL endpoints. Using the matrix's
        kilometres directly instead inflates a sub-2 km trip by 37% and its
        time by 59% (see `TRAVEL_SOURCE_CALIBRATION` for the measurement),
        which is a resolution error rather than a fact about the city, and
        most of this job happens at short range.

        An unroutable pair, or two points in the same cell, has no
        centroid-to-centroid baseline to take a ratio against, so it falls
        back to a flat detour factor at a flat door-to-door speed. A courier
        always finds some way around; they do not refuse the trip.
        """
        from_cell = self._skeleton.nearest_cell(from_lat, from_lon)
        route_factor, scooter_kmh = self._corridor(from_cell, to_cell)
        km = straight_km * route_factor
        return km, km / scooter_kmh * 60.0

    def _corridor(self, from_cell: str, to_cell: str) -> tuple[float, float]:
        """(route factor, scooter km/h) for one corridor, memoised.

        Both are properties of the road between two cells and of nothing
        else — not of the minute, not of the endpoints — so they are
        computed once per pair and reused. That matters for the seven-second
        budget: weighing every reachable cell as a repositioning target asks
        about a hundred corridors in one decision, and doing the Dijkstra
        lookups and the trigonometry again each time took the p99 decision
        from under a millisecond to thirty.
        """
        cached = self._corridor_cache.get((from_cell, to_cell))
        if cached is not None:
            return cached

        cal = TRAVEL_SOURCE_CALIBRATION
        pair = self._skeleton.leg(from_cell, to_cell)
        corridor_km = self._centroid_km(from_cell, to_cell)
        speed_kmh = (
            self._skeleton.corridor_speed_kmh(from_cell, to_cell) if pair is not None else None
        )

        if pair is None or speed_kmh is None or corridor_km < cal["min_corridor_km"]:
            result = (cal["street_detour_factor"], cal["fallback_scooter_kmh"])
        else:
            result = (
                _clamp(pair[0] / corridor_km, cal["min_route_factor"], cal["max_route_factor"]),
                speed_kmh / cal["free_flow_to_scooter_factor"],
            )
        self._corridor_cache[(from_cell, to_cell)] = result
        return result

    def _centroid_km(self, from_cell: str, to_cell: str) -> float:
        """Centroid-to-centroid great-circle km, the baseline the matrix's
        own kilometres are a ratio against."""
        a = self._skeleton.centroid(from_cell)
        b = self._skeleton.centroid(to_cell)
        if a is None or b is None:
            return 0.0
        return geo.great_circle_km(a[0], a[1], b[0], b[1])

    def congestion_near(
        self, lat: float, lon: float, radius_km: float, minute: int
    ) -> dict[str, Estimate]:
        """Travel-time multipliers for cells within reach. Sparse on purpose,
        and the sparseness comes from the SOURCE, not from the radius: a
        courier's traffic app only meaningfully covers their own cell, its
        neighbours and a corridor. Asking about a wider radius does not
        conjure readings that do not exist."""
        self._refresh(minute, lat, lon)
        return self._within(self._congestion, lat, lon, radius_km)

    # ------------------------------------------------------------------
    # RawSourcePort: commercial density, from POIs. Also planetary.
    # ------------------------------------------------------------------

    def poi_density_near(self, lat: float, lon: float, radius_km: float) -> dict[str, Estimate]:
        """How much food commerce sits in each nearby cell.

        This is what makes "will this drop-off strand me?" answerable in a
        city the agent knows nothing else about. It is geography, not a live
        reading: what turns it into a belief about NOW is the agent's own
        hour-of-day rhythm, applied on its side of the port.
        """
        if self._minute is None:
            # Queried before any refresh: a courier looking at their map
            # app before checking anything else. Take one reading from
            # here, at the minute the shift starts.
            self._refresh(self._shift_start_min, lat, lon)
        return self._within(self._poi_density, lat, lon, radius_km)

    # ------------------------------------------------------------------
    # RawSourcePort: disruptions the courier could plausibly have heard of
    # ------------------------------------------------------------------

    def perceived_disruptions(
        self, lat: float, lon: float, minute: int
    ) -> tuple[PerceivedEvent, ...]:
        """Only what detectability allows.

        Built exclusively from `perceivable_events` via
        `build_perceived_events`. This adapter never reads `active_events`,
        never inspects `Event.is_active`/`Event.start_min`, and has no other
        path to the event timeline at all.
        """
        self._refresh(minute, lat, lon)
        return self._disruptions

    # ------------------------------------------------------------------
    # RawSourcePort: coordinates for the cell ids this port hands back
    # ------------------------------------------------------------------

    def cell_coords(self, cells: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        """Without these the agent holds ids it cannot place on a map, so it
        cannot tell a believed-busy zone on its way from one across the
        city. Somebody looking at the zones on their own app plainly knows
        where those zones are, so this leaks nothing."""
        return {cell: self._coords(cell) for cell in cells if cell}

    # ------------------------------------------------------------------
    # RawSourcePort: the agent's own accumulated history
    # ------------------------------------------------------------------

    def recall_kitchen(self, denue_id: str) -> Estimate | None:
        """What this branch's prep time has been. `None` means never visited.

        `denue_id` is an OPAQUE VENUE KEY here and nothing more: whatever
        stable identifier the platform prints next to the branch name. A
        courier plainly sees which branch they are being sent to, and
        remembering that this one is always slow is exactly the knowledge a
        good courier accumulates. No country-specific registry is consulted
        to interpret it, and none could be in another city.
        """
        return self.history.recall_kitchen(denue_id, self._minute or self._shift_start_min)

    def record_kitchen(self, denue_id: str, observed_minutes: float, minute: int) -> None:
        """Called on pickup. The only way a kitchen memory comes to exist."""
        self.history.record_kitchen(denue_id, observed_minutes, minute)

    def recall_eta_bias(self, cell: str) -> Estimate | None:
        """How much the app's ETA has lied in this zone. `None` until the
        agent has both driven trips and re-fitted on them."""
        return self.history.recall_eta_bias(cell)

    def record_trip(
        self,
        from_cell: str,
        to_cell: str,
        minute: int,
        promised_minutes: float,
        actual_minutes: float,
        actual_km: float,
    ) -> None:
        """Called on completion. Feeds the offline re-fit.

        The skeleton's own prediction for this pair is stored alongside, so
        the travel correction remains fittable later without needing the
        skeleton that produced it.
        """
        # The skeleton's cell-to-cell prediction, which is the right
        # resolution here: what the engine realised is also a cell-to-cell
        # leg, so the ratio of the two is apples to apples.
        structural = 0.0
        pair = self._skeleton.leg(from_cell, to_cell)
        if pair is not None:
            _route_factor, scooter_kmh = self._corridor(from_cell, to_cell)
            structural = pair[0] / scooter_kmh * 60.0
        self.history.record_trip(
            TripRecord(
                from_cell=from_cell,
                to_cell=to_cell,
                minute=minute,
                promised_minutes=promised_minutes,
                actual_minutes=actual_minutes,
                actual_km=actual_km,
                structural_minutes=structural,
            )
        )

    def refit(self) -> dict[str, int]:
        """Re-fit the relationship tables from accumulated history.

        Called BETWEEN shifts, never inside a decision: fitting is
        expensive and using is free, which is the split that makes a
        seven-second budget workable at all.
        """
        return self.history.refit()
