"""The belief state the agent assembles for itself, by asking.

Nothing is handed to this layer. It holds a `RawSourcePort` and queries what
it decides it needs: weather by coordinate, congestion within reach,
food-commerce density, disruptions it could plausibly have heard about,
coordinates for the cell ids those answers name, and its own memory.

WHY THE DIRECTION MATTERS, since the information often ends up the same.
The previous design had the engine build an `Observation` and push it at the
policy, which meant the simulator author chose what the agent needed to
know. Two consequences, and the second is fatal:

  - "Where does the agent get that traffic estimate?" answered "we gave it
    to him", which is not an answer a judge accepts.
  - There is no engine in Guadalajara. An agent shaped around being pushed
    beliefs does not degrade in a new city; it does not start.

This module is the whole of the agent's perception, so it is also the whole
of the portability surface. Every query below is answerable from a pair of
coordinates and an app screen, and `tests/agent/test_portability.py` fails
the build if that stops being true.

WHAT IS A PULL AND WHAT IS SELF-KNOWLEDGE. Position, fuel, earnings,
acceptance rate and minutes left in the shift come off `CourierSnapshot` —
a courier knows their own situation and never needs to ask anybody. The
app's heatmap and acceptance rate come off `PlatformView`, which is the
screen. Everything else is a query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.ports import CourierSnapshot, Estimate, PerceivedEvent, PlatformView

from src.agent.calibration import (
    DEMAND_PRIOR_CALIBRATION,
    MEAL_RHYTHM_PRIOR,
    QUERY_CALIBRATION,
    SURGE_EVENT_KIND,
)

_MINUTES_PER_DAY = 1440


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def meal_rhythm(minute: int) -> float:
    """How busy this hour of the day is, as a STRUCTURAL prior.

    Bimodal, because people eat lunch and then dinner — that is human
    behaviour, not a fact about one city, which is why it transfers. The
    exact peak hours and the exact depth of the afternoon trough are what
    the agent learns; the shape is what it arrives with.

    Coarse and round-numbered on purpose: this is a prior a courier would
    describe to you in a sentence, not a fitted curve. It is read off the
    agent's own clock (`minute`, wrapped at a day) and nothing else, so it
    needs no source at all — which is exactly why it survives being dropped
    in a city the agent knows nothing about.

    One table, not a weekday/weekend pair: minutes are a continuous counter
    from the shift's reference start and carry no day-of-week, so the agent
    cannot tell which it is. A weekday shape is the honest default.
    """
    minute_of_day = float(minute % _MINUTES_PER_DAY)
    table = MEAL_RHYTHM_PRIOR
    if minute_of_day <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if minute_of_day <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (minute_of_day - x0) / (x1 - x0)
    return table[-1][1]


@dataclass
class BeliefState:
    """What the agent believes at one minute, and where each belief came from.

    Every value is an `Estimate` with a confidence and an age, never a bare
    float. A policy that ignores `confidence` is choosing to; it is never
    handed certainty it does not have.
    """

    minute: int
    at_lat: float
    at_lon: float
    at_cell: str

    temp_c: Estimate
    apparent_c: Estimate
    precip_mm: Estimate

    # Per-cell travel-time multipliers, as sparse as the courier's traffic
    # app actually is. Nobody knows congestion across a whole city.
    traffic_by_cell: dict[str, Estimate]

    perceived_events: tuple[PerceivedEvent, ...]

    # Composed, not queried: food-commerce density from the map times the
    # agent's own hour-of-day prior, bumped where a perceived surge window
    # corroborates it. See `_compose_demand`.
    demand_by_cell: dict[str, Estimate]

    cell_coords: dict[str, tuple[float, float]]

    minutes_left_in_shift: int
    fuel_minutes_remaining: float

    _sources: object = field(repr=False, default=None)
    _travel_cache: dict = field(repr=False, default_factory=dict)

    # ------------------------------------------------------------------
    # Assembly: the pull itself
    # ------------------------------------------------------------------

    @classmethod
    def pull(
        cls, sources, view: PlatformView, courier: CourierSnapshot
    ) -> "BeliefState":
        """Ask every source this minute's questions and compose the answers.

        Order is deliberate but not load-bearing for correctness: a port
        that refreshes once per minute answers the same regardless of which
        question comes first, and the port owns that contract rather than
        this caller.
        """
        cal = QUERY_CALIBRATION
        minute = view.minute
        lat, lon = courier.lat, courier.lon

        weather = sources.weather_at(lat, lon, minute)
        traffic = sources.congestion_near(lat, lon, cal["congestion_radius_km"], minute)
        events = sources.perceived_disruptions(lat, lon, minute)
        poi = sources.poi_density_near(lat, lon, cal["poi_radius_km"])

        demand = cls._compose_demand(poi, events, minute)

        # Coordinates for exactly the cells the courier was told about this
        # minute, plus the one they are standing in — what their tools just
        # named, not a gazetteer of the city. Without these the policy holds
        # ids it cannot place on a map, so it cannot tell a believed-busy
        # zone on its way from one across town.
        named = tuple(sorted(set(traffic) | set(demand) | {courier.cell}))
        coords = sources.cell_coords(named)

        return cls(
            minute=minute,
            at_lat=lat,
            at_lon=lon,
            at_cell=courier.cell,
            temp_c=weather["temp_c"],
            apparent_c=weather["apparent_c"],
            precip_mm=weather["precip_mm"],
            traffic_by_cell=traffic,
            perceived_events=events,
            demand_by_cell=demand,
            cell_coords=coords,
            minutes_left_in_shift=courier.minutes_left_in_shift,
            fuel_minutes_remaining=courier.fuel_minutes_remaining,
            _sources=sources,
        )

    @staticmethod
    def _compose_demand(
        poi: dict[str, Estimate], events: tuple[PerceivedEvent, ...], minute: int
    ) -> dict[str, Estimate]:
        """Where the agent believes demand is, built rather than received.

        Two portable ingredients and one corroboration:

          - POI density says WHERE the food is. A map fact, planetary.
          - `meal_rhythm` says how busy this hour is. A structural prior
            about people, read off the agent's own clock.
          - A perceived surge window is real, specific, located information
            about right now, so it bumps the cells it touches above the
            generic rhythm baseline.

        The product is banded into a handful of levels, because a courier
        thinks in "busy / quiet / dead" rather than in three decimal places
        — and because a difference finer than a band is noise, which is what
        stops the repositioning branch chasing it.

        Confidence is the map's own confidence DISCOUNTED by how much the
        agent trusts its rhythm prior. Multiplying a solid fact by a guess
        yields a guess, and saying so is what makes the cold start
        conservative instead of confidently wrong.
        """
        cal = DEMAND_PRIOR_CALIBRATION
        rhythm = meal_rhythm(minute)
        buckets = max(int(cal["bands"]), 2)
        out: dict[str, Estimate] = {}
        for cell, density in poi.items():
            value = _clamp(density.value * rhythm, 0.0, 1.0)
            banded = round(value * (buckets - 1)) / (buckets - 1)
            out[cell] = Estimate(
                value=banded,
                confidence=_clamp(density.confidence * cal["rhythm_prior_confidence"], 0.0, 1.0),
                age_minutes=density.age_minutes,
            )
        for event in events:
            if event.kind != SURGE_EVENT_KIND:
                continue
            for cell in event.affects_cells:
                base = out.get(cell)
                out[cell] = Estimate(
                    value=_clamp((base.value if base else 0.0) + cal["surge_value_bump"], 0.0, 1.0),
                    confidence=_clamp(
                        max(base.confidence if base else 0.0, event.confidence)
                        + cal["surge_confidence_bump"],
                        0.0,
                        1.0,
                    ),
                    age_minutes=cal["surge_age_minutes"],
                )
        return out

    # ------------------------------------------------------------------
    # Queries the agent makes per offer, not per minute
    # ------------------------------------------------------------------

    def travel(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float
    ) -> tuple[Estimate, Estimate]:
        """(km, minutes) for one leg, from the port's two-layer travel model.

        Memoised for this minute: scoring a handful of offers and then
        weighing every reachable cell as a repositioning target asks the
        same questions repeatedly, and a decision has seven seconds.
        """
        key = (round(from_lat, 5), round(from_lon, 5), round(to_lat, 5), round(to_lon, 5))
        cached = self._travel_cache.get(key)
        if cached is None:
            cached = self._sources.travel_estimate(
                from_lat, from_lon, to_lat, to_lon, self.minute
            )
            self._travel_cache[key] = cached
        return cached

    def kitchen(self, venue_key: str, venue_name: str = "") -> Estimate | None:
        """What this branch's prep time has been, or `None` if never visited.

        `venue_key` is whatever stable identifier the platform prints next
        to the branch name — opaque to the agent, which never parses it.
        `venue_name` is the fallback for a platform that supplies no key at
        all: the name is then the only identifier on the card, and matching
        on it beats not matching.
        """
        estimate = self._sources.recall_kitchen(venue_key) if venue_key else None
        if estimate is None and venue_name:
            estimate = self._sources.recall_kitchen(venue_name)
        return estimate

    def app_eta_bias(self, cell: str) -> Estimate | None:
        """How much the app's ETA has lied in this zone, from the agent's own
        promised-against-realised record. `None` until it has driven trips
        there and re-fitted on them, which on shift one it has not."""
        if not cell:
            return None
        return self._sources.recall_eta_bias(cell)
