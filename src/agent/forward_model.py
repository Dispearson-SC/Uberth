"""The agent's own forward model: what it believes will happen if it accepts.

This is the mirror image of the engine's travel oracle, and it is deliberately
NOT the same object. The engine's version is backed by the real network and
true conditions; this one is backed by whatever the courier happens to believe
at this minute, including the parts they believe wrongly. Same shape, different
source, and that symmetry is what lets the policy plan without seeing the
future.

Everything returned here carries a confidence, because everything it is built
from does.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.ports import Estimate, Observation, PerceivedEvent

from src.agent.calibration import (
    BELIEF_CALIBRATION,
    DESTINATION_CALIBRATION,
    HEATMAP_LEVEL_CONFIDENCE,
    HEATMAP_LEVEL_DEMAND,
    HANDLING_CALIBRATION,
    SAFETY_CALIBRATION,
    TRAVEL_CALIBRATION,
)
from src.agent.geometry import CellIndex, haversine_km


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ramp(x: float, start: float, end: float) -> float:
    """0 at `start`, 1 at `end`, linear in between, clamped outside."""
    if start == end:
        return 1.0 if x >= end else 0.0
    return clamp((x - start) / (end - start), 0.0, 1.0)


def aged_confidence(estimate: Estimate) -> float:
    """Confidence discounted for staleness. A 20-minute-old reading is not fresh."""
    horizon = BELIEF_CALIBRATION["staleness_horizon_minutes"]
    floor = BELIEF_CALIBRATION["min_staleness_factor"]
    freshness = clamp(1.0 - max(0.0, estimate.age_minutes) / horizon, floor, 1.0)
    return clamp(estimate.confidence, 0.0, 1.0) * freshness


@dataclass(frozen=True)
class Belief:
    """A number the agent is working from, and how much it trusts it."""

    value: float
    confidence: float
    known: bool  # False when this is a prior, not something the courier learned


@dataclass(frozen=True)
class LegEstimate:
    """One movement. Kilometres and minutes stay separate, always."""

    km: float
    minutes: float
    traffic_multiplier: float
    confidence: float


@dataclass(frozen=True)
class PerceivedDelay:
    """A delay the courier can currently justify believing in."""

    event: PerceivedEvent
    minutes: float


class ForwardModel:
    """Reads one `Observation` and answers 'what would that cost me?'."""

    def __init__(self, observation: Observation, index: CellIndex) -> None:
        self._observation = observation
        self._index = index

    # -- weather ---------------------------------------------------------

    @property
    def rain_multiplier(self) -> float:
        precip = max(0.0, self._observation.precip_mm.value)
        return min(
            TRAVEL_CALIBRATION["max_rain_multiplier"],
            1.0 + precip * TRAVEL_CALIBRATION["rain_slowdown_per_mm"],
        )

    def night_factor(self, minute: int) -> float:
        """0 in daylight, 1 deep at night. Minute is minute-of-day."""
        minute_of_day = minute % (24 * 60)
        return ramp(
            float(minute_of_day),
            SAFETY_CALIBRATION["night_starts_minute"],
            SAFETY_CALIBRATION["night_full_minute"],
        )

    def rain_risk_factor(self) -> float:
        return ramp(
            max(0.0, self._observation.precip_mm.value),
            0.0,
            SAFETY_CALIBRATION["rain_risk_full_mm"],
        )

    # -- traffic and travel ----------------------------------------------

    def traffic(self, cell: str) -> Belief:
        estimate = self._observation.traffic_by_cell.get(cell)
        if estimate is None:
            return Belief(
                value=TRAVEL_CALIBRATION["default_traffic_multiplier"],
                confidence=TRAVEL_CALIBRATION["default_traffic_confidence"],
                known=False,
            )
        return Belief(value=estimate.value, confidence=aged_confidence(estimate), known=True)

    def travel(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> LegEstimate:
        """Believed km and minutes for one leg, using the courier's own traffic
        picture. The app's `eta_minutes` is never consulted: it is the platform's
        optimism, not the courier's estimate."""
        straight_km = haversine_km(from_lat, from_lon, to_lat, to_lon)
        if straight_km < TRAVEL_CALIBRATION["same_place_km"]:
            return LegEstimate(km=0.0, minutes=0.0, traffic_multiplier=1.0, confidence=1.0)

        km = straight_km * TRAVEL_CALIBRATION["street_detour_factor"]
        origin = self.traffic(self._index.nearest(from_lat, from_lon))
        destination = self.traffic(self._index.nearest(to_lat, to_lon))
        multiplier = (origin.value + destination.value) / 2.0
        free_flow_minutes = km / TRAVEL_CALIBRATION["free_flow_speed_kmh"] * 60.0
        minutes = free_flow_minutes * multiplier * self.rain_multiplier
        return LegEstimate(
            km=km,
            minutes=max(TRAVEL_CALIBRATION["min_leg_minutes"], minutes),
            traffic_multiplier=multiplier,
            confidence=(origin.confidence + destination.confidence) / 2.0,
        )

    # -- kitchens --------------------------------------------------------

    def kitchen_wait(self, denue_id: str, restaurant_name: str = "") -> Belief:
        """How long this kitchen is believed to take.

        `Observation.kitchen_minutes_by_denue_id` is keyed by DENUE id, and
        `OfferCard.restaurant_denue_id` carries that exact id — it is the
        stable join key the card exists to provide, and looking memory up by
        `restaurant_name` instead (as an earlier revision did, before the
        card carried an id) meant the lookup NEVER hit and every kitchen was
        scored at the cold-start prior for the whole shift. Learned kitchen
        speed was dead weight.

        `restaurant_name` stays as a fallback for the case where a platform
        supplies no id at all: the name is then the only identifier on the
        card, and matching on it is better than not matching.
        """
        memory = self._observation.kitchen_minutes_by_denue_id
        estimate = memory.get(denue_id) if denue_id else None
        if estimate is None and restaurant_name:
            estimate = memory.get(restaurant_name)
        if estimate is None:
            return Belief(
                value=HANDLING_CALIBRATION["default_kitchen_minutes"],
                confidence=HANDLING_CALIBRATION["default_kitchen_confidence"],
                known=False,
            )
        return Belief(value=estimate.value, confidence=aged_confidence(estimate), known=True)

    # -- demand ----------------------------------------------------------

    def demand(self, cell: str, lat: float | None = None, lon: float | None = None) -> Belief:
        """How busy the courier believes a place is.

        Three sources, best first: their own sense of demand for that exact
        cell; failing that, the level the app's heatmap shows over the point
        (lagged, coarse and quantised, but visible and locatable — see
        `HEATMAP_LEVEL_DEMAND`); failing that, a flat prior they barely
        believe. The middle one matters more than it looks: the fine-grid
        cell ids in `demand_by_cell` are ones the courier can only place by
        having stood in them, so early in a shift the first source misses
        almost everywhere.
        """
        estimate = self._observation.demand_by_cell.get(cell)
        if estimate is not None:
            return Belief(value=estimate.value, confidence=aged_confidence(estimate), known=True)
        if lat is not None and lon is not None:
            level = self._index.nearest_level(lat, lon)
            if level is not None:
                return Belief(
                    value=HEATMAP_LEVEL_DEMAND.get(level, DESTINATION_CALIBRATION["default_demand"]),
                    confidence=HEATMAP_LEVEL_CONFIDENCE,
                    known=True,
                )
        return Belief(
            value=DESTINATION_CALIBRATION["default_demand"],
            confidence=DESTINATION_CALIBRATION["default_demand_confidence"],
            known=False,
        )

    def dead_minutes(self, demand_value: float, typical_wait_minutes: float | None = None) -> float:
        """Unpaid minutes expected before the next worthwhile offer in a cell.

        This is the cost of being left somewhere nobody orders from: either you
        wait, or you ride out of it. Both are minutes the platform never counts.

        `typical_wait_minutes` is what the courier has MEASURED their own wait
        to be at an ordinary spot this shift. Passed in, the fixed
        calibration below becomes a ratio around it rather than an absolute
        claim — which matters, because a wait of "2 to 14 minutes" is a
        guess about a city, and how long you actually stand around depends
        entirely on how thick the offer flow is tonight. Omitted (early in a
        shift, before the courier has measured anything) it falls back to
        the fixed calibration.
        """
        busy = clamp(demand_value, 0.0, 1.0)
        empty_wait = DESTINATION_CALIBRATION["dead_minutes_at_zero_demand"]
        busy_wait = DESTINATION_CALIBRATION["dead_minutes_at_full_demand"]
        absolute = empty_wait + (busy_wait - empty_wait) * busy
        if typical_wait_minutes is None:
            return absolute
        typical = empty_wait + (busy_wait - empty_wait) * DESTINATION_CALIBRATION["default_demand"]
        if typical <= 0.0:
            return absolute
        return typical_wait_minutes * (absolute / typical)

    # -- events ----------------------------------------------------------

    def delays_on(self, cells: tuple[str, ...]) -> list[PerceivedDelay]:
        """Delays from events the courier can perceive RIGHT NOW.

        Only `observation.perceived_events` is read. There is no other source of
        events reachable from this package, which is the point: a trace can only
        cite what this returns.
        """
        touched = {cell for cell in cells if cell}
        delays: list[PerceivedDelay] = []
        for event in self._observation.perceived_events:
            if not touched.intersection(event.affects_cells):
                continue
            believed = event.expected_delay_minutes.value * clamp(event.confidence, 0.0, 1.0)
            if believed <= 0.0:
                continue
            delays.append(PerceivedDelay(event=event, minutes=believed))
        return delays
