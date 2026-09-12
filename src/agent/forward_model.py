"""The agent's own forward model: what it believes will happen if it accepts.

This is the mirror image of the engine's travel oracle, and it is deliberately
NOT the same object. The engine's version is backed by the real network and
true conditions; this one is backed by whatever the courier happens to believe
at this minute, including the parts they believe wrongly. Same shape, different
source, and that symmetry is what lets the policy plan without seeing the
future.

It reads a `BeliefState` the agent assembled itself by querying raw sources,
rather than an `Observation` handed to it. Travel in particular is no longer
derived here from a straight line and an assumed speed: km and free-flow
minutes are PULLED, from a free-flow skeleton over the OSM drive graph plus a
correction fitted from the agent's own completed trips. What is still computed
here is what the agent does with that answer — the live congestion reading,
the rain slowdown, and the blend that stops a learned correction and a visible
jam being counted twice.

Everything returned here carries a confidence, because everything it is built
from does.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.ports import Estimate, PerceivedEvent

from src.agent.beliefs import BeliefState
from src.agent.calibration import (
    BELIEF_CALIBRATION,
    DESTINATION_CALIBRATION,
    HEATMAP_LEVEL_CONFIDENCE,
    HEATMAP_LEVEL_DEMAND,
    HANDLING_CALIBRATION,
    SAFETY_CALIBRATION,
    TRAVEL_CALIBRATION,
)
from src.agent.geometry import CellIndex


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
    """Reads one `BeliefState` and answers 'what would that cost me?'."""

    def __init__(self, beliefs: BeliefState, index: CellIndex) -> None:
        self._beliefs = beliefs
        self._index = index

    # -- weather ---------------------------------------------------------

    @property
    def rain_multiplier(self) -> float:
        precip = max(0.0, self._beliefs.precip_mm_per_hour.value)
        return min(
            TRAVEL_CALIBRATION["max_rain_multiplier"],
            1.0 + precip * TRAVEL_CALIBRATION["rain_slowdown_per_mm"],
        )

    def night_factor(self, minute: int) -> float:
        """0 in daylight, 1 deep at night. `minute` may be any absolute minute.

        Night WRAPS, and that has to be said in code rather than assumed.
        An earlier version was a single ramp over minute-of-day from
        `night_starts_minute` to `night_full_minute`; at 00:00 the
        minute-of-day reset to 0, dropped below the ramp's start, and the
        premium went to zero — so on the Night window (18:00-02:00) the two
        darkest hours of the shift, a quarter of it, were priced as broad
        daylight. Invisible except as a bad result, which is what it was.

        Four control points, in clock order: dusk, full dark, still dark,
        full light. Everything between full dark and still dark crosses
        midnight and is night at weight 1.
        """
        cal = SAFETY_CALIBRATION
        minute_of_day = float(minute % (24 * 60))
        if minute_of_day >= cal["night_starts_minute"]:
            return ramp(minute_of_day, cal["night_starts_minute"], cal["night_full_minute"])
        if minute_of_day <= cal["night_ends_minute"]:
            return 1.0
        if minute_of_day < cal["day_full_minute"]:
            return 1.0 - ramp(minute_of_day, cal["night_ends_minute"], cal["day_full_minute"])
        return 0.0

    def rain_risk_factor(self) -> float:
        return ramp(
            max(0.0, self._beliefs.precip_mm_per_hour.value),
            0.0,
            SAFETY_CALIBRATION["rain_risk_full_mm"],
        )

    def heat_exposure_factor(self) -> float:
        """0 below the comfort threshold, ramping to 1 at the punishing end.

        Heat is a physical cost, not a mood. Kept as a ramp rather than a
        binary "extreme heat" flag so a courier can price 38 C differently
        from 44 C, and read off this agent's own believed apparent
        temperature, which it queries by coordinate -- so it transfers to
        any city without knowing which one it is standing in.
        """
        return ramp(
            self._beliefs.apparent_c.value,
            SAFETY_CALIBRATION["heat_risk_onset_c"],
            SAFETY_CALIBRATION["heat_risk_full_c"],
        )

    # -- traffic and travel ----------------------------------------------

    def believed_traffic_multiplier_at(self, lat: float, lon: float) -> float:
        """Believed congestion at a COORDINATE, as a plain multiplier.

        Takes lat/lon rather than a cell id on purpose: an `OfferCard`
        carries coordinates, because that is what the app shows. Turning a
        point into a cell is the agent's own job, through its own index --
        so nothing here needs a cell vocabulary handed to it from outside.

        `traffic()` returns the belief with its confidence attached; this is
        the bare number, for a risk premium that must not re-weigh
        confidence because the final score already discounts it once.
        """
        return self.traffic(self._index.nearest(lat, lon)).value

    def traffic(self, cell: str) -> Belief:
        estimate = self._beliefs.traffic_by_cell.get(cell)
        if estimate is None:
            return Belief(
                value=TRAVEL_CALIBRATION["default_traffic_multiplier"],
                confidence=TRAVEL_CALIBRATION["default_traffic_confidence"],
                known=False,
            )
        return Belief(value=estimate.value, confidence=aged_confidence(estimate), known=True)

    def travel(self, from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> LegEstimate:
        """Believed km and minutes for one leg. The app's `eta_minutes` is
        never consulted here: it is the platform's optimism, not the
        courier's estimate.

        Three layers, in the order a courier would describe them:

          1. the geometry and the road, pulled from the agent's own OSM
             skeleton via `RawSourcePort.travel_estimate`;
          2. whatever its own completed trips taught it about this zone at
             this hour, already folded into that answer by the port;
          3. the traffic it can see RIGHT NOW, and the rain, applied here.

        Layers 2 and 3 estimate the same thing from different angles, so
        they are blended rather than multiplied. With nothing learned the
        live reading carries the whole correction, exactly as it did when
        this model derived everything itself. With a well-evidenced
        correction in hand the live multiplier is dropped, because "legs
        into this zone at this hour take 1.4x the skeleton" already contains
        the typical jam, and applying both counts it twice.
        """
        cal = TRAVEL_CALIBRATION
        km_estimate, minutes_estimate = self._beliefs.travel(from_lat, from_lon, to_lat, to_lon)
        if km_estimate.value <= 0.0:
            return LegEstimate(km=0.0, minutes=0.0, traffic_multiplier=1.0, confidence=1.0)

        origin = self.traffic(self._index.nearest(from_lat, from_lon))
        destination = self.traffic(self._index.nearest(to_lat, to_lon))
        live_multiplier = (origin.value + destination.value) / 2.0
        live_confidence = (origin.confidence + destination.confidence) / 2.0

        learned = ramp(
            minutes_estimate.confidence,
            cal["structural_only_confidence"],
            cal["fully_learned_confidence"],
        )
        multiplier = live_multiplier * (1.0 - learned) + learned
        minutes = minutes_estimate.value * multiplier * self.rain_multiplier
        return LegEstimate(
            km=km_estimate.value,
            minutes=max(cal["min_leg_minutes"], minutes),
            traffic_multiplier=multiplier,
            confidence=live_confidence * (1.0 - learned) + minutes_estimate.confidence * learned,
        )

    def reconcile_with_app_eta(
        self, leg: LegEstimate, cell: str, app_eta_minutes: float
    ) -> LegEstimate:
        """Blend the agent's own leg estimate with the app's ETA, corrected
        by how much the app has lied in this zone before.

        The purest arbitrage available to a courier, and entirely
        self-learned: it needs no external source at all. The app says
        eleven minutes and it took nineteen; after enough trips the agent
        knows that ratio per zone better than the platform will admit, and
        `app_eta x bias` is then a second, independent estimate of the same
        quantity. Two independent estimates are weighted by their own
        confidences rather than one overruling the other.

        Returns the leg unchanged when there is no fitted bias — which is
        the whole of shift one in a city the agent has never worked.
        """
        bias = self._beliefs.app_eta_bias(cell)
        if bias is None or app_eta_minutes <= 0.0 or leg.minutes <= 0.0:
            return leg
        corrected = app_eta_minutes * bias.value
        total = bias.confidence + leg.confidence
        if total <= 0.0:
            return leg
        weight = bias.confidence / total
        return LegEstimate(
            km=leg.km,
            minutes=max(
                TRAVEL_CALIBRATION["min_leg_minutes"],
                leg.minutes * (1.0 - weight) + corrected * weight,
            ),
            traffic_multiplier=leg.traffic_multiplier,
            confidence=max(leg.confidence, bias.confidence),
        )

    # -- kitchens --------------------------------------------------------

    def kitchen_wait(self, venue_key: str, restaurant_name: str = "") -> Belief:
        """How long this kitchen is believed to take.

        Queried per offer rather than received as a whole map. That is both
        what a courier actually does — you look up the one branch you are
        being sent to — and what keeps this answerable in a city where the
        agent has visited nothing at all.

        `venue_key` is `OfferCard.restaurant_denue_id`: an OPAQUE venue
        identifier the platform prints next to the branch, never parsed here
        and never resolved against any registry. A courier plainly sees
        which branch they are being sent to, and remembering that this one
        is always slow is exactly the knowledge a good courier accumulates.
        Looking memory up by `restaurant_name` instead — as an earlier
        revision did, before the card carried a key — meant the lookup NEVER
        hit and every kitchen was scored at the cold-start prior for a whole
        shift.

        `restaurant_name` stays as the fallback for a platform that supplies
        no key at all: the name is then the only identifier on the card, and
        matching on it beats not matching.
        """
        estimate = self._beliefs.kitchen(venue_key, restaurant_name)
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
        believe.

        The first source is the one that carries the shift now that
        `BeliefState.cell_coords` places every cell in `demand_by_cell`.
        Before it did, that lookup missed almost everywhere — the courier
        could only place a fine-grid cell by having stood in it — and the
        heatmap fallback was doing nearly all the work.
        """
        estimate = self._beliefs.demand_by_cell.get(cell)
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

        Only `BeliefState.perceived_events` is read. There is no other source of
        events reachable from this package, which is the point: a trace can only
        cite what this returns.
        """
        touched = {cell for cell in cells if cell}
        delays: list[PerceivedDelay] = []
        for event in self._beliefs.perceived_events:
            if not touched.intersection(event.affects_cells):
                continue
            believed = event.expected_delay_minutes.value * clamp(event.confidence, 0.0, 1.0)
            if believed <= 0.0:
                continue
            delays.append(PerceivedDelay(event=event, minutes=believed))
        return delays
