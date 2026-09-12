"""Hand-built fixtures for policy tests.

Nothing here touches `src/world/`, `src/engine/`, a file, or a network. The
whole point of the port boundary is that a policy can be tested from plain
dataclasses, so these builders construct `PlatformView`, `Observation` and
`CourierSnapshot` by hand.

Geography is a small made-up grid over Monterrey. Cell ids are opaque to the
agent, so readable names are used here instead of real H3 indices.
"""

from __future__ import annotations

import math
from typing import Any

from src.core.ports import (
    CourierActivity,
    CourierSnapshot,
    Estimate,
    HeatCell,
    Observation,
    OfferCard,
    PerceivedEvent,
    PlatformView,
)

# Cell centroids. Roughly 1 km per 0.009 degrees at this latitude.
CELLS: dict[str, tuple[float, float]] = {
    "MTY-C": (25.6700, -100.3100),    # home cell, city centre
    "MTY-N": (25.7000, -100.3100),    # ~3.3 km north
    "MTY-NN": (25.7400, -100.3100),   # ~7.8 km north
    "MTY-FAR": (25.7800, -100.3100),  # ~12 km north
    "MTY-SC": (25.6600, -100.3100),   # ~1.1 km south of home, 4.4 km south of MTY-N
    "MTY-S": (25.6400, -100.3100),    # ~3.3 km south
    "MTY-E": (25.6700, -100.2700),    # ~4.0 km east
    "MTY-W": (25.6700, -100.3500),    # ~4.0 km west
}

HOME_CELL = "MTY-C"
HOME_LAT, HOME_LON = CELLS[HOME_CELL]

EARTH_RADIUS_KM = 6371.0088


def straight_km(a_cell: str, b_cell: str) -> float:
    """Great-circle km between two cell centroids (test-side helper only)."""
    lat1, lon1 = CELLS[a_cell]
    lat2, lon2 = CELLS[b_cell]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def sure(value: float, confidence: float = 0.9, age_minutes: float = 1.0) -> Estimate:
    return Estimate(value=value, confidence=confidence, age_minutes=age_minutes)


def make_offer(
    order_id: str,
    *,
    pickup_cell: str,
    dropoff_cell: str,
    payout_mxn: float,
    restaurant_name: str | None = None,
    eta_minutes: float | None = None,
    distance_km: float | None = None,
    surge_flag: bool = False,
    expires_in_seconds: int = 45,
) -> OfferCard:
    """One offer card. The app eta/distance default to optimistic values."""
    pickup_lat, pickup_lon = CELLS[pickup_cell]
    dropoff_lat, dropoff_lon = CELLS[dropoff_cell]
    leg_km = straight_km(pickup_cell, dropoff_cell)
    return OfferCard(
        order_id=order_id,
        pickup_lat=pickup_lat,
        pickup_lon=pickup_lon,
        dropoff_lat=dropoff_lat,
        dropoff_lon=dropoff_lon,
        payout_mxn=payout_mxn,
        eta_minutes=eta_minutes if eta_minutes is not None else max(6.0, leg_km * 2.2),
        distance_km=distance_km if distance_km is not None else leg_km * 1.2,
        surge_flag=surge_flag,
        restaurant_name=restaurant_name if restaurant_name is not None else "Cocina " + order_id,
        expires_in_seconds=expires_in_seconds,
    )


def make_heatmap(levels: dict[str, int] | None = None) -> tuple[HeatCell, ...]:
    levels = levels or {}
    return tuple(
        HeatCell(cell=cell, lat=lat, lon=lon, level=levels.get(cell, 2))
        for cell, (lat, lon) in CELLS.items()
    )


def make_view(
    *,
    minute: int,
    offers: tuple[OfferCard, ...] = (),
    heat_levels: dict[str, int] | None = None,
    acceptance_rate: float = 0.7,
    deliveries_completed: int = 4,
    earnings_shown_mxn: float = 300.0,
) -> PlatformView:
    return PlatformView(
        minute=minute,
        offers=offers,
        heatmap=make_heatmap(heat_levels),
        acceptance_rate=acceptance_rate,
        deliveries_completed=deliveries_completed,
        earnings_shown_mxn=earnings_shown_mxn,
    )


def make_observation(
    *,
    minute: int,
    at_cell: str = HOME_CELL,
    demand: dict[str, float] | None = None,
    demand_confidence: float = 0.8,
    traffic: dict[str, float] | None = None,
    traffic_confidence: float = 0.85,
    kitchen: dict[str, Estimate] | None = None,
    events: tuple[PerceivedEvent, ...] = (),
    minutes_left_in_shift: int = 240,
    km_to_home: float | None = None,
    fuel_minutes_remaining: float = 180.0,
    precip_mm: float = 0.0,
    temp_c: float = 24.0,
) -> Observation:
    at_lat, at_lon = CELLS[at_cell]
    demand = demand if demand is not None else {cell: 0.5 for cell in CELLS}
    traffic = traffic if traffic is not None else {cell: 1.0 for cell in CELLS}
    return Observation(
        minute=minute,
        at_lat=at_lat,
        at_lon=at_lon,
        at_cell=at_cell,
        temp_c=sure(temp_c),
        apparent_c=sure(temp_c + 1.0),
        precip_mm=sure(precip_mm),
        traffic_by_cell={c: sure(v, traffic_confidence) for c, v in traffic.items()},
        perceived_events=events,
        kitchen_minutes_by_denue_id=dict(kitchen or {}),
        demand_by_cell={c: sure(v, demand_confidence) for c, v in demand.items()},
        minutes_left_in_shift=minutes_left_in_shift,
        km_to_home=km_to_home if km_to_home is not None else straight_km(at_cell, HOME_CELL),
        fuel_minutes_remaining=fuel_minutes_remaining,
    )


def make_courier(
    *,
    minute: int,
    cell: str = HOME_CELL,
    activity: CourierActivity = CourierActivity.IDLE,
    earnings_mxn: float = 300.0,
    deliveries_completed: int = 4,
    km_traveled: float = 22.0,
    minutes_elapsed: float = 200.0,
    minutes_idle: float = 35.0,
    offers_seen: int = 20,
    offers_accepted: int = 14,
    fuel_minutes_remaining: float = 180.0,
    minutes_left_in_shift: int = 240,
) -> CourierSnapshot:
    lat, lon = CELLS[cell]
    return CourierSnapshot(
        minute=minute,
        lat=lat,
        lon=lon,
        cell=cell,
        activity=activity,
        earnings_mxn=earnings_mxn,
        deliveries_completed=deliveries_completed,
        km_traveled=km_traveled,
        minutes_elapsed=minutes_elapsed,
        minutes_idle=minutes_idle,
        carrying_order_ids=(),
        offers_seen=offers_seen,
        offers_accepted=offers_accepted,
        fuel_minutes_remaining=fuel_minutes_remaining,
        home_lat=HOME_LAT,
        home_lon=HOME_LON,
        minutes_left_in_shift=minutes_left_in_shift,
    )


def scenario(
    *,
    minute: int,
    offers: tuple[OfferCard, ...] = (),
    at_cell: str = HOME_CELL,
    demand: dict[str, float] | None = None,
    traffic: dict[str, float] | None = None,
    kitchen: dict[str, Estimate] | None = None,
    events: tuple[PerceivedEvent, ...] = (),
    minutes_left_in_shift: int = 240,
    fuel_minutes_remaining: float = 180.0,
    offers_seen: int = 20,
    offers_accepted: int = 14,
    heat_levels: dict[str, int] | None = None,
    precip_mm: float = 0.0,
) -> tuple[PlatformView, Observation, CourierSnapshot]:
    """The three arguments `Policy.decide` takes, consistently built."""
    courier = make_courier(
        minute=minute,
        cell=at_cell,
        offers_seen=offers_seen,
        offers_accepted=offers_accepted,
        fuel_minutes_remaining=fuel_minutes_remaining,
        minutes_left_in_shift=minutes_left_in_shift,
    )
    view = make_view(
        minute=minute,
        offers=offers,
        heat_levels=heat_levels,
        acceptance_rate=courier.acceptance_rate,
    )
    observation = make_observation(
        minute=minute,
        at_cell=at_cell,
        demand=demand,
        traffic=traffic,
        kitchen=kitchen,
        events=events,
        minutes_left_in_shift=minutes_left_in_shift,
        fuel_minutes_remaining=fuel_minutes_remaining,
        precip_mm=precip_mm,
    )
    return view, observation, courier


def make_event(
    event_id: str,
    *,
    kind: str = "crash",
    cells: tuple[str, ...] = (),
    delay_minutes: float = 8.0,
    confidence: float = 0.7,
) -> PerceivedEvent:
    lat, lon = CELLS[cells[0]] if cells else (None, None)
    return PerceivedEvent(
        event_id=event_id,
        kind=kind,
        lat=lat,
        lon=lon,
        affects_cells=cells,
        expected_delay_minutes=sure(delay_minutes, confidence),
        confidence=confidence,
    )


def trace_text(trace: Any) -> str:
    """Everything a judge would read in a trace, flattened into one string."""
    parts: list[str] = [trace.summary, trace.binding_constraint, str(trace.chosen_order_id)]
    for evaluation in trace.considered:
        parts.append(evaluation.order_id)
        parts.append(evaluation.rejected_because or "")
        for factor in evaluation.factors:
            parts.append(factor.label)
            parts.append(factor.note)
    return " | ".join(parts)
