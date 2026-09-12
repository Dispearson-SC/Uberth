"""The simulation engine: the clock that runs one courier through one shift.

This is the integration point of the whole system. `run_shift` ticks one
simulated minute at a time and, every tick: activates exogenous events
(street closures get a real ground-truth travel recompute), advances the
courier along whatever leg/handling step it is mid-way through, and shows the
courier whatever the app is offering RIGHT NOW — idle or mid-job — so the
policy can either take a job to start immediately or queue its next one. Every
offer lives for that minute only; nothing is stored (see
`calibration.OFFER_FLOW_CALIBRATION`).

Depends only on `src.core.ports` (the hexagonal contract) plus `src.world`
(ground truth, which the engine is allowed to touch — it is the driving
adapter). It never imports anything from `src.agent`, and nothing here
creates an import path that would let `src.agent` reach `src.world`.

THREE CONSTRAINTS BUILT IN FROM THE START (see module-level calibration in
`calibration.py` for the tunable numbers):

  1. HOME. The courier starts the shift at `home_cell` and the engine keeps
     running — past the nominal `shift_end_min` if it has to — until the
     courier is back home with no order in hand. That extra time is real
     and dilutes `mxn_per_hour`; a policy that ignores `Observation.km_to_home`
     late in the shift pays for it in the numbers, not in a rule.
  2. ACCEPTANCE RATE. `CourierSnapshot.offers_seen`/`offers_accepted` are
     tracked every tick; `calibration.acceptance_rate_offer_multiplier` is
     the explicit hook a `PlatformPort` reads to throttle a low-acceptance
     courier's offer feed (see `stubs.StubPlatform` for a minimal user of it).
  3. FUEL AS TIME. `fuel_minutes_remaining` drains while moving, never while
     idle/handling/waiting. When it hits zero the engine FORCES a `REFUEL`
     regardless of what the policy wants next, and the stop itself costs
     real minutes that earn nothing (`calibration.FUEL_CALIBRATION`).

DESIGN SIMPLIFICATIONS, stated plainly rather than left implicit:
  - The courier CARRIES at most one order at a time. No batching: two
    orders are never in the bag together, and the phase machine has no
    representation for that. They may however ACCEPT one job ahead
    (`OFFER_FLOW_CALIBRATION["max_jobs_in_hand"]`) and start it the instant
    the current one settles. Queueing, not batching — strictly sequential.
  - A job accepted ahead has its kitchen cooking from the ACCEPT minute, not
    from the minute the courier finally rides for it. That is the real
    reward for committing early, and `_arrive` nets it off the prep time
    exactly as it does for an immediate accept.
  - Mid-job, ACCEPT is the only action available. REPOSITION, REFUEL and
    REST are ignored while committed to a leg: a courier holding food does
    not detour to a petrol station.
  - Kitchen wait realised on arrival nets off the travel time already spent:
    `wait = max(0, order.prep_minutes - (arrival_minute - accepted_minute))`.
    The COURIER never gets early knowledge of this (`prep_minutes` is never
    exposed before arrival) — only the engine's own bookkeeping is early.
  - A `STREET_CLOSURE` event is applied once, at its start minute, via
    `NetworkTravelOracle.apply_closure`, and is never reopened — see that
    class's docstring for why (the world layer's own `TravelMatrix` exposes
    no "reopen" primitive).
  - Every other exogenous event type (`CRASH`, `CHECKPOINT`, `SURGE_WINDOW`,
    `MASS_EVENT`, `KITCHEN_BACKLOG`) is ground truth the courier may come to
    perceive once a real `EnrichmentPort` is wired in, but this engine slice
    does not itself apply their physical effects to travel/kitchen time —
    only `TrafficTick` congestion and `STREET_CLOSURE` do. Scoped this way
    deliberately: modelling every event type's ground-truth physical effect
    is a large surface the brief does not ask this slice to own.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import StrEnum

from src.core.ports import (
    Action,
    CourierActivity,
    CourierSnapshot,
    Decision,
    DecisionTrace,
    DeliveryRecord,
    EnrichmentPort,
    PlatformPort,
    PlatformView,
    Policy,
    RecorderPort,
    ShiftResult,
    TickRecord,
    TravelOracle,
)
from src.engine.calibration import (
    FUEL_CALIBRATION,
    HANDLING_CALIBRATION,
    OFFER_FLOW_CALIBRATION,
    PLAUSIBILITY_CALIBRATION,
)
from src.engine.travel import NetworkTravelOracle
from src.world import geo
from src.world.scenario import Scenario
from src.world.state import CourierState, DestinationRef, OrderState, RestaurantRef, TripLeg, TripPurpose
from src.world.timeline import EventType, OrderOffer

# A single simulated tick is always exactly one minute. Every counter in this
# module ("phase_remaining", "leg_progress" step size, fuel drain) assumes
# this; changing it would require rederiving every per-tick increment below.
TICK_MINUTES = 1.0

# Safety valve on the post-shift "finish up and go home" loop: never spin
# forever if a routing bug ever left the courier unable to reach home. Two
# full days of simulated minutes is far beyond any real recovery, so hitting
# this is itself a bug worth surfacing loudly rather than hanging silently.
MAX_POST_SHIFT_TICKS = 2 * 24 * 60


class _Phase(StrEnum):
    """Engine-internal courier phase. More granular than `CourierActivity`
    (which is the reduced view exposed to the policy) — several phases here
    map onto the same `CourierActivity` value; see `_activity_for`."""

    IDLE = "idle"
    MOVING = "moving"
    WAITING_KITCHEN = "waiting_kitchen"
    HANDLING_PICKUP = "handling_pickup"
    HANDLING_DROPOFF = "handling_dropoff"
    REFUELLING = "refuelling"
    RESTING = "resting"


@dataclass(frozen=True)
class _AcceptedJob:
    """An order the courier has accepted but has not started yet.

    The kitchen starts cooking at ACCEPTANCE, not at arrival, so
    `accepted_at_min` has to survive the wait in the queue — it is what
    `_arrive` nets the prep time off against. Accepting a job ten minutes
    before you can start it means the food is ten minutes further along
    when you get there, and that is exactly why committing early is worth
    something.
    """

    order: OrderOffer
    accepted_at_min: int


@dataclass
class _RuntimeState:
    """Engine-only mutable bookkeeping not covered by `CourierState`
    (the world's own schema deliberately excludes the agent-facing
    constraint fields introduced at the `ports.py` layer)."""

    fuel_minutes_remaining: float
    offers_seen: int = 0
    offers_accepted: int = 0
    phase: _Phase = _Phase.IDLE
    phase_remaining: float = 0.0
    leg_progress: float = 0.0
    current_order_offer: OrderOffer | None = None
    current_kitchen_wait: float = 0.0
    current_delivery_km: float = 0.0
    current_delivery_minutes: float = 0.0
    route_counter: int = 0
    current_route_id: str | None = None
    unpaid_km: float = 0.0
    # Accepted-but-not-started jobs, oldest first. Strictly sequential: the
    # courier finishes the job in hand, then starts the next one off this
    # list. Never carried simultaneously.
    queued_jobs: list[_AcceptedJob] = field(default_factory=list)


def _default_home_cell() -> str:
    """Deterministic default home location: the cell nearest the operating
    area's own centroid (see `calibration.DEFAULT_HOME_SELECTION`). A pure
    function of the on-disk cell fixture — no RNG stream is consumed, so
    picking this default never perturbs any other draw. Callers should pass
    `home_cell` explicitly once a real courier-profile source exists."""
    cell_index = geo.load_cell_index()
    mean_lat = float(cell_index["lat"].mean())
    mean_lon = float(cell_index["lon"].mean())
    best_cell: str | None = None
    best_distance = math.inf
    for row in cell_index.itertuples():
        distance = geo.great_circle_km(mean_lat, mean_lon, row.lat, row.lon)
        if distance < best_distance:
            best_distance = distance
            best_cell = row.cell
    if best_cell is None:
        raise RuntimeError("Cell index is empty; cannot pick a default home cell.")
    return str(best_cell)


def _activity_for(phase: _Phase, purpose: TripPurpose | None) -> CourierActivity:
    """Reduce the engine's granular `_Phase` to the `CourierActivity` the
    policy is allowed to see. `HANDLING_PICKUP` reads as still
    `WAITING_KITCHEN` (stationary at the restaurant, wrapping up) and
    `HANDLING_DROPOFF` reads as still `TO_CUSTOMER` (stationary at the
    customer, wrapping up) — `CourierActivity` has no dedicated "handling"
    value, and both are the natural tail of their respective bucket."""
    if phase is _Phase.REFUELLING:
        return CourierActivity.REFUELLING
    if phase is _Phase.RESTING:
        return CourierActivity.RESTING
    if phase in (_Phase.WAITING_KITCHEN, _Phase.HANDLING_PICKUP):
        return CourierActivity.WAITING_KITCHEN
    if phase is _Phase.HANDLING_DROPOFF:
        return CourierActivity.TO_CUSTOMER
    if phase is _Phase.MOVING:
        if purpose is TripPurpose.TO_RESTAURANT:
            return CourierActivity.TO_RESTAURANT
        if purpose is TripPurpose.TO_CUSTOMER:
            return CourierActivity.TO_CUSTOMER
        return CourierActivity.REPOSITIONING
    return CourierActivity.IDLE


def run_shift(
    scenario: Scenario,
    policy: Policy,
    platform: PlatformPort,
    enrichment: EnrichmentPort,
    recorder: RecorderPort | None = None,
    *,
    travel: TravelOracle | None = None,
    home_cell: str | None = None,
    courier_id: str = "courier-1",
) -> ShiftResult:
    """Run one courier through one shift and return the `ShiftResult`.

    Depends only on the ports (`Policy`, `PlatformPort`, `EnrichmentPort`,
    `RecorderPort`, `TravelOracle`) plus `src.world` ground truth — never on
    a concrete adapter. `travel` defaults to the real
    `NetworkTravelOracle` (expensive to build; pass one in and reuse it
    across runs against the same date rather than rebuilding per call).
    `home_cell` defaults to `_default_home_cell()`.

    Deterministic: every random draw the shift depends on already happened
    inside `scenario` (built from named RNG streams — see
    `src.world.scenario.rng_streams`); this function itself draws no
    randomness of its own, so the same `scenario` always plays out
    identically for a given `policy`.

    CONTRACT (this is the one wiring mistake that fails silently, so it is
    called out explicitly): `scenario.order_stream` MUST be the SAME order
    list — or a superset built from the same `build_order_stream(...)` call
    — that was used to construct `platform`. This function never trusts the
    `OfferCard` the platform shows as ground truth (it deliberately omits
    fields like `prep_minutes` and the fare decomposition); an ACCEPT is
    resolved back to the real `OrderOffer` by looking it up in
    `scenario.order_stream`, keyed by the `spawn_min` this engine recorded
    when the platform surfaced that card (which may be several minutes
    before the courier was free to accept it — see `_QueuedOffer`). If `scenario` is built
    without its `order_stream` (e.g. `Scenario(seed=..., date=...)` with the
    field left at its `Field(default_factory=list)` default) while the
    platform is built from a real order stream, every single ACCEPT will
    fail this lookup — this now raises `ValueError` immediately, naming the
    order_id and minute, rather than silently no-op'ing for the rest of the
    shift (which used to look exactly like "the courier never leaves IDLE").
    """
    if travel is None:
        travel = NetworkTravelOracle.from_fixtures(scenario.traffic_timeline)
    if home_cell is None:
        home_cell = _default_home_cell()
    home_lat, home_lon = geo.cell_centroid(home_cell)

    # Extra capabilities beyond the bare TravelOracle protocol, used only
    # when the concrete oracle actually provides them (a minimal test
    # double still runs the full engine correctly, just without real
    # closures/route geometry).
    apply_closure = getattr(travel, "apply_closure", None)
    route_polyline = getattr(travel, "route_polyline", None)

    orders_by_minute: dict[int, list[OrderOffer]] = defaultdict(list)
    for order in scenario.order_stream:
        orders_by_minute[order.spawn_min].append(order)
    street_closures = [event for event in scenario.events_timeline if event.type == EventType.STREET_CLOSURE]

    courier = CourierState(courier_id=courier_id, cell=home_cell, lat=home_lat, lon=home_lon)
    state = _RuntimeState(fuel_minutes_remaining=FUEL_CALIBRATION["tank_minutes"])
    routes: dict[str, list[tuple[float, float]]] = {}
    deliveries: list[DeliveryRecord] = []
    ticks: list[TickRecord] = []

    # Cells this engine can route to, and a memo of how a policy's
    # out-of-grid REPOSITION targets map onto them (see
    # `_resolve_target_cell`). Built lazily: a policy that only ever names
    # real operating cells never pays for either.
    known_cells: set[str] = set(getattr(getattr(travel, "matrix", None), "cell_order", []) or [])
    if not known_cells:
        known_cells = set(geo.load_cell_index()["cell"].tolist())
    target_cell_cache: dict[str, str | None] = {}

    # Optional capability beyond the bare EnrichmentPort protocol: a real
    # enrichment adapter learns kitchen speed from the courier's own
    # experience and needs to be told what that experience was. A minimal
    # test double that does not implement it simply never learns.
    record_kitchen_visit = getattr(enrichment, "record_kitchen_visit", None)

    # -- small helpers closing over the mutable state above -----------------

    def _current_purpose() -> TripPurpose | None:
        leg = courier.current_leg
        return leg.purpose if leg is not None else None

    def _build_snapshot(minute: int, activity: CourierActivity) -> CourierSnapshot:
        return CourierSnapshot(
            minute=minute,
            lat=courier.lat,
            lon=courier.lon,
            cell=courier.cell,
            activity=activity,
            earnings_mxn=courier.earnings_mxn,
            deliveries_completed=len(courier.completed_orders),
            km_traveled=courier.km_traveled,
            minutes_elapsed=courier.minutes_elapsed,
            minutes_idle=courier.minutes_idle,
            # Everything the courier is on the hook for: the job in progress
            # first, then anything accepted ahead of it. A policy reading
            # this against `activity` can tell "riding, nothing queued"
            # (a slot is open) from "riding, one queued" (full) — which is
            # exactly the state its commit-early decision turns on.
            carrying_order_ids=tuple(o.order_id for o in courier.active_orders)
            + tuple(job.order.order_id for job in state.queued_jobs),
            offers_seen=state.offers_seen,
            offers_accepted=state.offers_accepted,
            fuel_minutes_remaining=state.fuel_minutes_remaining,
            home_lat=home_lat,
            home_lon=home_lon,
            minutes_left_in_shift=max(0, scenario.shift_end_min - minute),
        )

    def _resolve_target_cell(cell: str) -> str | None:
        """Turn a policy's REPOSITION target into a cell this engine can
        actually route to.

        A policy's spatial vocabulary is whatever the app showed it, and the
        in-app heatmap is drawn on a COARSER grid than the operating cells
        the travel matrix is built from — so "ride to that district" is a
        perfectly sensible instruction that names a cell id the matrix has
        never heard of. Resolving it to the nearest operating cell to that
        district's centre is what a courier does with the same instruction.
        An unresolvable target returns None and the move is a no-op rather
        than a crash: a courier cannot ride somewhere that does not exist,
        but they also do not fall over when told to.
        """
        if cell in known_cells:
            return cell
        if cell in target_cell_cache:
            return target_cell_cache[cell]
        resolved: str | None = None
        try:
            lat, lon = geo.cell_centroid(cell)
        except Exception:
            target_cell_cache[cell] = None
            return None
        best = math.inf
        for row in geo.load_cell_index().itertuples():
            distance = geo.great_circle_km(lat, lon, row.lat, row.lon)
            if distance < best:
                best = distance
                resolved = str(row.cell)
        target_cell_cache[cell] = resolved
        return resolved

    def _jobs_in_hand() -> int:
        return len(courier.active_orders) + len(state.queued_jobs)

    def _has_room_for_another_job() -> bool:
        """Is the app willing to offer this courier anything at all?

        A courier already holding the maximum gets shown nothing, exactly as
        a real app stops offering once your queue is full. This keeps
        `offers_seen` meaning "offers this courier could actually have
        taken", which is the only denominator that makes the acceptance rate
        a fair number.
        """
        return _jobs_in_hand() < int(OFFER_FLOW_CALIBRATION["max_jobs_in_hand"])

    def _start_route(from_cell: str, to_cell: str) -> str | None:
        if route_polyline is None:
            return None
        try:
            polyline = route_polyline(from_cell, to_cell)
        except Exception:
            return None
        route_id = f"route-{state.route_counter:05d}"
        state.route_counter += 1
        routes[route_id] = polyline
        return route_id

    def _settle_delivery(minute: int) -> DeliveryRecord:
        order_state = courier.active_orders.pop()
        order_state.delivered_at_min = minute
        payout = order_state.total_payout_mxn
        courier.earnings_mxn += payout
        courier.completed_orders.append(order_state)
        record = DeliveryRecord(
            order_id=order_state.order_id,
            accepted_at_min=int(order_state.accepted_at_min or minute),
            delivered_at_min=minute,
            payout_mxn=payout,
            tip_mxn=order_state.tip_mxn,
            surge_locked=order_state.surge_multiplier,
            km=order_state.km_actual or 0.0,
            minutes=order_state.minutes_actual or 0.0,
            kitchen_wait_minutes=state.current_kitchen_wait,
        )
        deliveries.append(record)
        state.current_order_offer = None
        state.current_kitchen_wait = 0.0
        return record

    def _arrive(minute: int, leg: TripLeg) -> None:
        courier.cell = leg.to_cell
        order_offer = state.current_order_offer
        if leg.purpose is TripPurpose.TO_RESTAURANT and order_offer is not None:
            courier.lat, courier.lon = order_offer.origin_lat, order_offer.origin_lon
            order_state = courier.active_orders[-1]
            elapsed_since_accept = minute - (order_state.accepted_at_min or minute)
            # Kitchen wait realised only on arrival: the kitchen started
            # cooking at accept time (not on the courier's schedule), so
            # travel time already spent nets off against prep_minutes. The
            # courier never learns `prep_minutes` before this moment.
            wait = max(0.0, order_offer.prep_minutes - elapsed_since_accept)
            state.current_kitchen_wait = wait
            # What the COURIER measures, on their own clock: the food was
            # ready `elapsed + wait` minutes after they accepted. When they
            # waited, that is exactly `prep_minutes`. When they arrived to
            # find it already done, it is `elapsed` — an over-estimate they
            # genuinely cannot see past, which is the honest reading. Never
            # `prep_minutes` directly: that would hand the courier a number
            # nobody told them.
            if record_kitchen_visit is not None:
                record_kitchen_visit(
                    order_offer.restaurant_denue_id, float(elapsed_since_accept + wait), minute
                )
            if wait > 0:
                state.phase = _Phase.WAITING_KITCHEN
                state.phase_remaining = wait
            else:
                state.phase = _Phase.HANDLING_PICKUP
                state.phase_remaining = HANDLING_CALIBRATION["pickup_minutes"]
        elif leg.purpose is TripPurpose.TO_CUSTOMER and order_offer is not None:
            courier.lat, courier.lon = order_offer.dest_lat, order_offer.dest_lon
            state.current_delivery_km = leg.km
            state.current_delivery_minutes = leg.minutes
            state.phase = _Phase.HANDLING_DROPOFF
            state.phase_remaining = HANDLING_CALIBRATION["dropoff_minutes"]
        else:
            # REPOSITION, including the mandatory unpaid trip home.
            courier.lat, courier.lon = geo.cell_centroid(leg.to_cell)
            state.phase = _Phase.IDLE
        courier.current_leg = None
        state.current_route_id = None
        state.leg_progress = 0.0

    def _advance_phase(minute: int) -> DeliveryRecord | None:
        """Advance whatever the courier is mid-way through by exactly one
        tick. Returns the `DeliveryRecord` if a delivery settled this tick."""
        phase = state.phase
        if phase is _Phase.REFUELLING:
            state.phase_remaining -= TICK_MINUTES
            courier.minutes_idle += TICK_MINUTES
            if state.phase_remaining <= 0.0:
                state.fuel_minutes_remaining = FUEL_CALIBRATION["tank_minutes"]
                state.phase = _Phase.IDLE
            return None
        if phase is _Phase.RESTING:
            state.phase_remaining -= TICK_MINUTES
            courier.minutes_idle += TICK_MINUTES
            if state.phase_remaining <= 0.0:
                state.phase = _Phase.IDLE
            return None
        if phase is _Phase.WAITING_KITCHEN:
            state.phase_remaining -= TICK_MINUTES
            if state.phase_remaining <= 0.0:
                state.phase = _Phase.HANDLING_PICKUP
                state.phase_remaining = HANDLING_CALIBRATION["pickup_minutes"]
            return None
        if phase is _Phase.HANDLING_PICKUP:
            state.phase_remaining -= TICK_MINUTES
            if state.phase_remaining <= 0.0:
                order_state = courier.active_orders[-1]
                order_state.picked_up_at_min = minute
                order_offer = state.current_order_offer
                assert order_offer is not None
                km, minutes = travel.travel(courier.cell, order_offer.dest_cell, minute)
                minutes = max(minutes, HANDLING_CALIBRATION["min_leg_minutes"])
                leg = TripLeg(
                    from_cell=courier.cell,
                    to_cell=order_offer.dest_cell,
                    km=km,
                    minutes=minutes,
                    purpose=TripPurpose.TO_CUSTOMER,
                    start_min=minute,
                )
                courier.current_leg = leg
                state.leg_progress = 0.0
                state.phase = _Phase.MOVING
                state.current_route_id = _start_route(courier.cell, order_offer.dest_cell)
            return None
        if phase is _Phase.HANDLING_DROPOFF:
            state.phase_remaining -= TICK_MINUTES
            if state.phase_remaining <= 0.0:
                order_state = courier.active_orders[-1]
                order_state.km_actual = state.current_delivery_km
                order_state.minutes_actual = state.current_delivery_minutes
                record = _settle_delivery(minute)
                state.phase = _Phase.IDLE
                # Chain straight into whatever was accepted ahead: a courier
                # with their next job already in hand does not stand around
                # waiting to be asked. This is where committing early pays
                # for itself — in idle minutes that never happen.
                _start_next_job(minute)
                return record
            return None
        if phase is _Phase.MOVING:
            leg = courier.current_leg
            assert leg is not None
            total_minutes = max(leg.minutes, HANDLING_CALIBRATION["min_leg_minutes"])
            step = min(TICK_MINUTES / total_minutes, 1.0 - state.leg_progress)
            state.leg_progress += step
            km_delta = leg.km * step
            courier.km_traveled += km_delta
            if leg.purpose is TripPurpose.REPOSITION:
                state.unpaid_km += km_delta
            state.fuel_minutes_remaining -= TICK_MINUTES
            if state.leg_progress >= 1.0 - 1e-9:
                _arrive(minute, leg)
            return None
        # IDLE: nothing to advance; genuinely unproductive time.
        courier.minutes_idle += TICK_MINUTES
        return None

    def _start_next_job(minute: int) -> None:
        """Pull the next accepted job off the queue and ride for its
        restaurant. Called the instant a delivery settles, which is what
        removes the idle gap between chained deliveries — and the reason a
        policy that commits early earns those minutes back."""
        if courier.active_orders or not state.queued_jobs:
            return
        job = state.queued_jobs.pop(0)
        order = job.order
        order_state = OrderState(
            order_id=order.order_id,
            restaurant=RestaurantRef(
                denue_id=order.restaurant_denue_id,
                cell=order.origin_cell,
                lat=order.origin_lat,
                lon=order.origin_lon,
            ),
            destination=DestinationRef(cell=order.dest_cell, lat=order.dest_lat, lon=order.dest_lon),
            base_payout_mxn=order.gross_payout_mxn,
            # Surge LOCKED at acceptance, never re-evaluated at delivery — a
            # courier cannot wait for the multiplier to rise while holding
            # food, and cannot wait for it to rise while queued either.
            surge_multiplier=order.surge_at_spawn,
            surge_locked=True,
            tip_mxn=order.tip_mxn,
            # The ACCEPTANCE minute, not the minute the ride starts: the
            # kitchen has been cooking since the courier tapped accept.
            accepted_at_min=job.accepted_at_min,
        )
        courier.active_orders.append(order_state)
        state.current_order_offer = order
        km, minutes = travel.travel(courier.cell, order.origin_cell, minute)
        minutes = max(minutes, HANDLING_CALIBRATION["min_leg_minutes"])
        leg = TripLeg(
            from_cell=courier.cell,
            to_cell=order.origin_cell,
            km=km,
            minutes=minutes,
            purpose=TripPurpose.TO_RESTAURANT,
            start_min=minute,
        )
        courier.current_leg = leg
        state.leg_progress = 0.0
        state.phase = _Phase.MOVING
        state.current_route_id = _start_route(courier.cell, order.origin_cell)

    def _apply_decision(decision: Decision, minute: int, offered_order_ids: frozenset[str], *,
                        free_to_act: bool) -> None:
        """Apply one policy decision.

        `free_to_act` is False when the courier is mid-job: the app can push
        them their NEXT order and they can tap accept at a red light, but
        they cannot reposition, refuel or rest while committed to a leg.
        Those actions are simply not available, so they are ignored rather
        than queued — a policy asking for them mid-job gets a no-op, exactly
        as it would from the real app.
        """
        if decision.action == Action.ACCEPT and decision.order_id is not None and _has_room_for_another_job():
            # Offers are fresh only: an order reaches the courier during the
            # minute it spawns and no other, so the ground-truth lookup is
            # keyed on THIS minute. There is no offer store to consult.
            order = next((o for o in orders_by_minute.get(minute, []) if o.order_id == decision.order_id), None)
            if order is None and decision.order_id in offered_order_ids:
                # The platform showed this exact order_id this exact tick
                # (it is in `view.offers`), so a failed lookup here is NOT a
                # stale/invalid reference from the policy — it means
                # `scenario.order_stream` (what this resolution is keyed on)
                # does not contain the order the platform's own index was
                # built from. This happens when a caller constructs
                # `Scenario(...)` without threading the same order list into
                # `order_stream=...` that was used to build the `PlatformPort`
                # — `Scenario.order_stream` then silently defaults to `[]`
                # and EVERY accept fails forever, with no other symptom than
                # an all-idle, all-zero shift. Fail loudly here instead of
                # letting that run for hours before anyone notices.
                raise ValueError(
                    f"ACCEPT for order_id={decision.order_id!r} at minute={minute} was shown by the "
                    f"platform (it is in this tick's PlatformView.offers) but does not exist in "
                    f"scenario.order_stream at spawn_min={minute}. run_shift resolves an accepted "
                    "order's ground truth (restaurant location, prep time, fare, surge, tip) from "
                    "scenario.order_stream, keyed by spawn_min — it is NOT derived from the "
                    "PlatformPort's OfferCard. Pass the exact same order list into both "
                    "Scenario(..., order_stream=...) and whatever built the PlatformPort "
                    "(e.g. StubPlatform(orders_by_minute)); they must be the same collection."
                )
            if order is not None:
                # Accepted. The kitchen starts cooking NOW, whether the
                # courier can ride for it now or has to finish the job in
                # hand first — which is the entire value of committing
                # early, and the entire risk of committing to the wrong one.
                state.queued_jobs.append(_AcceptedJob(order=order, accepted_at_min=minute))
                state.offers_accepted += 1
                if state.phase is _Phase.IDLE and not courier.active_orders:
                    _start_next_job(minute)
        elif not free_to_act:
            # Mid-job: nothing but ACCEPT is available. No-op.
            return
        elif decision.action == Action.REPOSITION and decision.target_cell is not None:
            target_cell = _resolve_target_cell(decision.target_cell)
            if target_cell is None or target_cell == courier.cell:
                return
            km, minutes = travel.travel(courier.cell, target_cell, minute)
            minutes = max(minutes, HANDLING_CALIBRATION["min_leg_minutes"])
            leg = TripLeg(
                from_cell=courier.cell,
                to_cell=target_cell,
                km=km,
                minutes=minutes,
                purpose=TripPurpose.REPOSITION,
                start_min=minute,
            )
            courier.current_leg = leg
            state.leg_progress = 0.0
            state.phase = _Phase.MOVING
            state.current_route_id = _start_route(courier.cell, target_cell)
        elif decision.action == Action.REFUEL:
            state.phase = _Phase.REFUELLING
            state.phase_remaining = FUEL_CALIBRATION["refuel_stop_minutes"]
        elif decision.action == Action.REST:
            state.phase = _Phase.RESTING
            state.phase_remaining = HANDLING_CALIBRATION["rest_minutes"]
        # REJECT / HOLD / an invalid ACCEPT|REPOSITION target: no-op, stays idle.

    def _record_tick(minute: int, decision: Decision | None, offers_shown: int, delivered: DeliveryRecord | None,
                      perceived_event_ids: tuple[str, ...]) -> None:
        activity = _activity_for(state.phase, _current_purpose())
        snapshot = _build_snapshot(minute, activity)
        moving = state.phase is _Phase.MOVING
        tick = TickRecord(
            minute=minute,
            courier=snapshot,
            decision=decision,
            offers_shown=offers_shown,
            delivered=delivered,
            route_id=state.current_route_id if moving else None,
            route_progress=state.leg_progress if moving else 0.0,
            perceived_event_ids=perceived_event_ids,
        )
        ticks.append(tick)
        if recorder is not None:
            recorder.record(tick)

    # -- main in-shift loop ---------------------------------------------------

    for minute in range(scenario.shift_start_min, scenario.shift_end_min):
        # 1. Activate exogenous events. Only STREET_CLOSURE forces a real
        #    ground-truth travel recompute (see NetworkTravelOracle).
        if apply_closure is not None:
            for event in street_closures:
                if event.start_min == minute:
                    apply_closure(event)

        decision: Decision | None = None

        # 2. Forced refuel overrides everything else, but only once
        #    physically free to stop (idle — not mid-leg, mid-wait, or
        #    mid-handling; a courier cannot teleport to a gas station).
        if state.fuel_minutes_remaining <= 0.0 and state.phase is _Phase.IDLE:
            state.phase = _Phase.REFUELLING
            state.phase_remaining = FUEL_CALIBRATION["refuel_stop_minutes"]
            decision = Decision(
                action=Action.REFUEL,
                order_id=None,
                target_cell=None,
                trace=DecisionTrace(
                    minute=minute,
                    considered=(),
                    chosen_order_id=None,
                    threshold_mxn_per_hour=0.0,
                    binding_constraint="fuel",
                    summary="Forced refuel: the tank ran out.",
                ),
            )

        # 3. Enrichment runs every tick regardless of activity — a courier's
        #    own tools keep working mid-delivery, and so does their phone.
        pre_snapshot = _build_snapshot(minute, _activity_for(state.phase, _current_purpose()))
        observation = enrichment.observe(minute, pre_snapshot)
        perceived_event_ids = tuple(pe.event_id for pe in observation.perceived_events)

        # 4. Offer flow. The app shows a FLOW of offers, and each one lives
        #    for this minute only — accept it now or somebody else takes it.
        #    It pushes them whether the courier is idle or mid-job (that is
        #    how a next order gets queued), and stops entirely once the
        #    courier is holding all they may hold. Nothing is ever stored
        #    for later: see `calibration.OFFER_FLOW_CALIBRATION`.
        offers_shown = 0
        if decision is None:  # not already commandeered by a forced refuel
            free_to_act = state.phase is _Phase.IDLE
            view: PlatformView | None = None
            if _has_room_for_another_job():
                view = platform.view_at(minute, pre_snapshot)
                state.offers_seen += len(view.offers)
            elif free_to_act:
                # Idle with a full queue cannot happen (an accepted job
                # starts the moment the courier is free), but a policy is
                # still owed a view to act on if it ever does.
                view = replace(platform.view_at(minute, pre_snapshot), offers=())

            offers_shown = len(view.offers) if view is not None else 0
            if view is not None and (free_to_act or view.offers):
                decision = policy.decide(view, observation, pre_snapshot)
                offered_order_ids = frozenset(offer.order_id for offer in view.offers)
                _apply_decision(decision, minute, offered_order_ids, free_to_act=free_to_act)

        delivered = _advance_phase(minute)
        courier.minutes_elapsed += TICK_MINUTES
        _record_tick(minute, decision, offers_shown, delivered, perceived_event_ids)

    # -- post-shift wrap-up: finish the last order, then go home (HOME) -----

    post_shift_minute = scenario.shift_end_min
    for _ in range(MAX_POST_SHIFT_TICKS):
        if (
            state.phase is _Phase.IDLE
            and courier.cell == home_cell
            and not courier.active_orders
            and not state.queued_jobs
        ):
            break

        decision = None
        if state.phase is _Phase.IDLE and not courier.active_orders and state.queued_jobs:
            # An order accepted before the bell still has to be delivered.
            # Committing early late in the shift buys chained minutes; it
            # also buys unpaid overtime, and the courier pays for both.
            _start_next_job(post_shift_minute)
            decision = Decision(
                action=Action.ACCEPT,
                order_id=courier.active_orders[-1].order_id,
                target_cell=None,
                trace=DecisionTrace(
                    minute=post_shift_minute,
                    considered=(),
                    chosen_order_id=courier.active_orders[-1].order_id,
                    threshold_mxn_per_hour=0.0,
                    binding_constraint="none",
                    summary="Shift over: starting an order accepted before the bell.",
                ),
            )
        elif state.fuel_minutes_remaining <= 0.0 and state.phase is _Phase.IDLE:
            state.phase = _Phase.REFUELLING
            state.phase_remaining = FUEL_CALIBRATION["refuel_stop_minutes"]
            decision = Decision(
                action=Action.REFUEL,
                order_id=None,
                target_cell=None,
                trace=DecisionTrace(
                    minute=post_shift_minute,
                    considered=(),
                    chosen_order_id=None,
                    threshold_mxn_per_hour=0.0,
                    binding_constraint="fuel",
                    summary="Forced refuel: the tank ran out on the way home.",
                ),
            )
        elif state.phase is _Phase.IDLE:
            # The shift is over: no new offers, no new decisions — the
            # engine itself drives the mandatory unpaid trip home.
            km, minutes = travel.travel(courier.cell, home_cell, post_shift_minute)
            minutes = max(minutes, HANDLING_CALIBRATION["min_leg_minutes"])
            leg = TripLeg(
                from_cell=courier.cell,
                to_cell=home_cell,
                km=km,
                minutes=minutes,
                purpose=TripPurpose.REPOSITION,
                start_min=post_shift_minute,
            )
            courier.current_leg = leg
            state.leg_progress = 0.0
            state.phase = _Phase.MOVING
            state.current_route_id = _start_route(courier.cell, home_cell)
            decision = Decision(
                action=Action.REPOSITION,
                order_id=None,
                target_cell=home_cell,
                trace=DecisionTrace(
                    minute=post_shift_minute,
                    considered=(),
                    chosen_order_id=None,
                    threshold_mxn_per_hour=0.0,
                    binding_constraint="home",
                    summary="Shift over: returning home (unpaid).",
                ),
            )

        delivered = _advance_phase(post_shift_minute)
        courier.minutes_elapsed += TICK_MINUTES
        _record_tick(post_shift_minute, decision, 0, delivered, ())
        post_shift_minute += 1
    else:
        raise RuntimeError(
            f"Courier did not reach home within {MAX_POST_SHIFT_TICKS} post-shift minutes; "
            "this indicates a routing bug, not a slow shift."
        )

    return ShiftResult(
        policy_name=getattr(policy, "name", policy.__class__.__name__),
        seed=scenario.seed,
        shift_start_min=scenario.shift_start_min,
        shift_end_min=scenario.shift_end_min,
        earnings_mxn=courier.earnings_mxn,
        deliveries_completed=len(courier.completed_orders),
        km_traveled=courier.km_traveled,
        minutes_elapsed=courier.minutes_elapsed,
        minutes_idle=courier.minutes_idle,
        offers_seen=state.offers_seen,
        offers_accepted=state.offers_accepted,
        ticks=ticks,
        deliveries=deliveries,
        routes=routes,
        unpaid_km=state.unpaid_km,
    )


def self_check(result: ShiftResult) -> None:
    """Plausibility self-check: the honesty gate for the whole simulator.

    A reasonable policy over an 8-hour shift should land near the bounds in
    `calibration.PLAUSIBILITY_CALIBRATION` (100-150 MXN/h gross,
    2-3 deliveries/hour, 2-6 km average trip, 15-40% idle). Raises
    `AssertionError`, naming the actual number, on the first bound that does
    not hold. Never loosen these bounds to make a run "pass" — if a policy
    fails this, the engine or its calibration is broken and everything
    downstream is fiction.
    """
    cal = PLAUSIBILITY_CALIBRATION

    mxn_per_hour = result.mxn_per_hour
    assert cal["mxn_per_hour_min"] <= mxn_per_hour <= cal["mxn_per_hour_max"], (
        f"mxn_per_hour {mxn_per_hour:.2f} outside plausible "
        f"[{cal['mxn_per_hour_min']}, {cal['mxn_per_hour_max']}]"
    )

    deliveries_per_hour = result.deliveries_per_hour
    assert cal["deliveries_per_hour_min"] <= deliveries_per_hour <= cal["deliveries_per_hour_max"], (
        f"deliveries_per_hour {deliveries_per_hour:.2f} outside plausible "
        f"[{cal['deliveries_per_hour_min']}, {cal['deliveries_per_hour_max']}]"
    )

    if result.deliveries:
        avg_trip_km = sum(d.km for d in result.deliveries) / len(result.deliveries)
        assert cal["avg_trip_km_min"] <= avg_trip_km <= cal["avg_trip_km_max"], (
            f"avg_trip_km {avg_trip_km:.2f} outside plausible "
            f"[{cal['avg_trip_km_min']}, {cal['avg_trip_km_max']}]"
        )

    idle_fraction = result.minutes_idle / result.minutes_elapsed if result.minutes_elapsed else 0.0
    assert cal["idle_fraction_min"] <= idle_fraction <= cal["idle_fraction_max"], (
        f"idle_fraction {idle_fraction:.2f} outside plausible "
        f"[{cal['idle_fraction_min']}, {cal['idle_fraction_max']}]"
    )
