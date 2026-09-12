"""Engine-owned calibration knobs.

Everything in this file is a CALIBRATION KNOB, not measured data — exactly
like the calibration dicts in `src/world/traffic.py`, `weather.py`,
`events.py`, `demand.py` and `surge.py`. Nothing here comes from a published
platform document; every value is tuned by hand for a plausible 8-hour
Monterrey courier shift and is meant to be retuned in one place if the
plausibility self-check in `engine.py` fails.

Kept separate from `engine.py` so every runtime constant the clock depends on
is auditable from one small file, the same way the world layer keeps its own
calibration dicts apart from the mechanism that reads them.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fuel as TIME, not pesos (constraint #3 in the brief).
#
# A tank is finite and measured directly in minutes of active driving, not
# litres or pesos — this simulator never prices fuel, it only ever costs the
# courier wall-clock time. `tank_minutes` is sized so a courier normally
# needs 0-2 refuels across an 8-hour (480-minute) shift, assuming roughly
# 60-70% of the shift is spent actually moving (the rest idle/handling).
# ---------------------------------------------------------------------------
FUEL_CALIBRATION: dict[str, float] = {
    "tank_minutes": 240.0,  # minutes of active driving before the tank empties
    "refuel_stop_minutes": 10.0,  # real time cost of a refuel stop; earns nothing
}

# ---------------------------------------------------------------------------
# Handling time: the small, unavoidable stops around every delivery.
# ---------------------------------------------------------------------------
HANDLING_CALIBRATION: dict[str, float] = {
    "pickup_minutes": 2.0,  # walk in, confirm the order, walk out
    "dropoff_minutes": 1.5,  # hand off, get confirmation
    "rest_minutes": 15.0,  # fixed duration of a deliberate REST action
    "min_leg_minutes": 1.0,  # a leg always takes at least one simulated tick
}

# ---------------------------------------------------------------------------
# Acceptance rate (constraint #2 in the brief).
#
# The platform hook: real gig platforms throttle couriers whose acceptance
# rate drops too low — fewer offers, and (per product common-knowledge, not
# a published number) worse ones. This engine implements the tracking
# (`CourierSnapshot.offers_seen` / `offers_accepted`) and this one explicit
# penalty curve as the hook a platform adapter reads. `StubPlatform` in
# `stubs.py` applies it as a straight offer-COUNT throttle only — it never
# fabricates a different price for the same real order, since a platform can
# legitimately show fewer offers but showing a false payout would not be a
# throttle, it would be a lie about ground truth.
# ---------------------------------------------------------------------------
ACCEPTANCE_RATE_CALIBRATION: dict[str, float] = {
    "reference_rate": 0.70,  # acceptance rate at/above which no penalty applies
    "min_offer_count_multiplier": 0.35,  # floor multiplier on offer count at a 0% acceptance rate
}


def acceptance_rate_offer_multiplier(
    acceptance_rate: float,
    calibration: dict[str, float] = ACCEPTANCE_RATE_CALIBRATION,
) -> float:
    """The platform hook: how much a courier's low acceptance rate shrinks
    the number of offers shown, as a multiplier in
    [`min_offer_count_multiplier`, 1.0].

    Linear ramp: 1.0 at/above `reference_rate`, `min_offer_count_multiplier`
    at a 0% acceptance rate. Any `PlatformPort` implementation (this
    engine's stub included) can call this directly rather than reinventing
    the curve.
    """
    reference_rate = calibration["reference_rate"]
    floor = calibration["min_offer_count_multiplier"]
    if reference_rate <= 0.0 or acceptance_rate >= reference_rate:
        return 1.0
    fraction = max(acceptance_rate, 0.0) / reference_rate
    return floor + (1.0 - floor) * fraction


# ---------------------------------------------------------------------------
# OFFER FLOW: how offers reach the courier, and how many jobs they may hold.
#
# The single most consequential modelling choice in the engine, so it is
# spelled out rather than buried. The first version of this loop only called
# `PlatformPort.view_at` while the courier was IDLE. Measured, that courier
# was busy 98% of the time, so they saw whatever happened to spawn within
# reach during the one minute between finishing one delivery and starting the
# next — usually one card, often none. Selectivity was worthless by
# construction: a selective policy could only idle more and earn less, and the
# A/B comparison measured nothing.
#
# What the app actually does, per the brief: "La app muestra un FLUJO de
# pedidos, y el repartidor tiene SEGUNDOS para aceptar o rechazar." A FLOW,
# and SECONDS. Not a menu. Three consequences, all of them load-bearing:
#
#   1. OFFERS ARE ALWAYS FRESH, AND NEVER COME BACK. An order reaches the
#      courier during the minute it spawns and no other. Decline it, or be
#      too busy to look, and another courier took it — it is gone forever.
#      This engine therefore has NO offer store of any kind, deliberately:
#      an accumulated backlog would let a choosy policy fish in a pond that
#      has already been emptied, inflating every number in the comparison.
#      (`OfferCard.expires_in_seconds` is 20-90 s; against a one-minute tick
#      that resolves to exactly "this tick", which is why no expiry
#      bookkeeping is needed to make the field real.)
#
#   2. A BUSY COURIER STILL GETS OFFERED THEIR NEXT JOB. Real platforms
#      surface the next order while a courier is still finishing the current
#      one, and accepting queues it. So the courier is not blind while
#      riding — they are committing EARLIER and with LESS information, and
#      that is the whole strategic tradeoff: commit now and eliminate the
#      idle gap, or stay free and keep the option on something better while
#      risking an empty screen at the moment you go idle.
#
#   3. THE QUEUE IS SHALLOW AND STRICTLY SEQUENTIAL. `max_jobs_in_hand`
#      counts the job in progress PLUS anything accepted ahead. At 2 the
#      courier may hold one job and have one waiting. They are done one
#      after the other — finish A, then start B. This is QUEUEING, not
#      batching: the courier never carries two orders at once, and the
#      engine's phase machine has no representation for doing so.
#      Once the courier is full the app stops offering, exactly as a real
#      one does, so `offers_seen` stays the count of offers they could
#      actually have taken.
#
# The brief's anchor is ~7.5 offers per courier-hour at roughly a 1-in-3
# acceptance rate, giving ~2.5 deliveries/hour. A courier who keeps their
# queue permanently full will see FEWER than 7.5 — that is not a calibration
# error, it is the cost of never leaving a slot open.
# ---------------------------------------------------------------------------
OFFER_FLOW_CALIBRATION: dict[str, float] = {
    "max_jobs_in_hand": 2.0,
}

# ---------------------------------------------------------------------------
# StubPlatform (see stubs.py) — trivial offer-surfacing rules, not a claim
# about how the real platform layer should route orders.
# ---------------------------------------------------------------------------
STUB_PLATFORM_CALIBRATION: dict[str, float] = {
    "offer_radius_km": 4.0,  # only orders within this straight-line radius are ever shown
    "max_offers_per_tick": 3,  # before the acceptance-rate throttle is applied
    "expires_in_seconds": 60.0,  # one simulated minute's worth of decision pressure
}

# ---------------------------------------------------------------------------
# Travel resilience: a courier never teleports and never gives up, even when
# the current working graph (post-closure) has no known path for a pair.
#
# `TravelMatrix` (world layer) exposes no "reopen a street" primitive, so
# `NetworkTravelOracle.apply_closure` accumulates closures for the rest of
# the shift once applied (see that class's docstring). Left unchecked over
# an 8-hour shift this can eventually disconnect part of the operating-area
# graph and make a real cell pair unroutable through the CLOSED working
# graph. Two independent mitigations, both calibration, not measured data:
#
#   1. `max_simultaneous_closures` bounds how many closures this oracle will
#      ever apply at once — further STREET_CLOSURE events are skipped
#      (logged, not applied) once the cap is hit, rather than the graph
#      degrading toward permanent demolition over the shift. This is the
#      deliberately CHEAP mitigation (no rebuild): the alternative (rebuild
#      affected rows from a pristine baseline whenever a closure passes its
#      `end_min`) needs a full re-run of Dijkstra over the whole drive graph
#      per rebuild (~100+ seconds measured on the real Monterrey graph),
#      which is too slow to pay mid-shift; the cap is the explicitly
#      sanctioned fallback for that cost, see `travel.py`.
#   2. If a pair is STILL unroutable through the current working graph (the
#      cap above only bounds future closures, it does not undo one already
#      applied), `NetworkTravelOracle` falls back to the PRE-CLOSURE
#      ("pristine") baseline km/minutes for that exact pair, penalised by
#      these multipliers — never a bare crash. If even the pristine pair has
#      no route (a base-graph gap, not caused by any closure this run
#      applied), it falls back once more to a straight-line estimate at
#      `fallback_speed_kmh`. Every fallback is logged so it is auditable
#      when it fires.
# ---------------------------------------------------------------------------
TRAVEL_CALIBRATION: dict[str, float] = {
    "max_simultaneous_closures": 3,
    "unroutable_detour_km_multiplier": 1.3,  # a real detour is real extra distance
    "unroutable_detour_minutes_multiplier": 1.6,  # ...and usually slower per km too
    "fallback_speed_kmh": 20.0,  # last-resort straight-line speed assumption
}

# ---------------------------------------------------------------------------
# Home: no numeric knob is needed beyond picking a default when the caller of
# `run_shift` does not supply one — see `engine.py::_default_home_cell`. Kept
# here as a named constant so the choice is easy to find and swap out.
# ---------------------------------------------------------------------------
DEFAULT_HOME_SELECTION = "nearest cell to the operating area's own centroid"

# ---------------------------------------------------------------------------
# Plausibility self-check (see `engine.py::self_check`).
#
# These bounds are exactly the honesty gate stated in the brief: a real
# Monterrey courier's shift is reported to land in these ranges. If a run
# lands outside them, the engine (or its calibration) is broken and nothing
# downstream can be trusted — do not loosen these to make a run "pass".
# ---------------------------------------------------------------------------
PLAUSIBILITY_CALIBRATION: dict[str, float] = {
    "mxn_per_hour_min": 100.0,
    "mxn_per_hour_max": 150.0,
    "deliveries_per_hour_min": 2.0,
    "deliveries_per_hour_max": 3.0,
    "avg_trip_km_min": 2.0,
    "avg_trip_km_max": 6.0,
    "idle_fraction_min": 0.15,
    "idle_fraction_max": 0.40,
}
