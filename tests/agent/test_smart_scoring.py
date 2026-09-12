"""What the smart policy is supposed to prefer, offer against offer.

Every case here is a comparison between two offers that differ in exactly one
thing, so a failure names the reasoning factor that broke.
"""

from __future__ import annotations

from src.core.ports import Action
from tests.agent.factories import make_offer, scenario, sure

from src.agent.smart import SmartPolicy

MINUTE_1700 = 17 * 60  # mid-shift: no end-of-shift pressure in these cases


def evaluation_for(trace, order_id: str):
    return next(e for e in trace.considered if e.order_id == order_id)


def test_short_well_paid_offer_beats_long_badly_paid_one() -> None:
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-C",
        offers=(
            make_offer("SHORT_RICH", pickup_cell="MTY-C", dropoff_cell="MTY-SC", payout_mxn=85.0),
            make_offer("LONG_POOR", pickup_cell="MTY-N", dropoff_cell="MTY-FAR", payout_mxn=45.0),
        ),
    )

    decision = SmartPolicy().decide(view, observation, courier)

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "SHORT_RICH"
    short = evaluation_for(decision.trace, "SHORT_RICH")
    long_ = evaluation_for(decision.trace, "LONG_POOR")
    assert short.expected_mxn_per_hour > long_.expected_mxn_per_hour


def test_far_pickup_loses_to_a_near_one_at_equal_pay() -> None:
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-C",
        offers=(
            make_offer("NEAR_PICKUP", pickup_cell="MTY-C", dropoff_cell="MTY-E", payout_mxn=80.0),
            make_offer("FAR_PICKUP", pickup_cell="MTY-NN", dropoff_cell="MTY-E", payout_mxn=80.0),
        ),
    )

    decision = SmartPolicy().decide(view, observation, courier)

    assert decision.order_id == "NEAR_PICKUP"
    near = evaluation_for(decision.trace, "NEAR_PICKUP")
    far = evaluation_for(decision.trace, "FAR_PICKUP")
    assert near.expected_minutes < far.expected_minutes
    assert near.expected_km < far.expected_km


def test_offer_into_a_dead_zone_loses_to_an_equally_paid_one_into_a_busy_zone() -> None:
    # Identical pay and identical leg lengths: MTY-N -> MTY-NN and MTY-N -> MTY-SC
    # are both 4.4 km. Only the demand at the destination differs.
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-N",
        demand={"MTY-N": 0.5, "MTY-NN": 0.9, "MTY-SC": 0.03, "MTY-C": 0.5},
        offers=(
            make_offer("INTO_BUSY", pickup_cell="MTY-N", dropoff_cell="MTY-NN", payout_mxn=80.0),
            make_offer("INTO_DEAD", pickup_cell="MTY-N", dropoff_cell="MTY-SC", payout_mxn=80.0),
        ),
    )

    decision = SmartPolicy().decide(view, observation, courier)

    assert decision.order_id == "INTO_BUSY"
    busy = evaluation_for(decision.trace, "INTO_BUSY")
    dead = evaluation_for(decision.trace, "INTO_DEAD")
    assert dead.expected_minutes > busy.expected_minutes
    assert any("demand" in f.label.lower() for f in dead.factors)


def test_high_score_on_a_low_confidence_belief_is_discounted_below_a_solid_one() -> None:
    # Same geometry, same destination. GUESS pays more but its kitchen estimate
    # is barely believed; KNOWN pays less and is backed by a solid memory.
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-N",
        kitchen={
            "Cocina Incierta": sure(6.0, confidence=0.05),
            "Cocina Conocida": sure(6.0, confidence=1.0),
        },
        offers=(
            make_offer(
                "GUESS",
                pickup_cell="MTY-N",
                dropoff_cell="MTY-SC",
                payout_mxn=95.0,
                restaurant_name="Cocina Incierta",
            ),
            make_offer(
                "KNOWN",
                pickup_cell="MTY-N",
                dropoff_cell="MTY-SC",
                payout_mxn=88.0,
                restaurant_name="Cocina Conocida",
            ),
        ),
    )

    decision = SmartPolicy().decide(view, observation, courier)

    guess = evaluation_for(decision.trace, "GUESS")
    known = evaluation_for(decision.trace, "KNOWN")
    assert guess.expected_net_mxn > known.expected_net_mxn  # raw money says GUESS
    assert decision.order_id == "KNOWN"  # confidence says otherwise
    assert any("confidence" in f.label.lower() for f in guess.factors)


def test_it_uses_its_own_travel_model_not_the_app_eta() -> None:
    offer = make_offer(
        "OPTIMISTIC",
        pickup_cell="MTY-C",
        dropoff_cell="MTY-FAR",
        payout_mxn=120.0,
        eta_minutes=3.0,  # the app is lying to the courier
        distance_km=1.0,
    )
    view, observation, courier = scenario(minute=MINUTE_1700, at_cell="MTY-C", offers=(offer,))

    decision = SmartPolicy().decide(view, observation, courier)

    evaluation = evaluation_for(decision.trace, "OPTIMISTIC")
    assert evaluation.expected_minutes > 4 * offer.eta_minutes
    assert evaluation.expected_km > 4 * offer.distance_km


def test_believed_traffic_slows_an_offer_down() -> None:
    clear_view, clear_obs, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-C",
        traffic={"MTY-C": 1.0, "MTY-E": 1.0},
        offers=(make_offer("A", pickup_cell="MTY-C", dropoff_cell="MTY-E", payout_mxn=80.0),),
    )
    jam_view, jam_obs, _ = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-C",
        traffic={"MTY-C": 1.0, "MTY-E": 2.2},
        offers=(make_offer("A", pickup_cell="MTY-C", dropoff_cell="MTY-E", payout_mxn=80.0),),
    )
    policy = SmartPolicy()

    clear = evaluation_for(policy.decide(clear_view, clear_obs, courier).trace, "A")
    jammed = evaluation_for(policy.decide(jam_view, jam_obs, courier).trace, "A")

    assert jammed.expected_minutes > clear.expected_minutes
    assert jammed.expected_km == clear.expected_km  # km and minutes are separate
