"""The constraints that stop the policy from degenerating.

Without these a "reject until perfect" policy looks optimal on paper and sinks
a real courier: deactivated for a low acceptance rate, or stranded dry.
"""

from __future__ import annotations

from src.core.ports import Action
from tests.agent.factories import make_offer, scenario

from src.agent.smart import SmartPolicy

MINUTE_1900 = 19 * 60
MINUTE_2130 = 21 * 60 + 30

# Long, badly paid, still profitable: the kind of offer a healthy courier skips.
MARGINAL_OFFER = make_offer(
    "MARGINAL",
    pickup_cell="MTY-E",
    dropoff_cell="MTY-W",
    payout_mxn=42.0,
)


def test_a_healthy_acceptance_rate_lets_the_policy_skip_a_marginal_offer() -> None:
    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            offers=(MARGINAL_OFFER,),
            offers_seen=20,
            offers_accepted=14,
        )
    )

    assert decision.action is not Action.ACCEPT


def test_a_low_acceptance_rate_relaxes_the_threshold() -> None:
    policy = SmartPolicy()

    healthy = policy.decide(
        *scenario(minute=MINUTE_1900, offers=(MARGINAL_OFFER,), offers_seen=20, offers_accepted=14)
    )
    choosy = policy.decide(
        *scenario(minute=MINUTE_1900, offers=(MARGINAL_OFFER,), offers_seen=20, offers_accepted=5)
    )

    assert choosy.trace.threshold_mxn_per_hour < healthy.trace.threshold_mxn_per_hour


def test_a_starved_acceptance_rate_forces_the_best_profitable_offer() -> None:
    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            offers=(MARGINAL_OFFER,),
            offers_seen=20,
            offers_accepted=3,
        )
    )

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "MARGINAL"
    assert decision.trace.binding_constraint == "acceptance_rate"


def test_a_starved_acceptance_rate_still_refuses_a_loss_making_offer() -> None:
    loss_maker = make_offer(
        "LOSS", pickup_cell="MTY-FAR", dropoff_cell="MTY-S", payout_mxn=15.0
    )

    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            offers=(loss_maker,),
            offers_seen=20,
            offers_accepted=2,
        )
    )

    assert decision.action is not Action.ACCEPT


def test_a_nearly_empty_tank_is_planned_for_not_ignored() -> None:
    good_offer = make_offer("GOOD", pickup_cell="MTY-C", dropoff_cell="MTY-SC", payout_mxn=95.0)

    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            offers=(good_offer,),
            fuel_minutes_remaining=8.0,
            minutes_left_in_shift=180,
        )
    )

    assert decision.action is Action.REFUEL
    assert decision.trace.binding_constraint == "fuel"
    assert decision.trace.considered  # the offer was still evaluated, not skipped
    assert "fuel" in decision.trace.summary.lower()


def test_an_offer_longer_than_the_remaining_fuel_is_refused_and_the_stop_is_planned() -> None:
    long_offer = make_offer("LONG", pickup_cell="MTY-E", dropoff_cell="MTY-W", payout_mxn=160.0)

    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            offers=(long_offer,),
            fuel_minutes_remaining=40.0,
            minutes_left_in_shift=180,
        )
    )

    assert decision.action is Action.REFUEL
    assert decision.order_id is None
    assert "fuel" in (decision.trace.considered[0].rejected_because or "")


def test_refuelling_is_not_worth_it_in_the_last_minutes_of_the_shift() -> None:
    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_2130,
            offers=(),
            fuel_minutes_remaining=9.0,
            minutes_left_in_shift=10,
        )
    )

    assert decision.action is not Action.REFUEL


def test_it_repositions_towards_believed_demand_when_nothing_is_on_screen() -> None:
    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            at_cell="MTY-C",
            offers=(),
            demand={"MTY-C": 0.02, "MTY-SC": 0.95, "MTY-N": 0.1},
            minutes_left_in_shift=180,
        )
    )

    assert decision.action is Action.REPOSITION
    assert decision.target_cell == "MTY-SC"
    assert decision.trace.summary


def test_it_does_not_chase_a_marginal_gain_across_the_city() -> None:
    decision = SmartPolicy().decide(
        *scenario(
            minute=MINUTE_1900,
            at_cell="MTY-C",
            offers=(),
            demand={"MTY-C": 0.5, "MTY-FAR": 0.6},
            minutes_left_in_shift=180,
        )
    )

    assert decision.action is Action.HOLD
    assert decision.target_cell is None


def test_night_risk_is_priced_into_a_long_ride() -> None:
    policy = SmartPolicy()
    offer = make_offer("LONG_RIDE", pickup_cell="MTY-C", dropoff_cell="MTY-FAR", payout_mxn=200.0)

    day = policy.decide(*scenario(minute=14 * 60, offers=(offer,), minutes_left_in_shift=300))
    night = policy.decide(*scenario(minute=23 * 60, offers=(offer,), minutes_left_in_shift=300))

    day_eval = day.trace.considered[0]
    night_eval = night.trace.considered[0]
    assert night_eval.expected_net_mxn < day_eval.expected_net_mxn
    assert any("night" in f.label.lower() for f in night_eval.factors)
