"""The constraints that stop the policy from degenerating.

Without these a "reject until perfect" policy looks optimal on paper and sinks
a real courier: deactivated for a low acceptance rate, or stranded dry.
"""

from __future__ import annotations

import pytest

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


def _stand_in_a_dead_cell(policy: SmartPolicy, minutes: int):
    """Stand still with an empty screen for `minutes`, returning each decision.

    Standing still is the precondition for moving: the policy will not ride
    off a spot it has no evidence about, which is what stops the
    repositioning branch turning into a shift-long loop.

    The demand figures below are BELIEVED demands, inverted through the
    agent's own hour-of-day rhythm by `factories.density_for_demand`. At
    19:00 that rhythm sits in its afternoon trough, so the agent cannot
    believe any cell is more than about half busy however much commerce is
    in it — which means the believed EDGE between a dead cell and a busy one
    is smaller than the raw numbers suggest, and the direct evidence of an
    empty screen has to run longer before a move pays for itself. That
    ceiling is a property of composing demand from density and a rhythm, not
    a quirk of these fixtures: the sensed demand this replaced had the same
    shape.
    """
    decisions = []
    for offset in range(minutes):
        decisions.append(
            policy.decide(
                *scenario(
                    minute=MINUTE_1900 + offset,
                    at_cell="MTY-C",
                    offers=(),
                    demand={"MTY-C": 0.02, "MTY-SC": 0.95, "MTY-N": 0.1},
                    minutes_left_in_shift=180 - offset,
                )
            )
        )
    return decisions


def test_it_will_not_ride_off_a_spot_it_has_no_evidence_about_yet() -> None:
    first = _stand_in_a_dead_cell(SmartPolicy(), 1)[0]

    assert first.action is Action.HOLD
    assert first.target_cell is None


def test_it_repositions_towards_believed_demand_after_standing_in_a_dead_cell() -> None:
    decisions = _stand_in_a_dead_cell(SmartPolicy(), 20)

    moves = [d for d in decisions if d.action is Action.REPOSITION]
    assert moves, "stood twenty minutes in a dead cell and never moved"
    assert moves[0].target_cell == "MTY-SC"
    assert moves[0].trace.summary


def test_it_stops_asking_for_a_move_that_demonstrably_never_happens() -> None:
    """A courier knows whether they actually got anywhere.

    "Ride to that district" can name a place the courier then cannot set off
    for, in which case nothing happens: same cell, same empty screen, same
    decision next minute, for the rest of the shift. Measured before this
    guard: 412 reposition decisions in a 540-minute shift, of which the
    courier acted on none, 89% of it idle, two offers seen all morning.

    Here the courier never moves no matter what it asks for, which is
    exactly the situation the guard is for.
    """
    policy = SmartPolicy()
    moves = []
    for offset in range(60):
        decision = policy.decide(
            *scenario(
                minute=MINUTE_1900 + offset,
                at_cell="MTY-C",  # never actually goes anywhere
                offers=(),
                demand={"MTY-C": 0.02, "MTY-SC": 0.95, "MTY-N": 0.9},
                minutes_left_in_shift=300 - offset,
            )
        )
        if decision.action is Action.REPOSITION:
            moves.append(decision.target_cell)

    # It may ask once per plausible destination, then it has to stop: it has
    # watched each instruction fail.
    assert len(moves) <= 3, "asked to move %d times without ever moving: %s" % (
        len(moves),
        moves,
    )
    assert len(set(moves)) == len(moves), "asked twice for the same unreachable cell"


def test_it_does_not_ride_in_circles_when_the_whole_city_reads_dead() -> None:
    """The regression guard for a measured death spiral.

    At 06:00 the sensed demand map reads "almost nothing" everywhere and is
    re-drawn with fresh noise every minute, so whichever cell happened to
    round up became the target — and a starved courier's measured wait is at
    its clamp, which inflates every wait estimate enough to pay for a six
    kilometre ride. Measured before the guards: 393 repositions in a
    540-minute shift, 88% of it idle, two offers seen all morning, 7.6 MXN/h
    where the same policy with the branch muted earned 84.7.

    Here the courier is walked across cells exactly as a repositioning
    courier would be, with demand flickering by one heatmap band. A policy
    that moves on that is thrashing.
    """
    policy = SmartPolicy()
    cells = ["MTY-C", "MTY-SC", "MTY-N", "MTY-E", "MTY-W"]
    moves = 0
    for offset in range(60):
        here = cells[offset // 12 % len(cells)]
        # Every cell reads "nothing", one flickers up a single band.
        flicker = cells[(offset * 7) % len(cells)]
        demand = {cell: 0.0 for cell in cells}
        demand[flicker] = 0.25
        decision = policy.decide(
            *scenario(
                minute=MINUTE_1900 + offset,
                at_cell=here,
                offers=(),
                demand=demand,
                minutes_left_in_shift=300 - offset,
            )
        )
        if decision.action is Action.REPOSITION:
            moves += 1

    assert moves == 0, "moved %d times chasing one band of heatmap noise" % moves


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


def test_night_risk_does_not_switch_itself_off_at_midnight() -> None:
    """The regression guard for an off-by-1440.

    The premium used to be a single ramp over minute-of-day, so at 00:00 the
    clock reset below the ramp's start and 01:00 was priced as broad
    daylight. On the Night window (18:00-02:00) that exempted the last 120
    of 480 minutes — a quarter of the shift, and the darkest quarter.

    Absolute minute 1500 is 01:00 the next day: the window's minutes are a
    continuous counter, so this is exactly what the policy is handed there.
    """
    policy = SmartPolicy()
    offer = make_offer("LONG_RIDE", pickup_cell="MTY-C", dropoff_cell="MTY-FAR", payout_mxn=200.0)

    deep_night = policy.decide(
        *scenario(minute=23 * 60, offers=(offer,), minutes_left_in_shift=300)
    )
    after_midnight = policy.decide(
        *scenario(minute=25 * 60, offers=(offer,), minutes_left_in_shift=300)
    )

    after_eval = after_midnight.trace.considered[0]
    assert any("night" in f.label.lower() for f in after_eval.factors), (
        "01:00 was priced with no night premium at all"
    )
    assert after_eval.expected_net_mxn == pytest.approx(
        deep_night.trace.considered[0].expected_net_mxn
    )


def test_the_night_premium_fades_out_after_dawn_rather_than_at_midnight() -> None:
    policy = SmartPolicy()
    offer = make_offer("LONG_RIDE", pickup_cell="MTY-C", dropoff_cell="MTY-FAR", payout_mxn=200.0)

    before_dawn = policy.decide(
        *scenario(minute=4 * 60, offers=(offer,), minutes_left_in_shift=300)
    )
    mid_morning = policy.decide(
        *scenario(minute=9 * 60, offers=(offer,), minutes_left_in_shift=300)
    )

    assert any("night" in f.label.lower() for f in before_dawn.trace.considered[0].factors)
    assert not any("night" in f.label.lower() for f in mid_morning.trace.considered[0].factors)
    assert (
        before_dawn.trace.considered[0].expected_net_mxn
        < mid_morning.trace.considered[0].expected_net_mxn
    )
