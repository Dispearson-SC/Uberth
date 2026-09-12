"""The policies to beat. A baseline that cannot explain itself is not a baseline."""

from __future__ import annotations

import pytest

from src.core.ports import Action, Decision, DecisionTrace
from tests.agent.factories import make_offer, scenario

from src.agent.baseline import AcceptAllPolicy, FixedPayoutThresholdPolicy
from src.agent.calibration import BASELINE_CALIBRATION

MINUTE_1700 = 17 * 60
FLOOR = BASELINE_CALIBRATION["fixed_payout_floor_mxn"]


@pytest.fixture(params=[AcceptAllPolicy, FixedPayoutThresholdPolicy])
def baseline(request) -> object:
    return request.param()


def test_baseline_has_a_name(baseline) -> None:
    assert isinstance(baseline.name, str) and baseline.name


def test_accept_all_takes_even_a_terrible_offer() -> None:
    view, sources, courier = scenario(
        minute=MINUTE_1700,
        offers=(
            make_offer("BAD", pickup_cell="MTY-FAR", dropoff_cell="MTY-S", payout_mxn=18.0),
        ),
    )

    decision = AcceptAllPolicy().decide(view, sources, courier)

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "BAD"


def test_fixed_threshold_rejects_the_one_card_below_its_floor() -> None:
    """The flow case, and the one that matters: one offer, under the floor."""
    view, sources, courier = scenario(
        minute=MINUTE_1700,
        offers=(
            make_offer("CHEAP", pickup_cell="MTY-C", dropoff_cell="MTY-N",
                       payout_mxn=FLOOR - 10.0),
        ),
    )

    decision = FixedPayoutThresholdPolicy().decide(view, sources, courier)

    assert decision.action is Action.REJECT
    assert decision.order_id is None
    assert "%.0f" % FLOOR in decision.trace.summary


def test_fixed_threshold_takes_the_one_card_that_clears_its_floor() -> None:
    view, sources, courier = scenario(
        minute=MINUTE_1700,
        offers=(
            make_offer("OK", pickup_cell="MTY-FAR", dropoff_cell="MTY-S",
                       payout_mxn=FLOOR + 5.0),
        ),
    )

    decision = FixedPayoutThresholdPolicy().decide(view, sources, courier)

    # Clears the floor even though the pickup is 12 km away and the drop-off
    # is on the far side of the city: the floor is the only thing it reads.
    assert decision.action is Action.ACCEPT
    assert decision.order_id == "OK"


def test_fixed_threshold_is_not_nearest_first_it_takes_the_best_payer() -> None:
    """The distinction that killed `NearestFirstPolicy`: this rule is about
    the money on the card, not the geometry, so a far rich offer beats a near
    poor one."""
    view, sources, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-C",
        offers=(
            make_offer("FAR_RICH", pickup_cell="MTY-FAR", dropoff_cell="MTY-N",
                       payout_mxn=200.0),
            make_offer("NEAR_POOR", pickup_cell="MTY-C", dropoff_cell="MTY-N",
                       payout_mxn=FLOOR + 1.0),
        ),
    )

    decision = FixedPayoutThresholdPolicy().decide(view, sources, courier)

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "FAR_RICH"


def test_fixed_threshold_relaxes_its_floor_near_the_bell() -> None:
    """A real courier's discipline slackens in the last hour: an order that
    pays something beats riding home empty."""
    payout = FLOOR * BASELINE_CALIBRATION["late_shift_floor_factor"] + 1.0
    offers = (
        make_offer("LATE", pickup_cell="MTY-C", dropoff_cell="MTY-SC", payout_mxn=payout),
    )

    mid_shift, sources, courier = scenario(
        minute=MINUTE_1700, offers=offers, minutes_left_in_shift=240
    )
    assert FixedPayoutThresholdPolicy().decide(mid_shift, sources, courier).action is (
        Action.REJECT
    )

    late_view, late_sources, late_courier = scenario(
        minute=MINUTE_1700,
        offers=offers,
        minutes_left_in_shift=int(BASELINE_CALIBRATION["late_shift_minutes"]) - 5,
    )
    late = FixedPayoutThresholdPolicy().decide(late_view, late_sources, late_courier)
    assert late.action is Action.ACCEPT
    assert late.order_id == "LATE"


def test_fixed_threshold_suspends_its_floor_once_the_app_starves_it() -> None:
    """A baseline that destroys itself is a straw man, not an opponent.

    Measured without this: on the Night window the baseline rejected its
    opening offers, the platform cut its offer reach in reply, and it
    finished an eight-hour shift with zero deliveries and 0.0 MXN/h.
    """
    offers = (
        make_offer("CHEAP", pickup_cell="MTY-C", dropoff_cell="MTY-N", payout_mxn=FLOOR - 25.0),
    )

    healthy_view, sources, healthy = scenario(
        minute=MINUTE_1700, offers=offers, offers_seen=20, offers_accepted=14
    )
    assert FixedPayoutThresholdPolicy().decide(
        healthy_view, sources, healthy
    ).action is Action.REJECT

    starved_view, starved_sources, starved = scenario(
        minute=MINUTE_1700, offers=offers, offers_seen=20, offers_accepted=2
    )
    rescued = FixedPayoutThresholdPolicy().decide(
        starved_view, starved_sources, starved
    )
    assert rescued.action is Action.ACCEPT
    assert rescued.order_id == "CHEAP"
    assert rescued.trace.binding_constraint == "acceptance_rate"


def test_fixed_threshold_ignores_an_early_acceptance_rate_as_noise() -> None:
    offers = (
        make_offer("CHEAP", pickup_cell="MTY-C", dropoff_cell="MTY-N", payout_mxn=FLOOR - 25.0),
    )
    view, sources, courier = scenario(
        minute=MINUTE_1700, offers=offers, offers_seen=2, offers_accepted=0
    )

    assert FixedPayoutThresholdPolicy().decide(
        view, sources, courier
    ).action is Action.REJECT


def test_fixed_threshold_floor_is_overridable_for_the_calibration_sweep() -> None:
    offers = (make_offer("A", pickup_cell="MTY-C", dropoff_cell="MTY-N", payout_mxn=60.0),)
    view, sources, courier = scenario(minute=MINUTE_1700, offers=offers)

    assert FixedPayoutThresholdPolicy(floor_mxn=40.0).decide(
        view, sources, courier
    ).action is Action.ACCEPT
    assert FixedPayoutThresholdPolicy(floor_mxn=90.0).decide(
        view, sources, courier
    ).action is Action.REJECT


def test_baselines_hold_when_there_is_nothing_on_screen(baseline) -> None:
    view, sources, courier = scenario(minute=MINUTE_1700, offers=())

    decision = baseline.decide(view, sources, courier)

    assert decision.action is Action.HOLD
    assert decision.order_id is None
    assert decision.trace.summary


def test_baselines_return_a_full_trace(baseline) -> None:
    view, sources, courier = scenario(
        minute=MINUTE_1700,
        offers=(
            make_offer("A", pickup_cell="MTY-N", dropoff_cell="MTY-SC", payout_mxn=70.0),
            make_offer("B", pickup_cell="MTY-E", dropoff_cell="MTY-W", payout_mxn=95.0),
        ),
    )

    decision = baseline.decide(view, sources, courier)

    assert isinstance(decision, Decision)
    assert isinstance(decision.trace, DecisionTrace)
    assert decision.trace.minute == MINUTE_1700
    assert {e.order_id for e in decision.trace.considered} == {"A", "B"}
    assert all(e.factors for e in decision.trace.considered)
    assert decision.trace.summary.strip()
    assert decision.trace.chosen_order_id == decision.order_id
