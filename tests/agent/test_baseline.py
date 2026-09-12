"""The policies to beat. A baseline that cannot explain itself is not a baseline."""

from __future__ import annotations

import pytest

from src.core.ports import Action, Decision, DecisionTrace
from tests.agent.factories import make_offer, scenario

from src.agent.baseline import AcceptAllPolicy, NearestFirstPolicy

MINUTE_1700 = 17 * 60


@pytest.fixture(params=[AcceptAllPolicy, NearestFirstPolicy])
def baseline(request) -> object:
    return request.param()


def test_baseline_has_a_name(baseline) -> None:
    assert isinstance(baseline.name, str) and baseline.name


def test_accept_all_takes_even_a_terrible_offer() -> None:
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        offers=(
            make_offer("BAD", pickup_cell="MTY-FAR", dropoff_cell="MTY-S", payout_mxn=18.0),
        ),
    )

    decision = AcceptAllPolicy().decide(view, observation, courier)

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "BAD"


def test_nearest_first_takes_the_closest_pickup_not_the_best_pay() -> None:
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-C",
        offers=(
            make_offer("FAR_RICH", pickup_cell="MTY-FAR", dropoff_cell="MTY-N", payout_mxn=200.0),
            make_offer("NEAR_POOR", pickup_cell="MTY-C", dropoff_cell="MTY-N", payout_mxn=40.0),
        ),
    )

    decision = NearestFirstPolicy().decide(view, observation, courier)

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "NEAR_POOR"


def test_baselines_hold_when_there_is_nothing_on_screen(baseline) -> None:
    view, observation, courier = scenario(minute=MINUTE_1700, offers=())

    decision = baseline.decide(view, observation, courier)

    assert decision.action is Action.HOLD
    assert decision.order_id is None
    assert decision.trace.summary


def test_baselines_return_a_full_trace(baseline) -> None:
    view, observation, courier = scenario(
        minute=MINUTE_1700,
        offers=(
            make_offer("A", pickup_cell="MTY-N", dropoff_cell="MTY-SC", payout_mxn=70.0),
            make_offer("B", pickup_cell="MTY-E", dropoff_cell="MTY-W", payout_mxn=95.0),
        ),
    )

    decision = baseline.decide(view, observation, courier)

    assert isinstance(decision, Decision)
    assert isinstance(decision.trace, DecisionTrace)
    assert decision.trace.minute == MINUTE_1700
    assert {e.order_id for e in decision.trace.considered} == {"A", "B"}
    assert all(e.factors for e in decision.trace.considered)
    assert decision.trace.summary.strip()
    assert decision.trace.chosen_order_id == decision.order_id
