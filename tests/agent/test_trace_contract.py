"""The trace is what a judge inspects, so it is held to a contract.

The hard one is honesty: the trace may contain only what the agent could know
at that minute. If it names an event that is not in `perceived_events`, the
agent saw the future, and the whole demo is a lie. That is asserted as a
property over many generated decisions, not as a single lucky case.
"""

from __future__ import annotations

import random

import pytest

from src.core.ports import Action, Policy
from tests.agent.factories import CELLS, make_event, make_offer, scenario, trace_text

from src.agent.baseline import AcceptAllPolicy, NearestFirstPolicy
from src.agent.smart import SmartPolicy

CELL_NAMES = list(CELLS)

# Every event that could exist in this world. Each decision perceives a random
# subset; naming any of the rest would be clairvoyance.
EVENT_UNIVERSE = tuple(
    make_event(
        "EV-%02d-%s" % (i, kind),
        kind=kind,
        cells=(CELL_NAMES[i % len(CELL_NAMES)],),
        delay_minutes=4.0 + i,
        confidence=0.4 + 0.05 * (i % 10),
    )
    for i, kind in enumerate(
        ["crash", "closure", "protest", "flood", "concert", "crash", "closure", "blackout"]
    )
)

POLICIES: list[type] = [AcceptAllPolicy, NearestFirstPolicy, SmartPolicy]


def random_scenario(rng: random.Random):
    perceived = tuple(e for e in EVENT_UNIVERSE if rng.random() < 0.4)
    offers = tuple(
        make_offer(
            "ORD-%d" % i,
            pickup_cell=rng.choice(CELL_NAMES),
            dropoff_cell=rng.choice(CELL_NAMES),
            payout_mxn=round(rng.uniform(25.0, 220.0), 1),
            surge_flag=rng.random() < 0.3,
            expires_in_seconds=rng.randint(10, 90),
        )
        for i in range(rng.randint(0, 4))
    )
    return scenario(
        minute=rng.randint(10 * 60, 23 * 60 + 59),
        at_cell=rng.choice(CELL_NAMES),
        offers=offers,
        demand={c: round(rng.uniform(0.0, 1.0), 2) for c in CELL_NAMES},
        traffic={c: round(rng.uniform(0.8, 2.5), 2) for c in CELL_NAMES},
        events=perceived,
        minutes_left_in_shift=rng.randint(5, 420),
        fuel_minutes_remaining=round(rng.uniform(2.0, 200.0), 1),
        offers_seen=rng.randint(1, 60),
        offers_accepted=rng.randint(0, 30),
        precip_mm=round(rng.choice([0.0, 0.0, 0.4, 3.0]), 1),
    )


@pytest.mark.parametrize("policy_class", POLICIES)
def test_the_trace_never_names_an_unperceived_event(policy_class: type) -> None:
    rng = random.Random(20260912)
    policy: Policy = policy_class()

    for _ in range(250):
        view, observation, courier = random_scenario(rng)
        if courier.offers_accepted > courier.offers_seen:
            courier.offers_accepted = courier.offers_seen

        text = trace_text(policy.decide(view, observation, courier).trace)

        perceived_ids = {e.event_id for e in observation.perceived_events}
        for event in EVENT_UNIVERSE:
            if event.event_id in perceived_ids:
                continue
            assert event.event_id not in text, (
                "trace named event %s which the courier could not perceive" % event.event_id
            )


@pytest.mark.parametrize("policy_class", POLICIES)
def test_every_decision_carries_a_usable_trace(policy_class: type) -> None:
    rng = random.Random(7)
    policy: Policy = policy_class()

    for _ in range(250):
        view, observation, courier = random_scenario(rng)
        if courier.offers_accepted > courier.offers_seen:
            courier.offers_accepted = courier.offers_seen

        decision = policy.decide(view, observation, courier)
        trace = decision.trace

        assert trace.minute == view.minute
        assert trace.summary.strip()
        assert trace.summary.strip().endswith(".")
        assert trace.binding_constraint in {
            "time_budget",
            "acceptance_rate",
            "fuel",
            "home",
            "none",
        }
        assert {e.order_id for e in trace.considered} == {o.order_id for o in view.offers}
        assert all(e.factors for e in trace.considered)
        assert all(
            (f.delta_mxn is not None or f.delta_minutes is not None or f.note)
            for e in trace.considered
            for f in e.factors
        )
        if decision.action is Action.ACCEPT:
            assert decision.order_id is not None
            assert trace.chosen_order_id == decision.order_id
        else:
            assert decision.order_id is None
            assert trace.chosen_order_id is None
        if decision.action is Action.REPOSITION:
            assert decision.target_cell in observation.demand_by_cell


@pytest.mark.parametrize("policy_class", POLICIES)
def test_a_chosen_offer_is_always_one_that_was_on_screen(policy_class: type) -> None:
    rng = random.Random(99)
    policy: Policy = policy_class()

    for _ in range(100):
        view, observation, courier = random_scenario(rng)
        decision = policy.decide(view, observation, courier)
        if decision.order_id is not None:
            assert decision.order_id in {o.order_id for o in view.offers}


def test_the_trace_reports_a_perceived_event_it_actually_used() -> None:
    event = make_event("EV-99-crash", kind="crash", cells=("MTY-E",), delay_minutes=15.0)
    view, observation, courier = scenario(
        minute=19 * 60,
        at_cell="MTY-C",
        offers=(make_offer("A", pickup_cell="MTY-E", dropoff_cell="MTY-SC", payout_mxn=90.0),),
        events=(event,),
    )

    decision = SmartPolicy().decide(view, observation, courier)

    assert "EV-99-crash" in trace_text(decision.trace)
