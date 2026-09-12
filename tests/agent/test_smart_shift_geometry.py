"""End-of-shift geometry: where a delivery leaves you stops being free.

The unpaid ride home is real cost the platform never counts. These are the
cases that encode it.
"""

from __future__ import annotations

from src.core.ports import Action
from tests.agent.factories import make_offer, scenario

from src.agent.smart import SmartPolicy

MINUTE_1700 = 17 * 60
MINUTE_2130 = 21 * 60 + 30

# Courier starts in the home cell. This offer ends 7.8 km north of home.
AWAY_FROM_HOME = make_offer(
    "NORTHBOUND",
    pickup_cell="MTY-N",
    dropoff_cell="MTY-NN",
    payout_mxn=130.0,
)


def test_the_same_offer_is_accepted_at_1700_and_rejected_at_2130() -> None:
    policy = SmartPolicy()

    early = policy.decide(
        *scenario(
            minute=MINUTE_1700,
            at_cell="MTY-C",
            offers=(AWAY_FROM_HOME,),
            minutes_left_in_shift=240,
        )
    )
    late = policy.decide(
        *scenario(
            minute=MINUTE_2130,
            at_cell="MTY-C",
            offers=(AWAY_FROM_HOME,),
            minutes_left_in_shift=55,
        )
    )

    assert early.action is Action.ACCEPT
    assert early.order_id == "NORTHBOUND"

    assert late.action is not Action.ACCEPT
    assert late.order_id is None
    # The job itself would fit in 55 minutes. The ride home afterwards is what
    # does not, and that is the constraint the trace must name.
    assert late.trace.binding_constraint == "home"
    late_eval = late.trace.considered[0]
    assert "ride home" in (late_eval.rejected_because or "")
    job_minutes = sum(
        f.delta_minutes or 0.0
        for f in late_eval.factors
        if "ride home" not in f.label.lower() and "leaves me" not in f.label.lower()
    )
    assert job_minutes < 55  # the delivery itself fits; the ride home is what does not
    assert late_eval.expected_minutes > 55  # the offer's true cost in minutes does not


def test_late_in_the_shift_a_homeward_offer_beats_an_identical_outbound_one() -> None:
    # Both legs are 4.4 km from the same pickup for the same money. The only
    # difference is which way they point relative to home.
    view, sources, courier = scenario(
        minute=MINUTE_2130,
        at_cell="MTY-N",
        minutes_left_in_shift=90,
        offers=(
            make_offer("OUTBOUND", pickup_cell="MTY-N", dropoff_cell="MTY-NN", payout_mxn=90.0),
            make_offer("HOMEWARD", pickup_cell="MTY-N", dropoff_cell="MTY-SC", payout_mxn=90.0),
        ),
    )

    decision = SmartPolicy().decide(view, sources, courier)

    assert decision.action is Action.ACCEPT
    assert decision.order_id == "HOMEWARD"
    outbound = next(e for e in decision.trace.considered if e.order_id == "OUTBOUND")
    assert any(
        "home" in f.label.lower() and (f.delta_minutes or 0.0) > 0.0 for f in outbound.factors
    )


def test_early_in_the_shift_the_outbound_penalty_is_not_applied() -> None:
    # Same pair of offers at 17:00: the ride home is hours away, so it must not
    # move the score. Direction only starts to matter as the shift closes.
    view, sources, courier = scenario(
        minute=MINUTE_1700,
        at_cell="MTY-N",
        minutes_left_in_shift=300,
        offers=(
            make_offer("OUTBOUND", pickup_cell="MTY-N", dropoff_cell="MTY-NN", payout_mxn=90.0),
            make_offer("HOMEWARD", pickup_cell="MTY-N", dropoff_cell="MTY-SC", payout_mxn=90.0),
        ),
    )

    decision = SmartPolicy().decide(view, sources, courier)

    outbound = next(e for e in decision.trace.considered if e.order_id == "OUTBOUND")
    homeward = next(e for e in decision.trace.considered if e.order_id == "HOMEWARD")
    assert outbound.expected_minutes == homeward.expected_minutes


def test_the_acceptance_threshold_rises_early_and_falls_late() -> None:
    policy = SmartPolicy()
    offer = make_offer("A", pickup_cell="MTY-C", dropoff_cell="MTY-SC", payout_mxn=70.0)

    early = policy.decide(
        *scenario(minute=MINUTE_1700, offers=(offer,), minutes_left_in_shift=300)
    )
    late = policy.decide(
        *scenario(minute=MINUTE_2130, offers=(offer,), minutes_left_in_shift=45)
    )

    assert early.trace.threshold_mxn_per_hour > late.trace.threshold_mxn_per_hour
