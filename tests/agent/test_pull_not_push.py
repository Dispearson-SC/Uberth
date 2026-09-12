"""The inversion itself, asserted rather than assumed.

It is possible to migrate a signature and change nothing real: hand the
policy a port, have it ignore it, and keep every belief hardcoded. These
tests fail if that ever happens. The fake port records every question it is
asked, so "the agent went and found out" is checkable rather than a claim in
a docstring.
"""

from __future__ import annotations

from src.core.ports import Action, Estimate
from tests.agent.factories import CELLS, make_event, make_offer, scenario

from src.agent.smart import SmartPolicy

MINUTE_1900 = 19 * 60

OFFER = make_offer("A", pickup_cell="MTY-N", dropoff_cell="MTY-E", payout_mxn=90.0)


def test_the_policy_asks_every_source_it_reasons_from() -> None:
    view, sources, courier = scenario(minute=MINUTE_1900, at_cell="MTY-C", offers=(OFFER,))

    SmartPolicy().decide(view, sources, courier)

    for method in (
        "weather_at",
        "congestion_near",
        "poi_density_near",
        "perceived_disruptions",
        "cell_coords",
        "travel_estimate",
        "recall_kitchen",
        "recall_eta_bias",
    ):
        assert sources.asked(method), (
            "the policy never asked %r. Either it reasons from something it was "
            "not given, or that belief is hardcoded — and a hardcoded belief does "
            "not survive being dropped in another city." % method
        )


def test_it_asks_about_the_coordinates_it_was_actually_given() -> None:
    """A courier asks about where they ARE, not about a place in a constant."""
    view, sources, courier = scenario(minute=MINUTE_1900, at_cell="MTY-N", offers=(OFFER,))

    SmartPolicy().decide(view, sources, courier)

    here = (round(courier.lat, 5), round(courier.lon, 5))
    assert any(q[1:3] == here for q in sources.asked("weather_at"))
    assert any(q[1:3] == here for q in sources.asked("congestion_near"))
    # And it priced the ride to this offer's actual pickup point.
    pickup = (round(OFFER.pickup_lat, 5), round(OFFER.pickup_lon, 5))
    assert any(q[3:5] == pickup for q in sources.asked("travel_estimate"))


def test_a_source_that_says_nothing_is_survivable() -> None:
    """Every query answering empty is exactly a first minute in a new city.

    The agent must still decide — on priors it knows it barely believes —
    rather than crash or refuse. This is the cold start, stated as a test.
    """
    view, sources, courier = scenario(
        minute=MINUTE_1900,
        at_cell="MTY-C",
        offers=(OFFER,),
        poi_density={},
        traffic={},
        cell_coords={},
    )

    decision = SmartPolicy().decide(view, sources, courier)

    assert decision.trace.summary.strip()
    assert decision.trace.considered
    assert decision.action in tuple(Action)


def test_a_learned_travel_correction_is_used_and_shown() -> None:
    """The second layer of the travel model, and it has to be visible.

    A port reporting a well-evidenced correction (high confidence) is
    telling the agent its own trips have this zone covered. The agent then
    leans on that instead of multiplying its live congestion reading on top,
    because the correction already contains the typical jam.
    """
    view, plain, courier = scenario(
        minute=MINUTE_1900, at_cell="MTY-C", offers=(OFFER,), traffic={c: 1.8 for c in CELLS}
    )
    view2, learned, courier2 = scenario(
        minute=MINUTE_1900,
        at_cell="MTY-C",
        offers=(OFFER,),
        traffic={c: 1.8 for c in CELLS},
        travel_correction=Estimate(value=1.0, confidence=0.92, age_minutes=0.0),
    )

    without = SmartPolicy().decide(view, plain, courier).trace.considered[0]
    with_fit = SmartPolicy().decide(view2, learned, courier2).trace.considered[0]

    # Believed jam of 1.8x, and a fully-evidenced correction of 1.0x. The
    # agent that has measured the zone believes the trip is faster than the
    # one reading a live multiplier it cannot check.
    assert with_fit.expected_minutes < without.expected_minutes
    assert learned.asked("travel_estimate")


def test_a_learned_app_eta_bias_moves_the_customer_leg() -> None:
    """The purest self-learned arbitrage: the app says eleven, it took nineteen."""
    optimistic = make_offer(
        "OPT", pickup_cell="MTY-C", dropoff_cell="MTY-E", payout_mxn=90.0, eta_minutes=30.0
    )
    view, plain, courier = scenario(minute=MINUTE_1900, at_cell="MTY-C", offers=(optimistic,))
    view2, biased, courier2 = scenario(
        minute=MINUTE_1900,
        at_cell="MTY-C",
        offers=(optimistic,),
        eta_bias={"MTY-E": Estimate(value=1.6, confidence=0.9, age_minutes=0.0)},
    )

    without = SmartPolicy().decide(view, plain, courier).trace.considered[0]
    with_bias = SmartPolicy().decide(view2, biased, courier2).trace.considered[0]

    assert biased.asked("recall_eta_bias")
    assert with_bias.expected_minutes > without.expected_minutes
    assert any("under-promised" in f.note for f in with_bias.factors)


def test_only_disclosed_disruptions_can_reach_a_score() -> None:
    """Detectability, from the agent's side of the port.

    The source discloses one crash and withholds another. The withheld one
    exists; the agent has no path to it, so it cannot appear in a score or
    in a trace.
    """
    disclosed = make_event("EV-SEEN", kind="crash", cells=("MTY-E",), delay_minutes=15.0)
    withheld = make_event("EV-HIDDEN", kind="crash", cells=("MTY-E",), delay_minutes=40.0)

    view, sources, courier = scenario(
        minute=MINUTE_1900,
        at_cell="MTY-C",
        offers=(make_offer("A", pickup_cell="MTY-E", dropoff_cell="MTY-SC", payout_mxn=120.0),),
        events=(disclosed,),
    )

    trace = SmartPolicy().decide(view, sources, courier).trace
    text = " ".join(f.label + f.note for e in trace.considered for f in e.factors)

    assert disclosed.event_id in text
    assert withheld.event_id not in text
