"""The policies to beat.

Two of them, because beating one weak baseline proves nothing. `AcceptAllPolicy`
is what an inexperienced courier does on day one. `NearestFirstPolicy` is the
heuristic most people would write if asked for one in a minute, and it is the
real bar: it is not stupid, it is just short-sighted.

Both return a genuine `DecisionTrace`. A comparison where only one side can
explain itself is not a comparison, it is an advertisement.

Both also do something the smart policy refuses to do: they believe the app.
They score with `eta_minutes` and `distance_km` straight off the offer card,
which is exactly the naivety being measured.
"""

from __future__ import annotations

from src.core.ports import (
    Action,
    CourierSnapshot,
    Decision,
    DecisionTrace,
    Observation,
    OfferCard,
    OfferEvaluation,
    PlatformView,
    ScoreFactor,
)

from src.agent.geometry import haversine_km

_MINUTES_PER_HOUR = 60.0


def _app_rate_mxn_per_hour(offer: OfferCard) -> float:
    """MXN per hour the app's own numbers imply. Optimistic by construction."""
    minutes = max(1.0, offer.eta_minutes)
    return offer.payout_mxn / minutes * _MINUTES_PER_HOUR


def _idle_trace(minute: int, name: str) -> DecisionTrace:
    return DecisionTrace(
        minute=minute,
        considered=(),
        chosen_order_id=None,
        threshold_mxn_per_hour=0.0,
        binding_constraint="none",
        summary="No offers on screen at minute %d, so %s waits." % (minute, name),
    )


class AcceptAllPolicy:
    """Takes every offer. No threshold, no geometry, no memory.

    It is not a straw man: a courier who rejects nothing keeps a perfect
    acceptance rate and is never starved of offers. It loses on where the
    offers leave it, not on how many it gets.
    """

    name: str = "accept_all"

    def decide(
        self, view: PlatformView, observation: Observation, courier: CourierSnapshot
    ) -> Decision:
        if not view.offers:
            return Decision(
                action=Action.HOLD,
                order_id=None,
                target_cell=None,
                trace=_idle_trace(view.minute, self.name),
            )

        # First on screen: an accept-everything courier does not rank, it taps.
        chosen = view.offers[0]
        considered = tuple(
            OfferEvaluation(
                order_id=offer.order_id,
                expected_net_mxn=offer.payout_mxn,
                expected_minutes=offer.eta_minutes,
                expected_km=offer.distance_km,
                expected_mxn_per_hour=_app_rate_mxn_per_hour(offer),
                factors=(
                    ScoreFactor(
                        label="Payout on the card",
                        delta_mxn=offer.payout_mxn,
                        note="taken at face value",
                    ),
                    ScoreFactor(
                        label="App ETA",
                        delta_minutes=offer.eta_minutes,
                        note="the app's estimate, believed without checking",
                    ),
                ),
                rejected_because=None
                if offer.order_id == chosen.order_id
                else "only one offer can be accepted this minute",
                surge_flag=offer.surge_flag,
            )
            for offer in view.offers
        )

        return Decision(
            action=Action.ACCEPT,
            order_id=chosen.order_id,
            target_cell=None,
            trace=DecisionTrace(
                minute=view.minute,
                considered=considered,
                chosen_order_id=chosen.order_id,
                threshold_mxn_per_hour=0.0,
                binding_constraint="none",
                summary=(
                    "Accepted %s from %s for %.0f MXN because this policy accepts every offer "
                    "it is shown." % (chosen.order_id, chosen.restaurant_name, chosen.payout_mxn)
                ),
            ),
        )


class NearestFirstPolicy:
    """Takes whichever offer starts closest. The naive heuristic, done properly.

    It ranks by straight-line distance to the pickup and ignores everything
    else: the payout, where the delivery ends, the kitchen, the clock. Beating
    this is the bar that matters.
    """

    name: str = "nearest_first"

    def decide(
        self, view: PlatformView, observation: Observation, courier: CourierSnapshot
    ) -> Decision:
        if not view.offers:
            return Decision(
                action=Action.HOLD,
                order_id=None,
                target_cell=None,
                trace=_idle_trace(view.minute, self.name),
            )

        pickup_km = {
            offer.order_id: haversine_km(
                courier.lat, courier.lon, offer.pickup_lat, offer.pickup_lon
            )
            for offer in view.offers
        }
        chosen = min(view.offers, key=lambda offer: pickup_km[offer.order_id])

        considered = tuple(
            OfferEvaluation(
                order_id=offer.order_id,
                expected_net_mxn=offer.payout_mxn,
                expected_minutes=offer.eta_minutes,
                expected_km=pickup_km[offer.order_id] + offer.distance_km,
                expected_mxn_per_hour=_app_rate_mxn_per_hour(offer),
                factors=(
                    ScoreFactor(
                        label="Distance to pickup",
                        delta_minutes=None,
                        note="%.2f km away in a straight line" % pickup_km[offer.order_id],
                    ),
                    ScoreFactor(
                        label="Payout on the card",
                        delta_mxn=offer.payout_mxn,
                        note="recorded but not used for ranking",
                    ),
                ),
                rejected_because=None
                if offer.order_id == chosen.order_id
                else "a closer pickup was available (%.2f km vs %.2f km)"
                % (pickup_km[chosen.order_id], pickup_km[offer.order_id]),
                surge_flag=offer.surge_flag,
            )
            for offer in view.offers
        )

        return Decision(
            action=Action.ACCEPT,
            order_id=chosen.order_id,
            target_cell=None,
            trace=DecisionTrace(
                minute=view.minute,
                considered=considered,
                chosen_order_id=chosen.order_id,
                threshold_mxn_per_hour=0.0,
                binding_constraint="none",
                summary=(
                    "Accepted %s because its pickup is the closest on screen at %.2f km, which is "
                    "the only thing this policy looks at."
                    % (chosen.order_id, pickup_km[chosen.order_id])
                ),
            ),
        )
