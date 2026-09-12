"""The policies to beat.

Two of them, because beating one weak baseline proves nothing.

`AcceptAllPolicy` is the naive floor: what an inexperienced courier does on
day one. `FixedPayoutThresholdPolicy` is the real bar — the rule of thumb an
experienced but unaided courier actually uses: "I don't take anything under
fifty pesos." (The floor is calibrated to 55 rather than 50 because 55 is
what measured strongest; see `calibration.BASELINE_CALIBRATION` for the
whole swept curve and why the choice is the hardest opponent rather than
the most convenient one.)

WHY NOT NEAREST-FIRST. This package used to ship a `NearestFirstPolicy` that
ranked the offers on screen by distance to the pickup. That is a MENU
heuristic, and this app is not a menu: it pushes one offer at a time, fresh,
live for the minute it appears. "Take the nearest of what is on screen" then
picks the only card on screen, so its decisions — and its numbers — were
arithmetically identical to accepting everything, byte for byte on every
seed. Two baselines printing the same row is not two baselines.

A payout floor is the right shape for a flow. It is a decision about ONE
offer, made without reference to any other, which is exactly the decision the
app forces; it is what real couriers describe when you ask them how they
choose; and it is genuinely good — it cuts the loss-making tail without ever
leaving the courier holding out for perfection. It is a fair opponent, which
is what makes beating it mean something.

Both baselines return a genuine `DecisionTrace`. A comparison where only one
side can explain itself is not a comparison, it is an advertisement.

Both also do something the smart policy refuses to do: they believe the app.
They score with `eta_minutes` and `distance_km` straight off the offer card,
and the payout floor is applied to the gross payout on the card rather than
to anything net of fuel, wear, kitchen wait or the unpaid minutes afterwards.
That is exactly the naivety being measured: the number on the card is not the
number in the courier's pocket.
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

from src.agent.calibration import BASELINE_CALIBRATION

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


class FixedPayoutThresholdPolicy:
    """"I don't take anything under fifty-odd pesos." The real bar.

    One number off `BASELINE_CALIBRATION`, applied to the gross payout on the
    card. No per-hour arithmetic, no kilometres, no clock, no memory of which
    kitchens are slow, no idea where the drop-off leaves them — the courier
    reads one figure and decides. That is what an experienced but unaided
    courier actually does, and it is a strong rule: it refuses the
    loss-making tail of the flow without ever holding out for perfection.

    Three concessions to realism, all calibrated rather than coded:

      - Late in the shift the floor relaxes (`late_shift_floor_factor`). A
        real courier's discipline slackens in the last hour, because an order
        that pays something beats riding home empty. Without this the
        baseline would reject its way through the end of every shift, which
        is a straw man, not a courier.
      - Below `acceptance_rescue_rate` the floor is suspended entirely. The
        app shows a courier their acceptance rate and every courier knows
        what a low one costs them, so they start taking work again once the
        screen goes quiet. Without this the baseline destroys itself on a
        window that opens on thin payouts: measured on the Night window it
        rejected its opening offers, the platform cut its reach in reply,
        and it finished the shift with zero deliveries and 0.0 MXN/h.
      - Given more than one card at once — which this app's flow does not
        normally do, but the port allows — it takes the best-paying of those
        clearing the floor. Still one number, still no arithmetic.

    What it cannot see is the whole point of the comparison: 60 MXN over a
    45-minute round trip into a dead zone clears the floor and loses the
    courier money, and 50 MXN two streets away does not clear it and would
    have been the best-paid hour of the shift.
    """

    name: str = "fixed_threshold"

    def __init__(self, floor_mxn: float | None = None) -> None:
        self._floor_mxn = (
            float(floor_mxn)
            if floor_mxn is not None
            else BASELINE_CALIBRATION["fixed_payout_floor_mxn"]
        )

    def _starved(self, courier: CourierSnapshot) -> bool:
        """Has the app started punishing this courier for being choosy?

        Both numbers are ones the courier is shown: their own tally of
        offers, and the acceptance rate the app puts on their own screen.
        """
        if courier.offers_seen < BASELINE_CALIBRATION["min_offers_before_rescue"]:
            return False
        return courier.acceptance_rate < BASELINE_CALIBRATION["acceptance_rescue_rate"]

    def _floor_now(self, observation: Observation, courier: CourierSnapshot) -> float:
        """The floor this minute: relaxed near the bell, suspended when the
        app has started starving the courier for a low acceptance rate."""
        if self._starved(courier):
            return 0.0
        if float(observation.minutes_left_in_shift) <= BASELINE_CALIBRATION["late_shift_minutes"]:
            return self._floor_mxn * BASELINE_CALIBRATION["late_shift_floor_factor"]
        return self._floor_mxn

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

        starved = self._starved(courier)
        floor = self._floor_now(observation, courier)
        # Why the floor is not the calibrated one, if it is not: the two get
        # different words in the trace, because they are different reasons.
        relaxation = (
            " (suspended: my acceptance rate is %.0f%% and the app has gone quiet)"
            % (courier.acceptance_rate * 100.0)
            if starved
            else " (relaxed, the shift is nearly over)" if floor < self._floor_mxn else ""
        )
        clearing = [offer for offer in view.offers if offer.payout_mxn >= floor]
        chosen = max(clearing, key=lambda offer: offer.payout_mxn) if clearing else None

        considered = tuple(
            OfferEvaluation(
                order_id=offer.order_id,
                expected_net_mxn=offer.payout_mxn,
                expected_minutes=offer.eta_minutes,
                expected_km=offer.distance_km,
                expected_mxn_per_hour=_app_rate_mxn_per_hour(offer),
                factors=(
                    ScoreFactor(
                        label="Payout on the card against my floor",
                        delta_mxn=offer.payout_mxn,
                        note="%.0f MXN against a floor of %.0f MXN%s"
                        % (offer.payout_mxn, floor, relaxation),
                    ),
                    ScoreFactor(
                        label="Everything else on the card",
                        delta_minutes=offer.eta_minutes,
                        note="the app says %.0f min and %.1f km; this policy does not look at "
                        "either, nor at where the drop-off leaves me"
                        % (offer.eta_minutes, offer.distance_km),
                    ),
                ),
                rejected_because=None
                if chosen is not None and offer.order_id == chosen.order_id
                else (
                    "%.0f MXN is under my %.0f MXN floor" % (offer.payout_mxn, floor)
                    if offer.payout_mxn < floor
                    else "%s paid more (%.0f MXN against %.0f MXN)"
                    % (chosen.order_id, chosen.payout_mxn, offer.payout_mxn)
                ),
                surge_flag=offer.surge_flag,
            )
            for offer in view.offers
        )

        if chosen is None:
            best = max(view.offers, key=lambda offer: offer.payout_mxn)
            return Decision(
                action=Action.REJECT,
                order_id=None,
                target_cell=None,
                trace=DecisionTrace(
                    minute=view.minute,
                    considered=considered,
                    chosen_order_id=None,
                    # This field is in MXN per HOUR and this policy's bar is
                    # not: it is a flat peso floor on the card, deliberately
                    # innocent of how long the job takes. Reporting the floor
                    # here would put pesos in a per-hour column, so the bar
                    # is stated in the summary and in every factor instead.
                    threshold_mxn_per_hour=0.0,
                    binding_constraint="none",
                    summary=(
                        "Rejected all %d offers: the best of them, %s, pays %.0f MXN and my rule "
                        "is that I do not take anything under %.0f MXN."
                        % (len(view.offers), best.order_id, best.payout_mxn, floor)
                    ),
                ),
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
                binding_constraint="acceptance_rate" if starved else "none",
                summary=(
                    "Took %s from %s for %.0f MXN: my acceptance rate has fallen to %.0f%% and "
                    "the app has stopped sending me work, so I am not turning anything down "
                    "until that recovers."
                    % (
                        chosen.order_id,
                        chosen.restaurant_name,
                        chosen.payout_mxn,
                        courier.acceptance_rate * 100.0,
                    )
                    if starved
                    else "Accepted %s from %s: it pays %.0f MXN, which clears the %.0f MXN floor "
                    "I hold myself to%s."
                    % (
                        chosen.order_id,
                        chosen.restaurant_name,
                        chosen.payout_mxn,
                        floor,
                        ", relaxed because the shift is nearly over"
                        if floor < self._floor_mxn
                        else "",
                    )
                ),
            ),
        )
