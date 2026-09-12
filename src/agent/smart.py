"""The policy that should win.

It scores every offer in EXPECTED NET MXN PER HOUR: the seventh number, the one
the platform never shows. Payout divided by the app's ETA is not that number,
and the gap between the two is where the whole edge lives:

  - the minutes are the courier's own estimate, not the app's optimism;
  - the kitchen wait comes from what this courier remembers about this kitchen,
    and is discounted when they barely remember it;
  - the minutes AFTER the drop count too, because an order that strands you in
    a dead zone is charging you for the ride out of it;
  - late in the shift, an order pointing away from home is charging you for the
    unpaid ride back;
  - a shaky belief is not a fact, so a high score built on one is discounted;
  - and rejecting is not free: the platform feeds choosy couriers worse offers,
    so the policy prices its own choosiness.

Everything it knows arrives through `PlatformView`, `Observation` and
`CourierSnapshot`. There is no import path from here to the world, which is why
the trace it returns can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

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

from src.agent.calibration import (
    ACCEPTANCE_CALIBRATION,
    BELIEF_CALIBRATION,
    ECONOMICS_CALIBRATION,
    FUEL_CALIBRATION,
    HANDLING_CALIBRATION,
    REPOSITION_CALIBRATION,
    SAFETY_CALIBRATION,
    SHIFT_CALIBRATION,
)
from src.agent.forward_model import ForwardModel, clamp, ramp
from src.agent.geometry import CellIndex

_MINUTES_PER_HOUR = 60.0

# Why an offer was ruled out before scoring. Mapped to the trace's
# `binding_constraint` vocabulary by `_BINDING_OF`.
_BLOCK_UNPROFITABLE = "unprofitable"
_BLOCK_SHIFT = "shift"
_BLOCK_HOME = "home"
_BLOCK_FUEL = "fuel"

_BINDING_OF: dict[str, str] = {
    _BLOCK_UNPROFITABLE: "time_budget",
    _BLOCK_SHIFT: "time_budget",
    _BLOCK_HOME: "home",
    _BLOCK_FUEL: "fuel",
}


@dataclass(frozen=True)
class _Plan:
    """One offer, fully thought through, before the policy picks between them."""

    offer: OfferCard
    pickup_cell: str
    dropoff_cell: str

    paid_minutes: float       # ride + kitchen + handling: the job itself
    dead_minutes: float       # unpaid minutes the destination will cost after
    homeward_minutes: float   # unpaid ride home, weighted by how late it is
    total_minutes: float      # what this offer really takes out of the shift
    km: float
    return_minutes: float     # ride home from the drop-off, unweighted

    gross_mxn: float
    cost_mxn: float
    risk_mxn: float
    net_mxn: float

    confidence: float
    rate_before_uncertainty: float
    rate: float

    factors: tuple[ScoreFactor, ...]
    blocked: str | None
    blocked_note: str | None

    def evaluation(self, rejected_because: str | None) -> OfferEvaluation:
        return OfferEvaluation(
            order_id=self.offer.order_id,
            expected_net_mxn=round(self.net_mxn, 2),
            expected_minutes=round(self.total_minutes, 1),
            expected_km=round(self.km, 2),
            expected_mxn_per_hour=round(self.rate, 1),
            factors=self.factors,
            rejected_because=rejected_because,
        )


class SmartPolicy:
    """Expected net MXN per hour, with the costs the platform leaves out."""

    name: str = "smart"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name

    # ------------------------------------------------------------------
    # The port method
    # ------------------------------------------------------------------

    def decide(
        self, view: PlatformView, observation: Observation, courier: CourierSnapshot
    ) -> Decision:
        index = CellIndex.from_heatmap(view.heatmap, observation.at_cell)
        model = ForwardModel(observation, index)

        plans = [self._plan(offer, view, observation, courier, model, index) for offer in view.offers]
        threshold = self._threshold(observation, courier)

        feasible = [plan for plan in plans if plan.blocked is None]
        best = max(feasible, key=lambda plan: plan.rate) if feasible else None

        # 1. Fuel is time, and running dry mid-delivery costs far more time than
        #    planning the stop does.
        refuel = self._refuel_decision(plans, observation, courier, threshold, best, view.minute)
        if refuel is not None:
            return refuel

        # 2. Take the best offer if it clears the bar.
        if best is not None and best.rate >= threshold:
            return self._accept(best, plans, threshold, "none", view.minute)

        # 3. Being choosy is not free: below the floor the platform starves the
        #    courier of offers, so anything profitable beats holding out.
        if best is not None and self._acceptance_is_distressed(courier):
            return self._accept(best, plans, threshold, "acceptance_rate", view.minute)

        # 4. Nothing worth taking. Moving towards believed demand may beat idling.
        reposition = self._reposition_decision(
            plans, observation, courier, model, index, threshold, view.minute
        )
        if reposition is not None:
            return reposition

        return self._stand_down(plans, threshold, courier, view.minute)

    # ------------------------------------------------------------------
    # Scoring one offer
    # ------------------------------------------------------------------

    def _plan(
        self,
        offer: OfferCard,
        view: PlatformView,
        observation: Observation,
        courier: CourierSnapshot,
        model: ForwardModel,
        index: CellIndex,
    ) -> _Plan:
        pickup_cell = index.nearest(offer.pickup_lat, offer.pickup_lon)
        dropoff_cell = index.nearest(offer.dropoff_lat, offer.dropoff_lon)
        factors: list[ScoreFactor] = [
            ScoreFactor(
                label="Payout offered",
                delta_mxn=offer.payout_mxn,
                note="surge flagged by the app" if offer.surge_flag else "no surge flag",
            )
        ]

        # -- the job itself ------------------------------------------------
        to_pickup = model.travel(courier.lat, courier.lon, offer.pickup_lat, offer.pickup_lon)
        factors.append(
            ScoreFactor(
                label="Ride to the restaurant",
                delta_minutes=to_pickup.minutes,
                note="%.1f km at a believed traffic factor of %.2f"
                % (to_pickup.km, to_pickup.traffic_multiplier),
            )
        )

        kitchen = model.kitchen_wait(offer.restaurant_name)
        factors.append(
            ScoreFactor(
                label="Kitchen wait at %s" % offer.restaurant_name,
                delta_minutes=kitchen.value,
                note="remembered from this shift (confidence %.2f)" % kitchen.confidence
                if kitchen.known
                else "never delivered from here, using a %.0f minute prior"
                % HANDLING_CALIBRATION["default_kitchen_minutes"],
            )
        )

        to_customer = model.travel(
            offer.pickup_lat, offer.pickup_lon, offer.dropoff_lat, offer.dropoff_lon
        )
        factors.append(
            ScoreFactor(
                label="Ride to the customer",
                delta_minutes=to_customer.minutes,
                note="%.1f km at a believed traffic factor of %.2f; the app said %.0f min"
                % (to_customer.km, to_customer.traffic_multiplier, offer.eta_minutes),
            )
        )

        handling = (
            HANDLING_CALIBRATION["pickup_handling_minutes"]
            + HANDLING_CALIBRATION["dropoff_handling_minutes"]
        )
        factors.append(
            ScoreFactor(
                label="Parking and handover",
                delta_minutes=handling,
                note="fixed overhead at both ends",
            )
        )

        # -- disruptions the courier can currently perceive -----------------
        event_minutes = 0.0
        for delay in model.delays_on((pickup_cell, dropoff_cell)):
            event_minutes += delay.minutes
            factors.append(
                ScoreFactor(
                    label="Perceived %s (%s)" % (delay.event.kind, delay.event.event_id),
                    delta_minutes=delay.minutes,
                    note="reported on my route, believed at confidence %.2f"
                    % delay.event.confidence,
                )
            )

        paid_minutes = (
            to_pickup.minutes + kitchen.value + to_customer.minutes + handling + event_minutes
        )
        km = to_pickup.km + to_customer.km

        # -- money ----------------------------------------------------------
        cost_per_km = (
            ECONOMICS_CALIBRATION["fuel_cost_per_km_mxn"]
            + ECONOMICS_CALIBRATION["vehicle_wear_per_km_mxn"]
        )
        cost_mxn = km * cost_per_km
        factors.append(
            ScoreFactor(
                label="Fuel and wear",
                delta_mxn=-cost_mxn,
                note="%.1f km at %.2f MXN/km" % (km, cost_per_km),
            )
        )

        risk_mxn = 0.0
        night = model.night_factor(view.minute)
        if night > 0.0:
            night_cost = km * SAFETY_CALIBRATION["night_risk_mxn_per_km"] * night
            risk_mxn += night_cost
            factors.append(
                ScoreFactor(
                    label="Night risk premium",
                    delta_mxn=-night_cost,
                    note="riding after dark, weighted %.2f" % night,
                )
            )
        rain = model.rain_risk_factor()
        if rain > 0.0:
            rain_cost = km * SAFETY_CALIBRATION["rain_risk_mxn_per_km"] * rain
            risk_mxn += rain_cost
            factors.append(
                ScoreFactor(
                    label="Wet road risk premium",
                    delta_mxn=-rain_cost,
                    note="%.1f mm of believed rain" % observation.precip_mm.value,
                )
            )

        net_mxn = offer.payout_mxn - cost_mxn - risk_mxn

        # -- where it leaves the courier -------------------------------------
        demand = model.demand(dropoff_cell)
        dead_minutes = model.dead_minutes(demand.value)
        factors.append(
            ScoreFactor(
                label="Demand where it leaves me",
                delta_minutes=dead_minutes,
                note="believed demand %.2f in %s; unpaid minutes before the next offer"
                % (demand.value, dropoff_cell),
            )
        )

        # -- end-of-shift geometry -------------------------------------------
        home_from_dropoff = model.travel(
            offer.dropoff_lat, offer.dropoff_lon, courier.home_lat, courier.home_lon
        )
        home_from_here = model.travel(
            courier.lat, courier.lon, courier.home_lat, courier.home_lon
        )
        urgency = self._homeward_urgency(observation)
        homeward_minutes = (home_from_dropoff.minutes - home_from_here.minutes) * urgency
        if urgency > 0.0 and abs(homeward_minutes) >= 0.05:
            factors.append(
                ScoreFactor(
                    label="Unpaid ride home afterwards",
                    delta_minutes=homeward_minutes,
                    note="%.0f min from the drop-off against %.0f min from here, weighted %.2f "
                    "with %d min of shift left"
                    % (
                        home_from_dropoff.minutes,
                        home_from_here.minutes,
                        urgency,
                        observation.minutes_left_in_shift,
                    ),
                )
            )

        total_minutes = max(
            paid_minutes * 0.5, paid_minutes + dead_minutes + homeward_minutes
        )
        rate_before_uncertainty = net_mxn / total_minutes * _MINUTES_PER_HOUR

        # -- uncertainty -------------------------------------------------------
        confidence = self._blended_confidence(
            traffic=(to_pickup.confidence + to_customer.confidence) / 2.0,
            demand=demand.confidence,
            kitchen=kitchen.confidence,
        )
        discount = 1.0 - BELIEF_CALIBRATION["uncertainty_penalty"] * (1.0 - confidence)
        rate = rate_before_uncertainty * discount
        factors.append(
            ScoreFactor(
                label="Confidence discount",
                delta_mxn=(rate - rate_before_uncertainty) / _MINUTES_PER_HOUR * total_minutes,
                note="this score rests on beliefs worth %.2f, so %.0f%% of it is trusted"
                % (confidence, discount * 100.0),
            )
        )

        blocked, blocked_note = self._blocking_reason(
            net_mxn=net_mxn,
            paid_minutes=paid_minutes,
            return_minutes=home_from_dropoff.minutes,
            observation=observation,
            courier=courier,
        )

        return _Plan(
            offer=offer,
            pickup_cell=pickup_cell,
            dropoff_cell=dropoff_cell,
            paid_minutes=paid_minutes,
            dead_minutes=dead_minutes,
            homeward_minutes=homeward_minutes,
            total_minutes=total_minutes,
            km=km,
            return_minutes=home_from_dropoff.minutes,
            gross_mxn=offer.payout_mxn,
            cost_mxn=cost_mxn,
            risk_mxn=risk_mxn,
            net_mxn=net_mxn,
            confidence=confidence,
            rate_before_uncertainty=rate_before_uncertainty,
            rate=rate,
            factors=tuple(factors),
            blocked=blocked,
            blocked_note=blocked_note,
        )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _blocking_reason(
        self,
        *,
        net_mxn: float,
        paid_minutes: float,
        return_minutes: float,
        observation: Observation,
        courier: CourierSnapshot,
    ) -> tuple[str | None, str | None]:
        """Hard reasons an offer cannot be taken, checked before any ranking."""
        if net_mxn <= 0.0:
            return _BLOCK_UNPROFITABLE, "costs %.0f MXN more to run than it pays" % -net_mxn

        fuel_needed = paid_minutes + FUEL_CALIBRATION["job_margin_minutes"]
        if fuel_needed > courier.fuel_minutes_remaining:
            return _BLOCK_FUEL, (
                "needs about %.0f min of fuel including margin and I have %.0f"
                % (fuel_needed, courier.fuel_minutes_remaining)
            )

        left = float(observation.minutes_left_in_shift)
        if paid_minutes > left:
            return _BLOCK_SHIFT, (
                "takes about %.0f min and only %.0f min of shift remain"
                % (paid_minutes, left)
            )

        margin = SHIFT_CALIBRATION["home_margin_minutes"]
        if paid_minutes + return_minutes + margin > left:
            return _BLOCK_HOME, (
                "the job fits in the %.0f min left but the %.0f min ride home afterwards does not"
                % (left, return_minutes)
            )

        return None, None

    def _homeward_urgency(self, observation: Observation) -> float:
        """0 while the shift is long, 1 when it is nearly over."""
        return 1.0 - ramp(
            float(observation.minutes_left_in_shift),
            SHIFT_CALIBRATION["homeward_full_weight_below_minutes"],
            SHIFT_CALIBRATION["homeward_ignored_above_minutes"],
        )

    def _blended_confidence(self, *, traffic: float, demand: float, kitchen: float) -> float:
        weights = (
            BELIEF_CALIBRATION["weight_traffic"],
            BELIEF_CALIBRATION["weight_demand"],
            BELIEF_CALIBRATION["weight_kitchen"],
        )
        values = (traffic, demand, kitchen)
        total = sum(weights)
        return clamp(sum(w * v for w, v in zip(weights, values)) / total, 0.0, 1.0)

    def _threshold(self, observation: Observation, courier: CourierSnapshot) -> float:
        """The MXN/hour bar an offer must clear, adapted to the situation.

        Early in the shift a courier can afford to wait for a good one. Late,
        or with an acceptance rate the platform is watching, they cannot.
        """
        shift_progress = ramp(
            float(observation.minutes_left_in_shift),
            SHIFT_CALIBRATION["homeward_full_weight_below_minutes"],
            SHIFT_CALIBRATION["homeward_ignored_above_minutes"],
        )
        shift_factor = SHIFT_CALIBRATION["desperate_factor_late"] + shift_progress * (
            SHIFT_CALIBRATION["choosy_factor_early"] - SHIFT_CALIBRATION["desperate_factor_late"]
        )

        if courier.offers_seen < ACCEPTANCE_CALIBRATION["min_offers_for_signal"]:
            acceptance_factor = 1.0
        else:
            standing = ramp(
                courier.acceptance_rate,
                ACCEPTANCE_CALIBRATION["distressed_rate"],
                ACCEPTANCE_CALIBRATION["comfortable_rate"],
            )
            acceptance_factor = ACCEPTANCE_CALIBRATION["distressed_threshold_factor"] + standing * (
                ACCEPTANCE_CALIBRATION["comfortable_threshold_factor"]
                - ACCEPTANCE_CALIBRATION["distressed_threshold_factor"]
            )

        return SHIFT_CALIBRATION["base_reservation_mxn_per_hour"] * shift_factor * acceptance_factor

    def _acceptance_is_distressed(self, courier: CourierSnapshot) -> bool:
        if courier.offers_seen < ACCEPTANCE_CALIBRATION["min_offers_for_signal"]:
            return False
        return courier.acceptance_rate < ACCEPTANCE_CALIBRATION["hard_floor_rate"]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _accept(
        self,
        chosen: _Plan,
        plans: list[_Plan],
        threshold: float,
        binding: str,
        minute: int,
    ) -> Decision:
        if binding == "acceptance_rate":
            summary = (
                "Accepted %s at %.0f net MXN per hour even though my bar is %.0f, because my "
                "acceptance rate is low enough that the app will start starving me of offers."
                % (chosen.offer.order_id, chosen.rate, threshold)
            )
        else:
            summary = (
                "Accepted %s from %s: about %.0f net MXN over %.0f minutes is %.0f MXN per hour "
                "against a bar of %.0f, and it leaves me in %s."
                % (
                    chosen.offer.order_id,
                    chosen.offer.restaurant_name,
                    chosen.net_mxn,
                    chosen.total_minutes,
                    chosen.rate,
                    threshold,
                    chosen.dropoff_cell,
                )
            )

        return Decision(
            action=Action.ACCEPT,
            order_id=chosen.offer.order_id,
            target_cell=None,
            trace=DecisionTrace(
                minute=minute,
                considered=self._considered(plans, chosen, threshold),
                chosen_order_id=chosen.offer.order_id,
                threshold_mxn_per_hour=round(threshold, 1),
                binding_constraint=binding,
                summary=summary,
            ),
        )

    def _refuel_decision(
        self,
        plans: list[_Plan],
        observation: Observation,
        courier: CourierSnapshot,
        threshold: float,
        best: _Plan | None,
        minute: int,
    ) -> Decision | None:
        """Plan the stop instead of being ambushed by it.

        Refuelling buys no pesos, it spends minutes. So it is worth doing only
        while there is enough shift left to earn those minutes back.
        """
        shift_left = float(observation.minutes_left_in_shift)
        worth_stopping = shift_left >= (
            FUEL_CALIBRATION["stop_minutes"] + FUEL_CALIBRATION["min_useful_shift_minutes"]
        )
        if not worth_stopping:
            return None

        below_reserve = courier.fuel_minutes_remaining < FUEL_CALIBRATION["reserve_minutes"]
        blocked_by_fuel = any(plan.blocked == _BLOCK_FUEL for plan in plans)
        taking_a_job_now = best is not None and best.rate >= threshold

        if below_reserve:
            reason = (
                "Refuelling now: %.0f minutes of range left is under my %.0f minute reserve, and "
                "running dry mid-delivery costs far more than the %.0f minute stop."
                % (
                    courier.fuel_minutes_remaining,
                    FUEL_CALIBRATION["reserve_minutes"],
                    FUEL_CALIBRATION["stop_minutes"],
                )
            )
        elif blocked_by_fuel and not taking_a_job_now:
            reason = (
                "Refuelling now: the offers on screen need more range than the %.0f minutes I "
                "have left, so the stop is the cheapest way to stay working."
                % courier.fuel_minutes_remaining
            )
        else:
            return None

        return Decision(
            action=Action.REFUEL,
            order_id=None,
            target_cell=None,
            trace=DecisionTrace(
                minute=minute,
                considered=self._considered(plans, None, threshold),
                chosen_order_id=None,
                threshold_mxn_per_hour=round(threshold, 1),
                binding_constraint="fuel",
                summary=reason,
            ),
        )

    def _reposition_decision(
        self,
        plans: list[_Plan],
        observation: Observation,
        courier: CourierSnapshot,
        model: ForwardModel,
        index: CellIndex,
        threshold: float,
        minute: int,
    ) -> Decision | None:
        """Move towards believed demand, but only when the move pays for itself.

        Repositioning costs kilometres and minutes and earns nothing, so the
        expected unpaid wait it saves has to be bigger than the ride.
        """
        here_demand = model.demand(observation.at_cell)
        wait_here = model.dead_minutes(here_demand.value)

        best_cell: str | None = None
        best_gain_minutes = 0.0
        best_km = 0.0
        best_travel_minutes = 0.0
        best_demand = 0.0

        for cell in observation.demand_by_cell:
            if cell == observation.at_cell:
                continue
            coords = index.coords_of(cell)
            if coords is None:
                continue  # the agent has no idea where that cell is
            leg = model.travel(observation.at_lat, observation.at_lon, coords[0], coords[1])
            if leg.km > REPOSITION_CALIBRATION["max_reposition_km"]:
                continue
            if leg.minutes + FUEL_CALIBRATION["job_margin_minutes"] > courier.fuel_minutes_remaining:
                continue
            if leg.minutes * 2.0 > float(observation.minutes_left_in_shift):
                continue
            there = model.demand(cell)
            gain = wait_here - (leg.minutes + model.dead_minutes(there.value))
            if gain > best_gain_minutes:
                best_gain_minutes = gain
                best_cell = cell
                best_km = leg.km
                best_travel_minutes = leg.minutes
                best_demand = there.value

        if best_cell is None or best_gain_minutes < REPOSITION_CALIBRATION["min_gain_minutes"]:
            return None

        cost_per_km = (
            ECONOMICS_CALIBRATION["fuel_cost_per_km_mxn"]
            + ECONOMICS_CALIBRATION["vehicle_wear_per_km_mxn"]
        )
        gain_mxn = (
            best_gain_minutes * ECONOMICS_CALIBRATION["opportunity_mxn_per_minute"]
            - best_km * cost_per_km
        )
        if gain_mxn < REPOSITION_CALIBRATION["min_gain_mxn"]:
            return None

        summary = (
            "Nothing on screen is worth %.0f MXN per hour, so I am riding %.1f km to %s: I believe "
            "demand there is %.2f against %.2f here, which should save about %.0f unpaid minutes "
            "for %.0f minutes of travel."
            % (
                threshold,
                best_km,
                best_cell,
                best_demand,
                here_demand.value,
                best_gain_minutes,
                best_travel_minutes,
            )
        )
        return Decision(
            action=Action.REPOSITION,
            order_id=None,
            target_cell=best_cell,
            trace=DecisionTrace(
                minute=minute,
                considered=self._considered(plans, None, threshold),
                chosen_order_id=None,
                threshold_mxn_per_hour=round(threshold, 1),
                binding_constraint=self._binding_when_idle(plans),
                summary=summary,
            ),
        )

    def _stand_down(
        self,
        plans: list[_Plan],
        threshold: float,
        courier: CourierSnapshot,
        minute: int,
    ) -> Decision:
        if not plans:
            summary = (
                "No offers on screen and no cell nearby I believe is busier than this one, so I "
                "hold position at minute %d." % minute
            )
            action = Action.HOLD
        else:
            best = max(plans, key=lambda plan: plan.rate)
            if best.blocked is not None:
                reason = best.blocked_note or "it cannot be completed"
            else:
                reason = "it scores %.0f MXN per hour against my %.0f bar" % (best.rate, threshold)
            summary = (
                "Rejected all %d offers: the best of them, %s, is out because %s."
                % (len(plans), best.offer.order_id, reason)
            )
            action = Action.REJECT

        return Decision(
            action=action,
            order_id=None,
            target_cell=None,
            trace=DecisionTrace(
                minute=minute,
                considered=self._considered(plans, None, threshold),
                chosen_order_id=None,
                threshold_mxn_per_hour=round(threshold, 1),
                binding_constraint=self._binding_when_idle(plans),
                summary=summary,
            ),
        )

    # ------------------------------------------------------------------
    # Trace assembly
    # ------------------------------------------------------------------

    def _binding_when_idle(self, plans: list[_Plan]) -> str:
        if not plans:
            return "none"
        best = max(plans, key=lambda plan: plan.rate)
        if best.blocked is not None:
            return _BINDING_OF[best.blocked]
        return "time_budget"

    def _considered(
        self, plans: list[_Plan], chosen: _Plan | None, threshold: float
    ) -> tuple[OfferEvaluation, ...]:
        evaluations: list[OfferEvaluation] = []
        for plan in plans:
            if chosen is not None and plan is chosen:
                evaluations.append(plan.evaluation(None))
                continue
            if plan.blocked is not None:
                reason = plan.blocked_note or plan.blocked
            elif chosen is not None:
                reason = "%s scored higher (%.0f against %.0f MXN per hour)" % (
                    chosen.offer.order_id,
                    chosen.rate,
                    plan.rate,
                )
            else:
                reason = "%.0f MXN per hour is below my %.0f bar" % (plan.rate, threshold)
            evaluations.append(plan.evaluation(reason))
        return tuple(evaluations)
