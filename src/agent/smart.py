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

And it knows the difference between the two questions the app actually asks.
The app shows a FLOW of offers with seconds to decide, not a menu, so the real
question is never "which of these is best" — it is:

  - FREE: is this worth more than waiting for the next one? That is an
    optimal-stopping problem, and its answer is a reservation price computed
    from how often offers arrive and what they are typically worth, both
    measured off this courier's own shift. Not a constant: at three offers an
    hour, fussiness is just unpaid waiting.
  - BUSY: is this worth COMMITTING to before I know what else is coming? The
    courier may accept their next job while still finishing this one. Doing so
    erases the idle gap and starts that kitchen cooking early; it also spends
    the option on something better, and the ride to that restaurant starts
    from where the CURRENT job drops them, not from here. So the bar for
    queueing ahead is higher while a long job still has far to run, and falls
    toward the idle bar as it ends.

Everything it knows arrives through `PlatformView`, `Observation` and
`CourierSnapshot`, plus its own memory of what it agreed to and where it has
been. There is no import path from here to the world, which is why the trace
it returns can be trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
    RESERVATION_CALIBRATION,
    SAFETY_CALIBRATION,
    SHIFT_CALIBRATION,
)
from src.agent.forward_model import ForwardModel, clamp, ramp
from src.agent.geometry import CellIndex, haversine_km

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


@dataclass
class _HeldJob:
    """A job this policy accepted and has not seen delivered yet.

    The policy keeps its own record because the port tells it only WHICH
    orders it is holding (`CourierSnapshot.carrying_order_ids`), never how
    long they have to run or where they end. Both matter:

      - `finish_estimate_min` is how the policy knows how much committed
        work stands between now and its next free minute, which is what
        decides whether queueing another job ahead costs it anything.
      - `dropoff_lat`/`dropoff_lon` are where the courier will BE when they
        start that queued job. Scoring the ride to its restaurant from the
        courier's current position instead — which is what a policy without
        this memory has to do — systematically understates it, because the
        courier is about to ride several kilometres away from here first.

    This is the courier's own memory of what they agreed to. Nothing here
    comes from anywhere but decisions this policy made itself.
    """

    order_id: str
    finish_estimate_min: float
    dropoff_lat: float
    dropoff_lon: float


@dataclass
class _ShiftMemory:
    """What this courier has learned since clocking on.

    Reset whenever a new shift starts. Everything in it is either something
    the courier was shown, something they did, or something they stood in
    the middle of.
    """

    last_minute: int = -1
    # (rate, confidence-discounted net MXN, minutes) for every offer this
    # policy has scored this shift, newest last. This IS the courier's
    # belief about the value distribution of the offer flow — the empirical
    # one they have lived through, not an assumed shape.
    scored: list[tuple[float, float, float]] = field(default_factory=list)
    # Jobs accepted and not yet known to be finished, oldest first.
    held: list[_HeldJob] = field(default_factory=list)
    # Consecutive minutes stood free with nothing on screen IN THE CELL
    # NAMED BY `idle_streak_cell`. Direct evidence about THAT spot, and the
    # only evidence about it the courier cannot argue with: whatever they
    # believed about local demand, an empty screen for twenty minutes says
    # the flow does not reach there.
    #
    # The cell is recorded alongside the count because the count is evidence
    # about one place and nowhere else. Carrying it across a move is what
    # turns "this corner is dead, go somewhere else" into a thrash: the
    # courier arrives in a new cell already holding forty minutes of proof
    # that somewhere ELSE was dead, immediately concludes this cell is dead
    # too, and rides on. Measured before this was tracked: 381 repositions
    # in a 540-minute shift, 75% of it idle, four offers seen all day, 14.7
    # MXN/h against a baseline's 72.9.
    idle_streak_minutes: float = 0.0
    idle_streak_cell: str = ""
    # The cell this policy last set off for, and the cell it was standing in
    # when it did. A courier knows whether they actually got anywhere, and
    # the answer is not always yes: "ride to that district" can name a place
    # the courier then cannot start for, and repeating an instruction that
    # demonstrably did nothing is how a whole shift disappears. Any target
    # asked for and observably not reached goes in `unreachable` and is
    # never asked for again this shift.
    pending_target: str = ""
    pending_from_cell: str = ""
    unreachable: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.last_minute = -1
        self.scored = []
        self.held = []
        self.idle_streak_minutes = 0.0
        self.idle_streak_cell = ""
        self.pending_target = ""
        self.pending_from_cell = ""
        self.unreachable = set()

    def record(self, rate: float, net_mxn: float, minutes: float, window: int) -> None:
        """Remember one scored offer, keeping only the recent `window`.

        Bounded on purpose: the offer flow at 21:00 is not the offer flow at
        14:00, so a reservation price built from the whole shift would keep
        arguing with a lunchtime that is over.
        """
        self.scored.append((rate, net_mxn, minutes))
        if len(self.scored) > window:
            del self.scored[: len(self.scored) - window]

    @property
    def offers_scored(self) -> int:
        return len(self.scored)


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
            # Carried through so the evaluation layer can measure surge
            # capture: the share of ACCEPTED orders that carried surge
            # against the share among all offers seen.
            surge_flag=self.offer.surge_flag,
        )


class SmartPolicy:
    """Expected net MXN per hour, with the costs the platform leaves out."""

    name: str = "smart"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name
        self._memory = _ShiftMemory()

    # ------------------------------------------------------------------
    # The port method
    # ------------------------------------------------------------------

    def decide(
        self, view: PlatformView, observation: Observation, courier: CourierSnapshot
    ) -> Decision:
        memory = self._memory
        self._begin_minute(view, observation, courier)

        # The courier's spatial vocabulary this minute: the app's own coarse
        # heatmap cells plus `Observation.cell_coords`, which places every
        # cell the courier's tools have an opinion about. Both belief maps
        # become usable at minute one rather than being learned a cell at a
        # time — see `CellIndex.from_heatmap`.
        index = CellIndex.from_heatmap(view.heatmap, observation.at_cell, observation.cell_coords)
        model = ForwardModel(observation, index)

        # Where the courier will actually BE when this job starts, and how
        # many committed minutes stand between now and then. Both are zero /
        # "here" when the courier is free; both matter when they are not.
        start_lat, start_lon, committed_minutes = self._projected_start(courier, observation)
        committing_early = bool(courier.carrying_order_ids)

        plans = [
            self._plan(offer, view, observation, courier, model, index, start_lat, start_lon)
            for offer in view.offers
        ]
        for plan in plans:
            memory.record(
                rate=plan.rate,
                # The confidence-discounted net, so the value distribution
                # and the rate being compared against it are the same
                # quantity in the same units.
                net_mxn=plan.rate * plan.total_minutes / _MINUTES_PER_HOUR,
                minutes=plan.total_minutes,
                window=int(RESERVATION_CALIBRATION["recent_offer_window"]),
            )

        threshold = self._threshold(observation, courier, committed_minutes)

        feasible = [plan for plan in plans if plan.blocked is None]
        best = max(feasible, key=lambda plan: plan.rate) if feasible else None

        if committing_early:
            # Mid-job: the only thing the courier can do with their phone is
            # take the next order or leave it. Repositioning, refuelling and
            # resting are not available while committed to a leg, and the
            # engine would ignore them anyway.
            if best is not None and best.rate >= threshold:
                return self._accept(best, plans, threshold, "none", view.minute,
                                    committed_minutes=committed_minutes, courier=courier)
            return self._stand_down(plans, threshold, courier, view.minute,
                                    committed_minutes=committed_minutes)

        # 1. Fuel is time, and running dry mid-delivery costs far more time than
        #    planning the stop does.
        refuel = self._refuel_decision(plans, observation, courier, threshold, best, view.minute)
        if refuel is not None:
            return refuel

        # 2. Take the best offer if it clears the bar.
        if best is not None and best.rate >= threshold:
            return self._accept(best, plans, threshold, "none", view.minute,
                                committed_minutes=committed_minutes, courier=courier)

        # 3. Being choosy is not free: below the floor the platform starves the
        #    courier of offers, so anything profitable beats holding out.
        if best is not None and self._acceptance_is_distressed(courier):
            return self._accept(best, plans, threshold, "acceptance_rate", view.minute,
                                committed_minutes=committed_minutes, courier=courier)

        # 4. Nothing worth taking. Moving towards believed demand may beat idling.
        reposition = self._reposition_decision(
            plans, observation, courier, model, index, threshold, view.minute
        )
        if reposition is not None:
            return reposition

        return self._stand_down(plans, threshold, courier, view.minute,
                                committed_minutes=committed_minutes)

    # ------------------------------------------------------------------
    # Shift memory
    # ------------------------------------------------------------------

    def _begin_minute(
        self, view: PlatformView, observation: Observation, courier: CourierSnapshot
    ) -> None:
        """Housekeeping the courier does for free: notice a new shift has
        started, drop the record of any job they are no longer holding, and
        keep count of how long they have stood with an empty screen."""
        memory = self._memory
        if view.minute < memory.last_minute or courier.minutes_elapsed <= 1.0:
            memory.reset()
        memory.last_minute = view.minute

        still_held = set(courier.carrying_order_ids)
        memory.held = [job for job in memory.held if job.order_id in still_held]

        # Did the last move actually happen? The courier set off for
        # somewhere and is being asked again, from the same cell they set
        # off from — so they never went. Whatever that instruction named,
        # they cannot get there, and asking for it again next minute (and
        # the minute after) is how a shift evaporates. Note it and stop
        # asking. See `_ShiftMemory.unreachable`.
        if memory.pending_target:
            if observation.at_cell == memory.pending_from_cell:
                memory.unreachable.add(memory.pending_target)
            memory.pending_target = ""
            memory.pending_from_cell = ""

        # An empty screen is evidence about the spot the courier is standing
        # in. Standing somewhere else makes it evidence about somewhere
        # else, so the count starts again — see `_ShiftMemory`.
        if observation.at_cell != memory.idle_streak_cell:
            memory.idle_streak_cell = observation.at_cell
            memory.idle_streak_minutes = 0.0

        if view.offers or courier.carrying_order_ids:
            memory.idle_streak_minutes = 0.0
        else:
            memory.idle_streak_minutes += 1.0

    def _projected_start(
        self, courier: CourierSnapshot, observation: Observation
    ) -> tuple[float, float, float]:
        """(lat, lon, committed_minutes) for the moment a NEW job would start.

        Free courier: right here, right now. Committed courier: wherever the
        last job they are holding drops them, after however many minutes of
        that work is left.
        """
        memory = self._memory
        if not courier.carrying_order_ids or not memory.held:
            return courier.lat, courier.lon, 0.0
        last = memory.held[-1]
        committed = max(0.0, last.finish_estimate_min - float(observation.minute))
        return last.dropoff_lat, last.dropoff_lon, committed

    def _measured_wait_minutes(self, courier: CourierSnapshot) -> float | None:
        """How long an ordinary wait for the next offer actually is, tonight.

        `None` until the courier has lived through enough of the shift for
        their own arrival count to mean anything — before that the fixed
        prior in `DESTINATION_CALIBRATION` is the honest answer.
        """
        if self._memory.offers_scored < RESERVATION_CALIBRATION["min_offers_for_running_mean"]:
            return None
        return 1.0 / self._offers_per_minute(courier)

    def _offers_per_minute(self, courier: CourierSnapshot) -> float:
        """How often an offer arrives overall, from the courier's own tally.

        `offers_seen` and `minutes_elapsed` are both self-knowledge. The
        prior stops a single early offer (or none) from setting a nonsense
        arrival rate; the clamps stop a degenerate one either way.

        LIFETIME, and that is a deliberate choice against a plausible
        alternative rather than an oversight. This is an average over the
        whole shift, and a shift is not one flow: on the Night window the
        offer flow collapses after 23:00 (ground truth drops from 0.85 of
        peak to 0.05 across midnight), and a courier who has just worked the
        dinner peak carries that peak in their lifetime average forever.
        Instrumented, the policy believed 5.9-6.1 offers an hour after
        midnight while the flow it actually experienced was 0.6-3.2, and
        "below the bar" was its commonest refusal of that shift (76 against
        8 blocked by the end-of-shift constraint). Earnings by hour of day,
        four seeds: it WON the 18:00-22:00 block 1721 MXN to accept-all's
        1541, and lost 23:00-02:00 by 692 to 1679. So the mis-specification
        is real and it is where the Night deficit lives.

        A recency-windowed rate (90 minutes, counted off this policy's own
        log of what it was shown, blended with the same prior) was therefore
        implemented and measured over 6 seeds x 4 windows. It made the
        policy WORSE, including on the window it was aimed at:

            window      lifetime   90-minute window
            early           87.5   83.9
            day             96.5   91.1
            reference      112.3   113.7
            night           79.7   77.0

        The reason it backfires is that this rate is used for two things
        that pull opposite ways. A lower believed arrival rate lowers the
        reservation price, which is the intended effect — but it also
        lengthens `_measured_wait_minutes`, which inflates `dead_minutes`
        for every destination, which lowers every offer's score. The two
        cancel, and on three of four windows the noise of a short window
        costs more than the adaptiveness buys. Fixing the Night window needs
        the reservation price to know the flow is ending WITHOUT making
        every destination look worse at the same time, and that is a
        two-term change this revision did not earn the evidence for.
        """
        return self._blended_rate(courier.offers_seen, courier.minutes_elapsed)

    def _blended_rate(self, offers: float, minutes: float) -> float:
        cal = RESERVATION_CALIBRATION
        prior_minutes = cal["prior_weight_minutes"]
        prior_offers = cal["prior_offers_per_hour"] / _MINUTES_PER_HOUR * prior_minutes
        rate = (prior_offers + offers) / (prior_minutes + max(0.0, minutes))
        return clamp(
            rate,
            cal["min_offers_per_hour"] / _MINUTES_PER_HOUR,
            cal["max_offers_per_hour"] / _MINUTES_PER_HOUR,
        )

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
        start_lat: float,
        start_lon: float,
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
        # From where the courier will BE when this job starts, which is not
        # where they are standing if they are still finishing another one.
        to_pickup = model.travel(start_lat, start_lon, offer.pickup_lat, offer.pickup_lon)
        away_from_here_km = haversine_km(start_lat, start_lon, courier.lat, courier.lon)
        factors.append(
            ScoreFactor(
                label="Ride to the restaurant",
                delta_minutes=to_pickup.minutes,
                note="%.1f km at a believed traffic factor of %.2f%s"
                % (
                    to_pickup.km,
                    to_pickup.traffic_multiplier,
                    "" if away_from_here_km < 0.05
                    else ", measured from where my current job drops me (%.1f km from here)"
                    % away_from_here_km,
                ),
            )
        )

        kitchen = model.kitchen_wait(offer.restaurant_denue_id, offer.restaurant_name)
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
        demand = model.demand(dropoff_cell, offer.dropoff_lat, offer.dropoff_lon)
        dead_minutes = model.dead_minutes(demand.value, self._measured_wait_minutes(courier))
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
            start_lat, start_lon, courier.home_lat, courier.home_lon
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

    def _threshold(
        self, observation: Observation, courier: CourierSnapshot, committed_minutes: float = 0.0
    ) -> float:
        """The MXN/hour bar an offer must clear: what rejecting is worth.

        Not a constant. Rejecting buys the chance of a better offer and
        charges the unpaid minutes until one turns up, so the bar is the
        rate the courier expects from waiting:

            mean_net / (mean_minutes + wait) x 60

        `wait` is `1 / arrival_rate`, both measured from this courier's own
        shift. A courier already committed to `committed_minutes` of work
        does not pay that wait — the next offer arrives while they are still
        riding — so their bar for QUEUEING a job ahead uses only the part of
        the wait their current job does not already cover. With a long job
        still to run the bar rises to "better than average or leave it";
        as the job nears its end it falls back toward the idle bar, because
        an empty screen at the moment they go free costs real idle minutes.

        Falls back to the fixed cold-start bar until enough offers have been
        scored for a running mean to mean anything. See
        `calibration.RESERVATION_CALIBRATION` for why a constant cannot be
        the answer and what the old one measured as.
        """
        cal = RESERVATION_CALIBRATION
        memory = self._memory
        cold_start = self._cold_start_threshold(observation, courier)
        if memory.offers_scored < cal["min_offers_for_running_mean"]:
            return cold_start

        arrival_rate = self._offers_per_minute(courier)
        # Sorted best-first, so the first k entries are exactly the offers a
        # bar set at the k-th rate would have accepted.
        ranked = sorted(memory.scored, key=lambda item: item[0], reverse=True)
        n = len(ranked)

        best_value = 0.0
        take_everything_value = 0.0
        net_sum = 0.0
        minutes_sum = 0.0
        min_sample = int(cal["min_accepted_sample"])
        for k, (_rate, net_mxn, minutes) in enumerate(ranked, start=1):
            net_sum += net_mxn
            minutes_sum += minutes
            if k < min_sample:
                # A bar only this-many offers ever cleared is a bar set from
                # a sample too small to mean anything — and setting it there
                # stops the courier accepting, which stops them gathering
                # the samples that would bring it back down.
                continue
            share_accepted = k / n
            # How long until an offer THIS FUSSY turns up. A stricter bar
            # takes longer, which is the cost being weighed.
            acceptable_per_minute = arrival_rate * share_accepted
            # Committed work absorbs part of that wait: an offer arriving
            # while the courier is still riding costs them nothing to have
            # waited for. But offers arrive at RANDOM, not on a timetable —
            # so what committed work buys is the PROBABILITY of being
            # covered, not a guaranteed subtraction. With Poisson arrivals
            # the chance nothing acceptable turns up in the remaining
            # `committed_minutes` is exp(-rate x minutes), and only then
            # does the courier eat the full wait.
            #
            # (Splitting this rate into a higher "while riding" and a lower
            # "while parked" one is defensible and was tried: measured over
            # five seeds it made the policy WORSE, -8.3% against the
            # baseline versus -4.7% for this single-rate version, so the
            # simpler model stands.)
            missed = math.exp(-acceptable_per_minute * committed_minutes)
            expected_idle = missed / acceptable_per_minute
            denominator = minutes_sum / k + expected_idle
            if denominator <= 0.0:
                continue
            value = net_sum / k / denominator * _MINUTES_PER_HOUR
            best_value = max(best_value, value)
            if k == n:
                # The bar that rejects nothing: estimated from the whole
                # sample, so the robust end of this calculation and the
                # anchor the maximised value is shrunk toward.
                take_everything_value = value

        if best_value <= 0.0:
            return cold_start
        trust = n / (n + cal["optimism_shrinkage_offers"])
        return take_everything_value + (best_value - take_everything_value) * trust

    def _cold_start_threshold(self, observation: Observation, courier: CourierSnapshot) -> float:
        """The bar before the courier has seen enough offers to compute one.

        Kept as the old fixed reservation rate scaled by how much shift is
        left and how the platform is treating them: a reasonable opening
        guess, replaced by evidence within the first few offers.
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
        *,
        committed_minutes: float = 0.0,
        courier: CourierSnapshot | None = None,
    ) -> Decision:
        # Remember what was agreed to: when this job should be done, and
        # where it drops the courier. Both feed the next decision.
        self._memory.held.append(
            _HeldJob(
                order_id=chosen.offer.order_id,
                finish_estimate_min=minute + committed_minutes + chosen.paid_minutes,
                dropoff_lat=chosen.offer.dropoff_lat,
                dropoff_lon=chosen.offer.dropoff_lon,
            )
        )

        if binding == "acceptance_rate":
            summary = (
                "Accepted %s at %.0f net MXN per hour even though my bar is %.0f, because my "
                "acceptance rate is low enough that the app will start starving me of offers."
                % (chosen.offer.order_id, chosen.rate, threshold)
            )
        elif committed_minutes > 0.0:
            summary = (
                "Queued %s from %s behind the job I am on: %.0f MXN per hour against a bar of "
                "%.0f. I am committing about %.0f minutes early, which is worth it because the "
                "next offer would most likely arrive while I am still riding anyway, so taking "
                "this one costs me no waiting, and its kitchen starts cooking now."
                % (
                    chosen.offer.order_id,
                    chosen.offer.restaurant_name,
                    chosen.rate,
                    threshold,
                    committed_minutes,
                )
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
        # Stand still and see, first. Riding off before having any evidence
        # about this spot is how the branch turns into a shift-long loop —
        # see `REPOSITION_CALIBRATION` for the measured spiral. The streak
        # resets on arrival in a new cell, so this also guarantees a move
        # cannot immediately follow a move.
        if (
            self._memory.idle_streak_minutes
            < REPOSITION_CALIBRATION["min_idle_minutes_before_moving"]
        ):
            return None

        here_demand = model.demand(observation.at_cell, observation.at_lat, observation.at_lon)
        # What standing here is expected to cost. Two sources, and the
        # courier takes the worse: what they BELIEVE about local demand, and
        # what has actually happened to them. A twenty-minute empty screen
        # is evidence the flow does not reach this spot — evidence that
        # beats any belief, and the thing that turns "go stand where it
        # pays" from an opinion into a decision. Without it a courier parked
        # in a dead zone believes the wait is eight minutes forever and
        # never moves; measured, that cost one seed 82% of its shift idle
        # for three deliveries.
        wait_here = max(
            model.dead_minutes(here_demand.value, self._measured_wait_minutes(courier)),
            self._memory.idle_streak_minutes,
        )

        best_cell: str | None = None
        best_gain_minutes = 0.0
        best_km = 0.0
        best_travel_minutes = 0.0
        best_demand = 0.0

        # Every cell the courier can both NAME and PLACE: the app's own
        # heatmap cells plus every cell `Observation.cell_coords` locates.
        # Iterating `observation.demand_by_cell` against no coordinate
        # source — as an earlier revision did — meant `coords_of` returned
        # None for every fine-grid id and this whole function was dead code
        # that could never move the courier anywhere.
        for cell in index.known_cells():
            if cell == observation.at_cell:
                continue
            if cell in self._memory.unreachable:
                continue  # asked for it once, demonstrably never got there
            if cell not in observation.demand_by_cell:
                # Only somewhere the courier has an actual opinion about.
                # The app's own display cells are a several-kilometre smear
                # whose centroid is not a place: "ride to that district"
                # resolves to wherever is nearest that fiction, which can be
                # the corner the courier is already standing on — and then
                # nothing happens, the screen stays empty, and the same
                # instruction comes out again next minute. Before
                # `Observation.cell_coords` existed these smears were the
                # only spatial vocabulary the policy had. They are not any
                # more.
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
            there = model.demand(cell, coords[0], coords[1])
            # Believed BUSIER, by more than one band of the app's own
            # quantised heatmap. Without this the target is whichever cell's
            # demand noise happened to round up this minute, which at 06:00
            # — when the whole city reads "almost nothing" — is a different
            # cell every minute.
            if there.value - here_demand.value < REPOSITION_CALIBRATION["min_demand_edge"]:
                continue
            gain = wait_here - (
                leg.minutes + model.dead_minutes(there.value, self._measured_wait_minutes(courier))
            )
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

        # Remember what was asked for, so next minute the courier can tell
        # whether they actually got going — see `_begin_minute`.
        self._memory.pending_target = best_cell
        self._memory.pending_from_cell = observation.at_cell

        summary = (
            "%s, so I am riding %.1f km to %s: I believe demand there is %.2f against %.2f here, "
            "which should save about %.0f unpaid minutes for %.0f minutes of travel."
            % (
                "I have stood here %.0f minutes with an empty screen"
                % self._memory.idle_streak_minutes
                if self._memory.idle_streak_minutes >= wait_here
                else "Nothing on screen is worth %.0f MXN per hour" % threshold,
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
        *,
        committed_minutes: float = 0.0,
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
            if committed_minutes > 0.0:
                summary = (
                    "Left all %d offers: I still have about %.0f minutes of work in hand, so I can "
                    "wait for something better than average without losing a minute to it, and "
                    "the best on screen, %s, is out because %s."
                    % (len(plans), committed_minutes, best.offer.order_id, reason)
                )
            else:
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
