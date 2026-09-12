"""Calibration: every number the policies argue with, in one place.

These are CALIBRATION CONSTANTS, not discovered truths. They encode a courier's
working assumptions about a scooter in Monterrey, and a judge is entitled to
disagree with any of them. They are grouped and named so that disagreement is
a one-line edit rather than an archaeology exercise.

Kilometres and minutes never collapse into one "cost" number here either: the
travel block converts between them explicitly, and the conversion is the part
that changes when traffic hits.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Turning straight-line geometry into street travel
# --------------------------------------------------------------------------

TRAVEL_CALIBRATION: dict[str, float] = {
    # Streets are not straight lines. Multiplier from great-circle km to
    # ridden km; ~1.35 is the usual figure for a dense grid city.
    "street_detour_factor": 1.35,
    # Scooter average including stops, lights and parking hunts.
    "free_flow_speed_kmh": 22.0,
    # A leg that exists at all costs at least this long (start, park, unlock).
    "min_leg_minutes": 1.0,
    # Below this, two points are treated as the same place.
    "same_place_km": 0.05,
    # Rain slows traffic and makes the courier ride slower. Multiplier added
    # per mm of believed precipitation, capped.
    "rain_slowdown_per_mm": 0.08,
    "max_rain_multiplier": 1.45,
    # Missing traffic belief for a cell: assume free flow, but barely believe it.
    "default_traffic_multiplier": 1.0,
    "default_traffic_confidence": 0.30,
}

# --------------------------------------------------------------------------
# Time that the platform does not show on the offer card
# --------------------------------------------------------------------------

HANDLING_CALIBRATION: dict[str, float] = {
    "pickup_handling_minutes": 3.0,   # parking, queueing, confirming the order
    "dropoff_handling_minutes": 4.0,  # finding the door, waiting for the customer
    # What the courier assumes about a kitchen they have never visited.
    "default_kitchen_minutes": 9.0,
    "default_kitchen_confidence": 0.25,
}

# --------------------------------------------------------------------------
# Money. Payout is gross; these are what the courier actually keeps.
# --------------------------------------------------------------------------

ECONOMICS_CALIBRATION: dict[str, float] = {
    "fuel_cost_per_km_mxn": 1.10,
    "vehicle_wear_per_km_mxn": 0.40,
    # What a minute of the courier's time is worth when deciding whether an
    # unpaid movement is justified (~60 MXN/hour).
    "opportunity_mxn_per_minute": 1.00,
}

# --------------------------------------------------------------------------
# Where the delivery leaves you
# --------------------------------------------------------------------------

# Believed demand per quantised in-app heatmap level (1 calm .. 4 hot). The
# heatmap is the only demand signal the courier can both SEE and LOCATE: the
# per-cell demand estimates in `Observation.demand_by_cell` are keyed by a
# finer grid whose coordinates the courier only learns by standing in them.
# Lagged, coarse and quantised though it is, reading a level off the app's
# own map beats assuming every unknown corner of the city is equally busy.
HEATMAP_LEVEL_DEMAND: dict[int, float] = {1: 0.20, 2: 0.45, 3: 0.70, 4: 0.90}
# How much to believe it, given the lag and the coarseness.
HEATMAP_LEVEL_CONFIDENCE = 0.40

DESTINATION_CALIBRATION: dict[str, float] = {
    # Expected unpaid minutes before the next worthwhile offer, as a function
    # of believed demand at the destination. A dead zone is a long wait or a
    # long ride out of it; either way the minutes are yours, not the platform's.
    "dead_minutes_at_zero_demand": 14.0,
    "dead_minutes_at_full_demand": 2.0,
    # Missing demand belief: assume the middle, and say so.
    "default_demand": 0.45,
    "default_demand_confidence": 0.30,
}

# --------------------------------------------------------------------------
# End-of-shift geometry
# --------------------------------------------------------------------------

SHIFT_CALIBRATION: dict[str, float] = {
    # Above this many minutes left, the ride home is somebody else's problem.
    # An earlier value of 180 charged the ride home across the last THREE
    # hours of an eight-hour shift — over a third of it — which quietly
    # biased the policy against every order pointing away from home long
    # before going home was a real constraint. Two hours is the last quarter
    # of a shift, which is when it genuinely starts to bind.
    "homeward_ignored_above_minutes": 120.0,
    # At or below this, every extra minute away from home is fully charged.
    "homeward_full_weight_below_minutes": 25.0,
    # Reservation rate: the MXN/hour below which the courier would rather
    # wait. Used ONLY as the cold-start bar, for the first handful of offers
    # of a shift, before the courier has scored enough of them to compute
    # the real one (see RESERVATION_CALIBRATION and `SmartPolicy._threshold`).
    #
    # 20, not the 55 an earlier revision used, and the difference is a units
    # correction rather than a softening. The bar is compared against a
    # CONFIDENCE-DISCOUNTED rate over the job's FULL time cost including the
    # unpaid minutes after the drop — a number that runs roughly half the
    # gross MXN/hour a courier would quote you. Measured on the reference
    # shift, offers scored 30-35 on that scale against a 63 bar, so the
    # policy rejected every offer of the entire shift and earned nothing.
    # A cold-start bar must be clearable by an ordinary offer, because its
    # only job is to stop the courier taking outright loss-making work while
    # they gather the evidence for a real bar.
    "base_reservation_mxn_per_hour": 20.0,
    # The cold-start bar is scaled by how much shift is left: choosy early,
    # pragmatic late.
    "choosy_factor_early": 1.15,
    "desperate_factor_late": 0.60,
    # Safety margin kept free at the end of the shift for the ride home.
    "home_margin_minutes": 3.0,
}

# --------------------------------------------------------------------------
# The reservation price: what rejecting actually costs
# --------------------------------------------------------------------------
#
# Rejecting an offer is not free and it is not a fixed cost either. It buys
# the courier the chance of a better offer, and charges them the unpaid
# minutes until one arrives. At a busy lunchtime with an offer every three
# minutes a courier can afford to be fussy; at 4 offers an hour the same
# fussiness is just unpaid waiting. The bar therefore has to be COMPUTED,
# from two things the courier genuinely knows about their own shift:
#
#   - how often offers arrive (`CourierSnapshot.offers_seen` against
#     `minutes_elapsed` — their own count, not the platform's);
#   - what an offer is typically worth (the running mean of what this policy
#     itself has scored so far this shift).
#
# Accept when this offer's rate beats what waiting is worth:
#
#     value(b) = mean_net(rate >= b) / (mean_minutes(rate >= b) + wait(b)) x 60
#     wait(b)  = 1 / (arrival_rate x P(rate >= b))
#
# and the bar is the value of the BEST b available — the reservation price
# is the continuation value itself, and the rule is "accept iff this offer
# beats what holding out is worth". A fussier bar raises the numerator and
# the wait together; the maximisation is what balances them, over the
# empirical distribution of offers this courier has actually been shown.
#
# Two things the previous fixed constant got wrong, both measured:
#
#   - UNITS. It compared a confidence-DISCOUNTED rate against an
#     UNDISCOUNTED constant, so a policy appropriately unsure about a kitchen
#     it had never visited rejected the offer for being uncertain rather than
#     for being bad. On the reference shift that rejected 100% of offers for
#     a whole shift and earned nothing at all.
#   - SCARCITY. A constant cannot know that at 4 offers an hour, holding out
#     for a better one costs a quarter of an hour of unpaid waiting.
#
# `wait(b)` is also where committing early enters. A courier already
# committed to `R` more minutes of work does not pay that wait — the offer
# arrives while they are still riding — so their bar for QUEUEING a job
# ahead uses `max(0, wait(b) - R)`. With a long job still to run that
# tightens toward "better than average or leave it"; as the job nears its
# end it relaxes back to the idle bar, because an empty screen at the moment
# they go free is real unpaid time. That is the whole commit-early tradeoff,
# in one term.
# --------------------------------------------------------------------------
RESERVATION_CALIBRATION: dict[str, float] = {
    # How many recent offers the value distribution is built from. The flow
    # at 21:00 is not the flow at 14:00, so a bar built from the whole shift
    # would keep arguing with a lunchtime that is over.
    "recent_offer_window": 60.0,
    # Prior belief about the offer flow before the shift has produced
    # evidence: roughly 7.5 offers an hour, held with the weight of about
    # half an hour of observation. Blended with the courier's own observed
    # count so the early minutes are not ruled by a sample of one.
    "prior_offers_per_hour": 7.5,
    "prior_weight_minutes": 30.0,
    # Never believe offers arrive faster than this, nor slower: both ends
    # guard against a degenerate early ratio producing a nonsense bar.
    "min_offers_per_hour": 1.0,
    "max_offers_per_hour": 30.0,
    # Offers scored before the empirical distribution is trusted at all.
    "min_offers_for_running_mean": 6.0,
    # A candidate bar must be one at least this many observed offers would
    # have cleared. Without it the maximisation happily sets the bar at the
    # single best offer ever seen — a sample of one — and the courier then
    # rejects everything forever, never gathers another sample, and the bar
    # never comes back down. Measured: that death spiral took a working
    # policy to zero deliveries on a whole shift.
    "min_accepted_sample": 5.0,
    # Shrinkage against the optimiser's curse, in units of "offers seen".
    #
    # The bar is a MAXIMUM over candidate bars, each scored from a small
    # sample. Taking the max of noisy estimates systematically overstates
    # the true best — the candidate that wins is disproportionately likely
    # to be the one whose sample was luckiest — so a courier who believed it
    # would hold out for a rate the flow does not actually contain, and a
    # courier who holds out earns nothing while they do it.
    #
    # The correction shrinks the maximised value toward the value of simply
    # ACCEPTING EVERYTHING, which is estimated from the whole sample and so
    # is the robust end of the same calculation. Weight `n / (n + this)`:
    # with a handful of offers the courier mostly trusts the robust number,
    # and earns the right to be fussy as the evidence accumulates.
    "optimism_shrinkage_offers": 25.0,
}

# --------------------------------------------------------------------------
# Acceptance rate: the platform punishes choosiness, so the policy must too
# --------------------------------------------------------------------------

ACCEPTANCE_CALIBRATION: dict[str, float] = {
    # Above this the courier is in good standing and can afford to skip.
    "comfortable_rate": 0.65,
    # At or below this the account is in trouble; the threshold collapses.
    "distressed_rate": 0.25,
    "comfortable_threshold_factor": 1.05,
    "distressed_threshold_factor": 0.55,
    # Below this, take anything that makes money: a deactivated courier earns
    # zero per hour, which beats every clever rejection.
    "hard_floor_rate": 0.40,
    # Ignore the rate until this many offers have been seen; early ratios are noise.
    "min_offers_for_signal": 5.0,
}

# --------------------------------------------------------------------------
# Fuel: minutes, never pesos
# --------------------------------------------------------------------------

FUEL_CALIBRATION: dict[str, float] = {
    # Plan the stop at this much range left rather than being ambushed by it.
    "reserve_minutes": 25.0,
    # Range kept spare on top of a job's own duration before accepting it.
    "job_margin_minutes": 10.0,
    # What a refuelling stop costs: detour, queue, pump, nothing earned.
    "stop_minutes": 12.0,
    # Refuelling with less shift left than this is throwing minutes away.
    "min_useful_shift_minutes": 15.0,
}

# --------------------------------------------------------------------------
# Uncertainty and safety
# --------------------------------------------------------------------------

BELIEF_CALIBRATION: dict[str, float] = {
    # How much each belief contributes to the confidence behind a score.
    "weight_traffic": 0.35,
    "weight_demand": 0.25,
    "weight_kitchen": 0.40,
    # How hard a shaky belief is discounted. 0 would treat a guess as a fact.
    "uncertainty_penalty": 0.45,
    # An estimate this old is worth roughly nothing extra in confidence terms.
    "staleness_horizon_minutes": 60.0,
    "min_staleness_factor": 0.40,
}

SAFETY_CALIBRATION: dict[str, float] = {
    # Risk premium the courier charges for night kilometres, in MXN per km.
    "night_risk_mxn_per_km": 0.35,
    # Night ramps in between these two minutes-of-day.
    "night_starts_minute": 20 * 60,
    "night_full_minute": 23 * 60,
    # Rain is both slower and more dangerous.
    "rain_risk_mxn_per_km": 0.25,
    "rain_risk_full_mm": 4.0,
}

REPOSITION_CALIBRATION: dict[str, float] = {
    # Moving must save at least this many expected unpaid minutes to be worth it.
    "min_gain_minutes": 3.0,
    # And must still be worth it after the kilometres are paid for.
    "min_gain_mxn": 1.0,
    # Never ride further than this speculatively.
    "max_reposition_km": 6.0,
}
