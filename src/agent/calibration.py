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
    "homeward_ignored_above_minutes": 180.0,
    # At or below this, every extra minute away from home is fully charged.
    "homeward_full_weight_below_minutes": 30.0,
    # Reservation rate: the MXN/hour below which the courier would rather wait.
    "base_reservation_mxn_per_hour": 55.0,
    # The reservation rate is scaled by how much shift is left: choosy early,
    # pragmatic late.
    "choosy_factor_early": 1.15,
    "desperate_factor_late": 0.60,
    # Safety margin kept free at the end of the shift for the ride home.
    "home_margin_minutes": 3.0,
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
