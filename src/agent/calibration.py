"""Calibration: every number the policies argue with, in one place.

These are CALIBRATION CONSTANTS, not discovered truths. They encode a courier's
working assumptions about a scooter in a dense metro, and a judge is entitled
to disagree with any of them. They are grouped and named so that disagreement is
a one-line edit rather than an archaeology exercise.

Kilometres and minutes never collapse into one "cost" number here either: the
travel block converts between them explicitly, and the conversion is the part
that changes when traffic hits.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The unaided courier's rule of thumb — the baseline to beat
# --------------------------------------------------------------------------
#
# Ask an experienced courier how they decide and you do not get a heuristic
# over a menu, because the app never shows them a menu. You get a floor:
# "I don't take anything under fifty pesos." One number, applied to the one
# card on screen, and it is genuinely good — it filters the loss-making tail
# without ever leaving the courier waiting for perfection.
#
# That floor is what `baseline.FixedPayoutThresholdPolicy` implements, and it
# is the honest opponent: it is what the smart policy has to beat to be worth
# anything at all. So the value below is the one that makes the BASELINE
# strongest, not the one that flatters the smart policy.
#
# MEASURED, 6 seeds x 4 windows, mean MXN/h (the sweep is reproducible with
# `FixedPayoutThresholdPolicy(floor_mxn=...)`, which exists for exactly this):
#
#   floor    0     30     40     45     50     55     60     70     85
#   ref     91.3   94.0   95.7  101.7  100.4  115.6  116.9   96.0   95.0
#   night  102.3   99.0   88.7   89.9   93.3   90.6   90.6   84.2   86.4
#   early   77.8   77.7   88.9   82.6   90.6   87.9   86.9   81.4   54.3
#   day    110.0  114.4  114.9  113.7  104.0   94.4   93.9   90.8   83.5
#   POOLED  95.3   96.3   97.0   97.0   97.1   97.1   97.1   88.1   80.9
#
# Two things that curve says out loud. First, the pooled optimum is FLAT
# from 40 to 60 — pooled evidence cannot separate those, because the right
# floor genuinely depends on the window: 60 is best on the reference shift,
# 0 (take everything) is best at night, 50 in the early morning, 40 midday.
# Second, above about 70 the baseline degrades everywhere, because a floor
# that high leaves it living permanently on the acceptance-rate rescue.
#
# 55 is the choice: inside the flat pooled optimum, and within it the
# strongest on the REFERENCE window — the shift the comparison is headlined
# on — by a wide and seed-robust margin (115.6 against 100.4 at a floor of
# 50; every seed better, not one lucky one). Picking the flat optimum's
# reference-strongest member is picking the hardest opponent available
# without overfitting to a single window.
#
# Why a payout floor and not a nearest-first rule: with one offer per minute,
# "take the nearest of what is on screen" picks the only card on screen, so a
# nearest-first baseline is arithmetically identical to accepting everything.
# It is a menu heuristic, and this app is a flow.
# CHANGED TO 40 BY THE USER. The reasoning above is kept rather than
# rewritten, because it is what makes this change legible: 55 was not a
# guess, it was the sweep's reference-strongest floor, chosen deliberately to
# make the opponent as hard as available. 40 is a different criterion --
# closer to the rule of thumb a working courier actually uses -- and it
# almost certainly makes the baseline weaker, which flatters the agent.
#
# So the comparison is no longer "against the hardest opponent we could
# build". Anyone quoting a margin against this baseline has to say which
# floor it was measured at. Both are recorded in the commit that made the
# change.
BASELINE_CALIBRATION: dict[str, float] = {
    # The floor, in gross MXN off the card. The courier does no arithmetic
    # on it: no per-hour rate, no kilometres, no clock. That is the point.
    "fixed_payout_floor_mxn": 40.0,
    # Late in the shift even a disciplined courier stops arguing: an order
    # that gets them home paid beats riding home empty. Below this many
    # minutes of shift left the floor is scaled by the factor underneath.
    # Without it the baseline would reject its way through the last hour of
    # every shift, which no real courier does — and a baseline that behaves
    # worse than the person it represents is a straw man.
    "late_shift_minutes": 45.0,
    "late_shift_floor_factor": 0.6,
    # The app shows a courier their own acceptance rate, and every courier
    # knows what happens when it sinks: the offers dry up. So the floor is
    # suspended below this rate, exactly as a real courier starts taking
    # work again once they notice the screen has gone quiet.
    #
    # This is not a concession, it is the difference between an opponent and
    # a straw man. Measured without it, on the Night window (18:00 start,
    # when payouts are at their thinnest before the dinner peak), the
    # baseline rejected its opening offers, the platform's acceptance-rate
    # retaliation cut its reach, and it never recovered: 11 offers seen in
    # an eight-hour shift, ZERO deliveries, 0.0 MXN/h on both seeds tested.
    # A baseline earning nothing proves nothing.
    "acceptance_rescue_rate": 0.40,
    # Early ratios are noise; one rejection out of one offer is not a
    # crisis. Ignore the rate until this many offers have been seen.
    "min_offers_before_rescue": 5.0,
}

# --------------------------------------------------------------------------
# Turning straight-line geometry into street travel
# --------------------------------------------------------------------------

# Kilometres and free-flow minutes for a leg are no longer derived here.
# They are PULLED from `RawSourcePort.travel_estimate`, which answers from
# two layers and nothing else: a free-flow skeleton over the OSM drive
# graph, built once offline per city, plus a correction by zone and hour
# fitted from the agent's own completed trips. The agent still owns what it
# does with that answer — the live congestion reading and the rain
# slowdown below are applied on top of it, here.
#
# What used to live here was a single effective 22 km/h over great-circle
# km times 1.35. That number was a reasonable guess about a city; the
# skeleton is the city's actual geometry, and the correction is the
# courier's own measurement of their own vehicle on it.
TRAVEL_CALIBRATION: dict[str, float] = {
    # A leg that exists at all costs at least this long (start, park, unlock).
    "min_leg_minutes": 1.0,
    # Below this, two points are treated as the same place.
    "same_place_km": 0.05,
    # The shared confidence scale for a travel-time answer, and the only
    # coupling between the agent and its sources beyond the port's types.
    # A port reporting `structural_only_confidence` is saying "this is the
    # skeleton and a prior, nothing measured"; one reporting
    # `fully_learned_confidence` is saying "my own trips have this zone and
    # hour covered". The agent reads its own live congestion multiplier at
    # full weight at the first and drops it entirely at the second —
    # because a well-evidenced learned correction has already absorbed the
    # typical congestion, and multiplying by it again counts the same jam
    # twice.
    "structural_only_confidence": 0.55,
    "fully_learned_confidence": 0.92,
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

# Believed demand per quantised in-app heatmap level (1 calm .. 4 hot). Now
# that `BeliefState.cell_coords` places every cell in
# `BeliefState.demand_by_cell`, the courier's own per-cell sense is the
# better signal and is tried first — this is the fallback for a point no
# demand belief covers (the app's heatmap reaches slightly further out than
# the operating grid). Lagged, coarse and quantised though it is, reading a
# level off the app's own map beats assuming every unknown corner of the
# city is equally busy.
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
    # --- the dry spell ---
    # How much evidence the lifetime arrival rate is worth, measured in
    # OFFERS, when a dry spell argues against it. Standing free with an
    # empty screen is not neutral: if the courier believes offers arrive
    # every eight minutes and twenty minutes pass with nothing, those
    # twenty minutes are evidence the belief is wrong. Treating the prior
    # as worth three offers and the silence as a Poisson observation of
    # zero, the posterior rate shrinks by tau / (tau + idle), where tau is
    # this many offers divided by the believed rate.
    #
    # Why it goes HERE and not into `_offers_per_minute`: that rate has two
    # consumers. One prices the wait for a better offer (the bar); the
    # other prices the dead minutes a destination will cost. A dry spell
    # where the courier is STANDING is evidence about the first and not the
    # second, and conflating them is exactly why the earlier
    # recency-windowed rate measured worse across every window. The two
    # uses had to be decoupled, and this is the decoupling.
    "dry_spell_prior_offers": 3.0,
    # Never let the bar collapse to nothing: at some point a courier is
    # accepting work that loses money, and a shrinking bar must not be
    # allowed to argue for that.
    "dry_spell_min_rate_fraction": 0.25,

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
    # Night ramps IN between these two minutes-of-day: dusk, then full dark.
    "night_starts_minute": 20 * 60,
    "night_full_minute": 23 * 60,
    # ... and ramps OUT between these two. Without them `night_factor` read
    # its ramp off minute-of-day alone, so at 00:00 the minute-of-day reset
    # to 0, fell below `night_starts_minute`, and the premium went to ZERO
    # for the darkest hours of the shift. Measured on the Night window
    # (1080-1560, 18:00-02:00) that silently exempted the last 120 of 480
    # minutes — a quarter of the shift, and the quarter a courier is most
    # wary of. A risk premium that switches itself off at midnight is not a
    # calibration choice, it is an off-by-1440.
    #
    # 05:00 for "still fully dark" and 07:00 for "fully light" bracket
    # sunrise across the year at about 25 degrees north (roughly 06:55 in
    # July, 07:15 in December). CALIBRATION VALUES, like everything else
    # here — and the two in this file that are genuinely a function of WHERE
    # rather than of people. Said out loud because it is the honest caveat:
    # at a far northern latitude these two control points are the thing to
    # re-derive, and re-deriving them needs a coordinate and a date, which
    # is arithmetic rather than a dataset. That keeps the agent portable in
    # principle; it does not make these numbers right in Oslo.
    "night_ends_minute": 5 * 60,
    "day_full_minute": 7 * 60,
    # Rain is both slower and more dangerous.
    # What a courier charges for riding INTO believed congestion, on top of
    # the time it costs. The time cost is already priced by the travel
    # model; this is the separate fact that a jam is miserable and
    # dangerous on two wheels. Scaled by `SmartPolicy.risk_posture`, which
    # is how one courier takes the jam head-on and another goes around.
    "congestion_aversion_mxn_per_km": 0.60,
    # Heat. 32 C is where a rider starts paying for it, 42 C the punishing
    # end. Ramp, not a switch: 38 C and 44 C are not the same shift. These
    # are human physiology, not a local climate table, which is why they
    # are allowed to live inside the agent.
    "heat_risk_mxn_per_km": 0.45,
    "heat_risk_onset_c": 32.0,
    "heat_risk_full_c": 42.0,
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
    # Stand still and see, before riding off. A courier who has just parked
    # has no evidence about this spot yet, and the two guards below are what
    # stop the branch from eating an entire shift.
    #
    # Both were added after a measured death spiral, and the spiral is worth
    # writing down because neither guard looks necessary until you see it.
    # When a courier is starved of offers, their measured typical wait blows
    # up to the clamp (`RESERVATION_CALIBRATION["min_offers_per_hour"]`, so
    # an hour). `ForwardModel.dead_minutes` scales every wait estimate by
    # that measurement — correctly — so the gap between a cell believed at
    # demand 0.00 and one believed at 0.25 stops being three minutes and
    # becomes twenty, which pays for a six-kilometre ride. And the sensed
    # demand map is re-drawn with fresh noise every minute, so at 06:00,
    # when the whole city reads "almost nothing", whichever cell's noise
    # draw happened to round up this minute becomes the target. The courier
    # then rides all shift, never stands still long enough to be offered
    # anything, and the starvation that started it never lifts. Measured:
    # 393 repositions in a 540-minute shift, 88% of it idle, two offers seen
    # all morning, 7.6 MXN/h where the same policy with the branch muted
    # earned 84.7.
    #
    # Minutes stood here with an empty screen before moving is even
    # considered. Direct evidence about THIS spot, and it resets on arrival,
    # so a move cannot follow a move.
    "min_idle_minutes_before_moving": 8.0,
    # And the destination has to be believed BUSIER, by a margin wider than
    # one band of the app's own quantised heatmap — not merely to have
    # rounded up this minute.
    "min_demand_edge": 0.20,
}

# --------------------------------------------------------------------------
# What the agent asks its sources for, and how far
# --------------------------------------------------------------------------
#
# A radius, not a cell list: the agent names places by coordinate because
# that is all it has in a city it has never worked. Both radii are wide
# enough to span a metro area from anywhere inside it, which is the whole
# of what one shift touches.
#
# A wide radius does not mean a wide answer. Congestion comes back as
# sparse as the courier's traffic app actually is — own cell, neighbours,
# one corridor — because the SOURCE is sparse, not because the question
# was narrow. POI density does come back everywhere in range, and should:
# it is a map, and a courier can look at any part of a map.
QUERY_CALIBRATION: dict[str, float] = {
    "congestion_radius_km": 30.0,
    "poi_radius_km": 30.0,
}

# The app's own label for a surge window, as it appears on screen. A string
# the agent reads off a disruption feed, never a type imported from the
# world — there is no import path from here to one.
SURGE_EVENT_KIND = "surge_window"

# --------------------------------------------------------------------------
# The hour-of-day rhythm the agent arrives with
# --------------------------------------------------------------------------
#
# Bimodal, because people eat lunch and then dinner. THE SHAPE transfers —
# it is a fact about people, not about any one city — and the exact peaks
# and the depth of the afternoon trough are what the agent learns for itself.
#
# Deliberately coarse and round-numbered: twelve control points a courier
# could describe out loud, linearly interpolated. It is not the simulator's
# own demand profile and is not meant to be; a prior that matched the
# world's curve exactly would be local knowledge wearing a prior's clothes.
# Minute-of-day, wrapping at midnight.
MEAL_RHYTHM_PRIOR: tuple[tuple[int, float], ...] = (
    (0, 0.05),     # 00:00 dead
    (360, 0.05),   # 06:00 still dead
    (480, 0.10),   # 08:00 breakfast, thin
    (600, 0.25),   # 10:00 climbing
    (720, 0.70),   # 12:00 lunch ramp
    (840, 1.00),   # 14:00 lunch peak
    (960, 0.70),   # 16:00 falling away
    (1080, 0.35),  # 18:00 the afternoon trough
    (1200, 0.70),  # 20:00 dinner ramp
    (1290, 1.00),  # 21:30 dinner peak
    (1380, 0.70),  # 23:00 winding down
    (1440, 0.05),  # 24:00 dead again (wraps to 0)
)

# --------------------------------------------------------------------------
# Turning POI density plus that rhythm into a demand belief
# --------------------------------------------------------------------------

DEMAND_PRIOR_CALIBRATION: dict[str, float] = {
    # Bands, not a continuous number. A courier thinks "busy / quiet /
    # dead", and a difference finer than a band is noise — which is what
    # stops the repositioning branch chasing whichever cell rounded up.
    "bands": 5.0,
    # How much the agent trusts its own hour-of-day prior. The POI count is
    # close to a fact; that this hour is busy is a guess, and the product of
    # a fact and a guess is a guess.
    "rhythm_prior_confidence": 0.80,
    # A perceived surge window is real, specific, located information about
    # right now, so it lifts the cells it touches above the generic rhythm
    # baseline and raises confidence there.
    "surge_value_bump": 0.35,
    "surge_confidence_bump": 0.25,
    "surge_age_minutes": 2.0,
}
