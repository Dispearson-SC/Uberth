"""Calibration knobs for the courier's own external tools.

Every dict in this module is a hand-tuned CALIBRATION VALUE, not measured
data — there is no public dataset describing how wrong a Monterrey courier's
weather app, traffic app, or "feel for the day" actually is. Values are
chosen to be *plausible* (a weather app is quite reliable; a traffic app on a
far-away cell is not; a courier's memory of a kitchen sharpens with repeat
visits) and are kept here, in one auditable place, exactly like the
calibration dicts in `src/world/weather.py`, `traffic.py` and `events.py`.

Nothing in this module is a measured quantity. Say so out loud whenever
presenting numbers derived from it.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Weather: a real forecast app, so it is close but not exact. Precipitation
# is inherently patchier/harder to nowcast than temperature, so it is
# deliberately noisier and less confident.
# --------------------------------------------------------------------------

WEATHER_NOISE_CALIBRATION: dict[str, float] = {
    "temp_c_std": 0.4,
    "temp_c_confidence": 0.95,
    "apparent_c_std": 0.6,
    "apparent_c_confidence": 0.90,
    # Precipitation noise scales with the reading itself (a heavier rain
    # reading is harder to pin to the exact millimetre) plus a small floor
    # so a dry reading is not reported with zero uncertainty.
    "precip_mm_std_fraction": 0.35,
    "precip_mm_std_floor": 0.03,
    "precip_mm_confidence": 0.65,
    # A weather app is not instantaneous: this is its typical refresh lag.
    "age_minutes": 6.0,
}

# --------------------------------------------------------------------------
# Traffic: sparse by construction. A courier's traffic app only meaningfully
# covers their own cell, its immediate neighbours, and a plausible corridor
# toward wherever the day's activity centres on (modelled here as the
# downtown landmark `src.world.events.DEMO_DOWNTOWN_LAT/LON`, a reasonable
# stand-in for "the direction the app bothers to show detail in"). Error and
# staleness both grow with distance from the courier: nobody's app shows
# real-time detail three kilometres away.
# --------------------------------------------------------------------------

TRAFFIC_NOISE_CALIBRATION: dict[str, float] = {
    "own_cell_ring": 0,
    "neighbor_ring_k": 1,
    "corridor_sample_points": 5,
    # Multiplier error std, as a fraction of the true multiplier, at zero
    # distance from the courier (own cell) — never exactly zero: even your
    # own cell's reading has some app-side rounding/measurement noise.
    "base_error_std_fraction": 0.04,
    "error_growth_per_km": 0.045,
    "min_error_std_fraction": 0.03,
    "base_confidence": 0.88,
    "confidence_decay_per_km": 0.07,
    "min_confidence": 0.20,
    # A traffic reading is never instantaneous, and gets staler the further
    # away it is (the app polls distant corridors less often than your own
    # block).
    "own_cell_age_minutes": 1.5,
    "age_minutes_per_km": 1.6,
}

# --------------------------------------------------------------------------
# Events: the honesty-critical layer. All timing/visibility filtering comes
# from `src.world.events.perceivable_events` — nothing here decides whether
# an event is visible. This calibration only controls how a *visible*
# event's effect is translated into a delay-minutes estimate the agent can
# reason with, plus a small amount of courier uncertainty about that
# translation.
# --------------------------------------------------------------------------

EVENT_DELAY_CALIBRATION: dict[str, float] = {
    # Assumed length/speed of the route segment a localized event (crash,
    # slowdown) plausibly touches, used only to convert a speed multiplier
    # into a delay-minutes figure. Not a real routed leg — a rough,
    # documented stand-in.
    "typical_affected_leg_km": 1.5,
    "assumed_free_flow_kmh": 22.0,
    # A closed street forces a real reroute, not just a slower crawl — a
    # flat, larger delay stands in for "go the long way around".
    "closure_detour_minutes": 9.0,
    # The courier's own read of "how much will this cost me" is imprecise
    # even once the event itself is confirmed.
    "value_noise_std_fraction": 0.18,
}

# --------------------------------------------------------------------------
# Kitchen memory: earned, not given. Confidence tightens with repeat visits;
# there is deliberately no cold-start entry for a restaurant never visited
# (see `kitchen_memory.py`) — an unvisited kitchen is simply absent from the
# map, never guessed at.
# --------------------------------------------------------------------------

KITCHEN_MEMORY_CALIBRATION: dict[str, float] = {
    # Confidence after n visits: 1 - (1 - first_visit_confidence) *
    # decay_per_visit ** (n - 1), asymptoting toward 1.0 but never reaching
    # it exactly.
    "first_visit_confidence": 0.35,
    "decay_per_visit": 0.55,
    "max_confidence": 0.97,
}

# --------------------------------------------------------------------------
# Demand sense: the app's own lagged, coarse, quantised heatmap plus the
# courier's feel for the day's rhythm. Deliberately built from restaurant
# density (`fixtures/restaurants.parquet`) and the public-shape temporal
# profile in `src.world.demand`, never from the ground-truth surge field —
# see `demand_sense.py` for why that keeps this layer honest by
# construction rather than by discipline.
# --------------------------------------------------------------------------

DEMAND_SENSE_CALIBRATION: dict[str, float] = {
    # Number of discrete buckets the sensed intensity is quantised into
    # (like a heatmap with a handful of colour bands, not a continuous
    # number).
    "quantise_buckets": 5,
    # The heatmap is not live: it reflects the rhythm a few minutes back.
    "lag_minutes": 12.0,
    "age_minutes": 12.0,
    "base_confidence": 0.55,
    "confidence_noise_std": 0.05,
    "value_noise_std": 0.04,
    # A perceived SURGE_WINDOW event is real, specific, corroborating
    # information — it bumps the affected cells' sensed level up and raises
    # confidence/freshness there, rather than leaving them at the generic
    # rhythm-only baseline.
    "surge_event_value_bump": 0.35,
    "surge_event_confidence_bump": 0.25,
    "surge_event_age_minutes": 2.0,
}

# --------------------------------------------------------------------------
# POI density: the portable half of "will this drop-off strand me?"
#
# `RawSourcePort.poi_density_near` answers how much food commerce sits in a
# cell. That is a fact about a map, not a live reading, so it carries no
# staleness — but it is a PROXY for where orders come from rather than a
# measurement of them, which is what the confidence below is about.
#
# The noise is the coarseness of the count itself (a POI table is never a
# perfect census of what is actually open and delivering today), and it is
# drawn with the same std and in the same per-cell order as the sensed
# demand it replaces, so migrating from the push-based `Observation` did not
# silently re-roll every other estimate's noise as a side effect.
# --------------------------------------------------------------------------

POI_DENSITY_CALIBRATION: dict[str, float] = {
    "value_noise_std": 0.04,
    "confidence_noise_std": 0.05,
    "base_confidence": 0.55,
    # A map is not a stale reading. Staleness belongs to traffic and
    # weather, which are conditions; this is geography.
    "age_minutes": 0.0,
}

# --------------------------------------------------------------------------
# The travel skeleton the agent bootstraps for itself (see `osm_travel.py`)
# --------------------------------------------------------------------------

TRAVEL_SKELETON_CALIBRATION: dict[str, float] = {
    # Buffer on the hull of the POI coordinates, so a drop-off just past
    # the last restaurant still has street context to route through.
    "operating_buffer_km": 2.0,
    # A corridor speed read off the matrix is clamped to a plausible band:
    # a degenerate pair (two cells a hundred metres apart, or a motorway
    # ramp counted end to end) must not turn into a 200 km/h belief.
    "min_corridor_speed_kmh": 12.0,
    "max_corridor_speed_kmh": 70.0,
}

# --------------------------------------------------------------------------
# Turning the skeleton into a travel-time estimate
#
# The matrix's free-flow times are CAR free-flow times off the OSM
# `highway` class. A scooter doing deliveries is a different vehicle in a
# different job: lights, junctions, parking hunts and the walk to a door.
# `free_flow_to_scooter_factor` is the structural gap between the two, and
# it is a calibration value, not a measurement.
#
# It is set so the two-layer model's COLD-START door-to-door speed matches
# the single effective 22 km/h the agent's previous single-layer model
# assumed (`src.agent.calibration.TRAVEL_CALIBRATION`). That is deliberate:
# inverting perception from push to pull must not silently re-tune what the
# agent believes about travel, or the before/after comparison would measure
# two changes at once and attribute both to the inversion.
#
# MEASURED over the 3,998 real order legs of the reference shift: mean
# corridor free-flow speed 47.46 km/h, median 46.81. So 47.46 / 22.0 =
# 2.157. The factor is large because the free-flow table is a CAR table
# with 90 km/h motorways in it, and a courier on a scooter is neither doing
# 90 nor spending a whole leg on a motorway: they are stopping at lights,
# crossing junctions, hunting for a parking spot and walking to a door. One
# factor preserves the mean while keeping the per-corridor spread, which is
# the structural signal the skeleton exists to carry — a fast-corridor leg
# still prices out faster than a residential-grid leg of the same length.
#
# WHY THE SKELETON CONTRIBUTES RATIOS, NOT ABSOLUTE KILOMETRES, and this
# was measured the hard way. The matrix is CELL-resolution: it answers
# centroid to centroid. Used as an absolute distance it is badly wrong at
# short range, where most of this job happens — over those same real legs,
# taking matrix kilometres directly inflated the agent's belief about a
# sub-2 km trip by 37% in distance and 59% in time (n=1,819, 45% of all
# legs) while leaving legs over 5 km almost untouched. An agent that
# believes short trips cost half again what they really do stops taking
# them, and short trips are exactly where this policy's edge lives:
# measured, it drove 78 km instead of 102 and earned 96.4 MXN/h instead of
# 112.3.
#
# So what is read off the matrix is two SCALE-FREE properties of the
# corridor — how much longer the road is than the crow flies, and how fast
# that road is — and both are applied to the ACTUAL endpoints. That keeps
# the structural information (a circuitous corridor still prices as
# circuitous) without importing the matrix's resolution error. MEASURED
# route factor over the same legs: mean 1.449, median 1.391, p5 1.17, p95
# 1.92, against the flat 1.35 the previous model assumed everywhere.
#
# The learned correction is what replaces this whole prior with evidence.
# On shift one there is none, so the agent rides on structure alone and
# knows it; by shift three its own trips have measured the gap per zone
# and hour.
# --------------------------------------------------------------------------

TRAVEL_SOURCE_CALIBRATION: dict[str, float] = {
    "free_flow_to_scooter_factor": 2.157,
    # Streets are not straight lines, and the matrix knows by how much per
    # corridor. This flat figure is only the fallback for a pair the matrix
    # cannot answer for — an unroutable pair, or two points in the same
    # cell, where there is no centroid-to-centroid baseline to take a ratio
    # against. ~1.35 is the usual figure for a dense grid city.
    "street_detour_factor": 1.35,
    # A corridor's route factor is clamped: a pair of adjacent centroids
    # joined by one looping road must not become a belief that every trip
    # there is three times its straight line.
    "min_route_factor": 1.05,
    "max_route_factor": 2.50,
    # Centroids closer together than this give no usable ratio.
    "min_corridor_km": 0.20,
    # Door-to-door speed used when the matrix has no route for a pair at
    # all. Not a free-flow figure: this one is already the scooter's.
    "fallback_scooter_kmh": 22.0,
    # Below this, two points are the same place.
    "same_place_km": 0.05,
    # A leg that exists at all costs at least this long.
    "min_leg_minutes": 1.0,
    # How much to believe the geometry (solid: it is a map) against the
    # time (a free-flow skeleton with a structural prior on top).
    "km_confidence": 0.85,
    "structural_minutes_confidence": 0.55,
}

# --------------------------------------------------------------------------
# What the agent learns from its own completed trips (see `history.py`)
#
# Both tables below are keyed ONLY on things the agent can observe — a cell
# id it was told about and the hour on its own clock — and both carry their
# own sample count. A zone visited twice must not speak with the authority
# of one visited two hundred times, which is what makes a cold start safe
# rather than reckless.
# --------------------------------------------------------------------------

HISTORY_FIT_CALIBRATION: dict[str, float] = {
    # Shrinkage toward the city-wide pooled ratio, in units of "trips".
    # A (zone, hour) cell with 5 trips behind it is believed half on its own
    # evidence and half on the city's.
    "shrinkage_trips": 5.0,
    # Ratios outside this band are a recording bug or a freak trip, not
    # evidence about a zone.
    "min_ratio": 0.5,
    "max_ratio": 4.0,
    # Confidence of a fitted correction as its sample grows: base at one
    # trip, asymptoting toward max.
    "first_trip_confidence": 0.45,
    "max_confidence": 0.92,
    "confidence_trips": 8.0,
    # Hours are bucketed so a zone's evening and its lunchtime are not
    # averaged together, but not so finely that nothing ever has a sample.
    "hour_bucket_hours": 3.0,
}
