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
