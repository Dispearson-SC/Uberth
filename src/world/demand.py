"""Ground truth: the order-arrival stream is structured, not uniform noise.

This module is ground truth. `src/agent/` must never import it directly —
a courier only ever sees the offers the platform layer actually routes to
their screen, never the full city-wide stream or `surge_at_spawn`.

Order generation is a non-homogeneous Poisson process, per H3 cell per
minute:

    lambda(cell, t) = sum(restaurant weights in cell)
                      * temporal_profile(t, day_type)
                      * weather_mult(t) * event_mult(t)

`sum(restaurant weights in cell)` comes straight from the real DENUE
establishment fixture (`fixtures/restaurants.parquet`, column `weight`,
already sublinear in staff size and scaled by a SCIAN delivery-likelihood
factor — used as-is here, never recomputed). `temporal_profile` is an
explicit, auditable bimodal lunch/dinner table (see
`WEEKDAY_TEMPORAL_PROFILE` / `WEEKEND_TEMPORAL_PROFILE` below).
`weather_mult` and `event_mult` are optional per-minute multiplier
sequences supplied by the caller — this module never imports `weather.py`
or `events.py`, it only accepts their effect as a plain number.

Destinations are sampled with a gravity model: population-weighted, decaying
with great-circle distance from the exact restaurant coordinates (not the
cell centroid — H3 res-7 cells are ~1.2 km across, so restaurant-level
precision matters for the destination gravity calculation even though it
doesn't change which cell the restaurant itself is in).

Surge is not computed here: this module hands the raw per-cell-per-minute
demand intensity to `surge.py`, which returns the ground-truth
`surge_at_spawn` this module stamps on every order. `demand.py` never
guesses at surge itself.

Every constant below (temporal profile control points, the fare model
coefficients, the gravity decay distance, prep-time and tip distributions)
is an explicit calibration knob, not measured data. In particular: the
exact Uber Eats / DiDi Food fare coefficients are not public. `base_mxn`,
`per_km_mxn` and `per_min_mxn` here are tuned so a whole shift's earnings
land in the range real Monterrey couriers report — they are never platform-
published figures, and must never be presented as such.

Determinism: only the named generators from `Scenario.rng_streams()` are
used — `orders` (arrival counts, restaurant/destination sampling, jitter,
tip), and `kitchen` (per-restaurant prep-time profile and per-order prep
noise). Never a global RNG, never the stdlib `random` module, never
`np.random.seed`.
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.world import geo, surge as surge_mod
from src.world.scenario import rng_streams
from src.world.timeline import OrderOffer

PROJECT_ROOT = geo.PROJECT_ROOT
FIXTURES_DIR = geo.FIXTURES_DIR
RESTAURANTS_FIXTURE_PATH = FIXTURES_DIR / "restaurants.parquet"
POPULATION_FIXTURE_PATH = FIXTURES_DIR / "population.parquet"

# --------------------------------------------------------------------------
# Temporal profile: explicit, auditable bimodal lunch/dinner tables.
# Each entry is (minute_of_day, multiplier); linearly interpolated between
# points and clamped at the ends. CALIBRATION VALUES, not measured data —
# shaped from the brief's stated lunch (12:30-15:30) / dinner (19:30-23:00)
# windows, not fit to any observed order log.
# --------------------------------------------------------------------------

WEEKDAY_TEMPORAL_PROFILE: list[tuple[int, float]] = [
    (0, 0.05),
    (360, 0.04),  # 06:00 pre-dawn quiet
    (420, 0.10),  # 07:00 light breakfast/commute pickup
    (600, 0.22),  # 10:00 rising toward lunch
    (690, 0.55),  # 11:30 fast ramp into lunch peak
    (750, 0.95),  # 12:30 lunch peak starts
    (840, 1.00),  # 14:00 lunch peak
    (930, 0.85),  # 15:30 lunch peak ends, decline starts
    (1000, 0.40),  # 16:40 afternoon lull
    (1080, 0.30),  # 18:00 low point before dinner ramp
    (1170, 0.55),  # 19:30 dinner peak starts
    (1230, 0.90),  # 20:30 rising
    (1290, 1.00),  # 21:30 dinner peak
    (1380, 0.85),  # 23:00 dinner peak ends
    (1410, 0.35),  # 23:30 winding down
    (1440, 0.05),  # 24:00 midnight (wraps back to 0)
]

WEEKEND_TEMPORAL_PROFILE: list[tuple[int, float]] = [
    (0, 0.12),
    (360, 0.06),  # 06:00
    (480, 0.10),  # 08:00
    (600, 0.25),  # 10:00
    (720, 0.55),  # 12:00 brunch ramp
    (780, 0.85),  # 13:00
    (870, 1.00),  # 14:30 weekend lunch peak (later/flatter than weekday)
    (960, 0.90),  # 16:00
    (1050, 0.55),  # 17:30 pre-dinner lull
    (1140, 0.60),  # 19:00 dinner ramp starts
    (1200, 0.85),  # 20:00
    (1260, 1.00),  # 21:00 dinner peak
    (1380, 1.00),  # 23:00 weekend nights run later, stay at peak
    (1410, 0.75),  # 23:30
    (1440, 0.15),  # 24:00 midnight
]

WEEKEND_DAYS: frozenset[str] = frozenset({"Saturday", "Sunday"})


def day_type_for(day_of_week: str) -> str:
    """Classify a day-of-week name into 'weekday' or 'weekend'."""
    return "weekend" if day_of_week in WEEKEND_DAYS else "weekday"


def temporal_profile(minute_of_day: float, day_type: str) -> float:
    """Bimodal lunch/dinner multiplier at one minute-of-day, linearly
    interpolated from the explicit control-point table above."""
    table = WEEKEND_TEMPORAL_PROFILE if day_type == "weekend" else WEEKDAY_TEMPORAL_PROFILE
    xs = np.array([p[0] for p in table], dtype=float)
    ys = np.array([p[1] for p in table], dtype=float)
    return float(np.interp(minute_of_day, xs, ys))


# --------------------------------------------------------------------------
# Calibration knobs (all explicit, all tunable, none of this is measured)
# --------------------------------------------------------------------------

DEMAND_CALIBRATION: dict[str, float] = {
    # Converts summed restaurant `weight` in a cell into an expected
    # orders-per-minute rate at temporal-profile peak (multiplier == 1.0).
    # Tuned so a Friday 8h shift produces a plausible city-wide order count
    # for a hackathon demo (~26k orders across 2.3M people), not a measured
    # conversion factor. Kept in a fixed ratio with
    # `SUPPLY_CALIBRATION["total_couriers"]` in surge.py: both were scaled
    # up together 10x from an earlier low-volume calibration, because at
    # low volume the supply field had too little mass per cell to behave
    # as a continuum (see surge.py's `eps` comment) — scaling both together
    # keeps the demand/supply ratio, and therefore the surge mechanism's
    # shape, unchanged while smoothing out that small-number artefact.
    "orders_per_weight_unit_per_min": 0.0028,
    # Gravity-model decay distance (km): most food delivery in Monterrey is
    # 1-5 km, so destinations should decay fast beyond ~2-3 km.
    "gravity_d0_km": 1.4,
    # Destination point is jittered within its cell (uniform over a disk of
    # this radius) so deliveries don't all land on cell centroids.
    "jitter_radius_km": 0.35,
}

FARE_CALIBRATION: dict[str, float] = {
    # NOT platform-published figures. Uber Eats / DiDi Food coefficients are
    # not public; these are tuned so a typical short trip pays ~35-70 MXN
    # and a whole shift lands in the range real Monterrey couriers report.
    "base_mxn": 18.0,
    "per_km_mxn": 6.5,
    "per_min_mxn": 1.2,
    # Reference speed used only to derive the straight-line `ref_minutes`
    # figure from `ref_km` — an assumed average incl. traffic/stops, not a
    # real routed ETA (that belongs to `network.py` / the sim engine).
    "ref_speed_kmh": 16.0,
}

PREP_TIME_CALIBRATION: dict[str, float] = {
    # Each restaurant's mean prep time is drawn once (uniformly in this
    # range) and reused for the rest of the shift — that persistence is
    # what makes "this kitchen is reliably slow" a learnable strategy.
    "restaurant_mean_min": 8.0,
    "restaurant_mean_max": 28.0,
    # Per-order noise around that restaurant's own mean, as a coefficient
    # of variation.
    "per_order_cv": 0.25,
    "min_minutes": 3.0,
    "max_minutes": 60.0,
}

TIP_CALIBRATION: dict[str, float] = {
    # Tip modelled as a percentage of gross payout, correlated with a proxy
    # for adverse conditions (the caller's `weather_mult` deviating from
    # 1.0 — this module cannot see actual rain, only its demand effect, so
    # it is used here as the only available conditions signal).
    "base_pct_mean": 0.10,
    "base_pct_std": 0.04,
    "conditions_pct_gain": 0.05,
    "min_pct": 0.0,
    "max_pct": 0.35,
}

# Baseline supply allocation used only to seed where competing couriers
# start the shift (see surge.py) — population is a reasonable proxy for
# "where idle couriers wait", not a measured courier census.
SUPPLY_SEED_CALIBRATION: dict[str, float] = {
    "min_baseline_weight": 1.0,
}


# --------------------------------------------------------------------------
# Fixture loading
# --------------------------------------------------------------------------


def _load_restaurants(path: Path = RESTAURANTS_FIXTURE_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("denue_id").reset_index(drop=True)


def _load_population(path: Path = POPULATION_FIXTURE_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("cell").reset_index(drop=True)


# --------------------------------------------------------------------------
# Precomputed spatial structures
# --------------------------------------------------------------------------


def _haversine_km_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance (km) between every (lat1, lon1) row
    and every (lat2, lon2) row. Same formula as `geo.great_circle_km`,
    broadcast for a full (len(lat1), len(lat2)) matrix."""
    phi1 = np.radians(lat1)[:, None]
    phi2 = np.radians(lat2)[None, :]
    dphi = np.radians(lat2[None, :] - lat1[:, None])
    dlambda = np.radians(lon2[None, :] - lon1[:, None])
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * geo.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _offset_point_km(lat: float, lon: float, distance_km: float, bearing_rad: float) -> tuple[float, float]:
    """Offset a lat/lon point by `distance_km` along `bearing_rad`, using a
    flat-earth approximation appropriate for sub-kilometer jitter."""
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(np.cos(np.radians(lat)), 0.1)
    dlat = (distance_km * np.cos(bearing_rad)) / km_per_deg_lat
    dlon = (distance_km * np.sin(bearing_rad)) / km_per_deg_lon
    return lat + dlat, lon + dlon


class _DemandModel:
    """Precomputed, read-only spatial/statistical structures shared by every
    order draw in one `build_order_stream` call. Deterministic given the
    same fixtures and scenario seed."""

    def __init__(self, restaurants: pd.DataFrame, population: pd.DataFrame, kitchen_rng: np.random.Generator):
        self.restaurants = restaurants
        n = len(restaurants)

        # Per-restaurant persistent prep-time profile, drawn once in
        # denue_id-sorted order (restaurants is already sorted by denue_id)
        # from the `kitchen` stream. This is what makes "this kitchen is
        # slow" a stable, learnable property instead of per-order noise.
        low = PREP_TIME_CALIBRATION["restaurant_mean_min"]
        high = PREP_TIME_CALIBRATION["restaurant_mean_max"]
        self.restaurant_prep_mean = kitchen_rng.uniform(low, high, size=n)

        # Per-cell restaurant indices + normalized selection probability
        # (proportional to the DENUE-derived `weight` column, used as-is).
        self.cell_restaurant_idx: dict[str, np.ndarray] = {}
        self.cell_restaurant_probs: dict[str, np.ndarray] = {}
        self.cell_weight_sum: dict[str, float] = {}
        weights = restaurants["weight"].to_numpy(dtype=float)
        for cell, group in restaurants.groupby("cell"):
            idx = group.index.to_numpy()
            w = weights[idx]
            self.cell_restaurant_idx[cell] = idx
            self.cell_restaurant_probs[cell] = w / w.sum()
            self.cell_weight_sum[cell] = float(w.sum())

        self.origin_cells: list[str] = sorted(self.cell_weight_sum.keys())

        # Gravity destination model, precomputed per restaurant (not per
        # cell) so the decay uses the restaurant's exact coordinates.
        self.population = population
        self.pop_cells: list[str] = population["cell"].tolist()
        pop_values = population["population"].to_numpy(dtype=float)
        pop_centroids = [geo.cell_centroid(c) for c in self.pop_cells]
        pop_lat = np.array([c[0] for c in pop_centroids])
        pop_lon = np.array([c[1] for c in pop_centroids])

        r_lat = restaurants["lat"].to_numpy(dtype=float)
        r_lon = restaurants["lon"].to_numpy(dtype=float)
        dist = _haversine_km_matrix(r_lat, r_lon, pop_lat, pop_lon)  # (n_restaurants, n_pop_cells)
        gravity_weight = pop_values[None, :] * np.exp(-dist / DEMAND_CALIBRATION["gravity_d0_km"])
        row_sums = gravity_weight.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        self.dest_probs = gravity_weight / row_sums  # (n_restaurants, n_pop_cells)

        # Cell grid tracked by the surge field: union of restaurant cells
        # and population cells, deterministic order.
        self.full_cell_grid: list[str] = sorted(set(self.origin_cells) | set(self.pop_cells))

        pop_map = dict(zip(self.pop_cells, pop_values))
        floor = SUPPLY_SEED_CALIBRATION["min_baseline_weight"]
        self.baseline_weights: dict[str, float] = {c: max(pop_map.get(c, 0.0), floor) for c in self.full_cell_grid}


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def build_order_stream(
    scenario_seed: int,
    date: Date,
    shift_start_min: int,
    shift_end_min: int,
    day_of_week: str = "Friday",
    weather_mult: Sequence[float] | None = None,
    event_mult: Sequence[float] | None = None,
    courier_supply_mult: Sequence[float] | None = None,
    restaurants_path: Path = RESTAURANTS_FIXTURE_PATH,
    population_path: Path = POPULATION_FIXTURE_PATH,
) -> list[OrderOffer]:
    """Generate the full ground-truth order stream for one shift.

    `weather_mult` and `event_mult` are optional per-minute sequences
    (length `shift_end_min - shift_start_min`), each defaulting to all-ones.
    They are accepted as plain numbers only — this module never imports
    `weather.py` or `events.py`. `courier_supply_mult` is forwarded to
    `surge.py` unchanged (see its docstring).

    Deterministic: identical `scenario_seed` (and identical fixtures/
    calibration) always produces byte-identical output. Uses only the
    `orders` and `kitchen` streams from `Scenario.rng_streams()` directly,
    plus `competitors` (via `surge.py`) for the supply field.
    """
    minutes = list(range(shift_start_min, shift_end_min))
    n_min = len(minutes)
    if n_min <= 0:
        return []

    weather_arr = np.asarray(weather_mult, dtype=float) if weather_mult is not None else np.ones(n_min)
    event_arr = np.asarray(event_mult, dtype=float) if event_mult is not None else np.ones(n_min)
    if len(weather_arr) != n_min:
        raise ValueError(f"weather_mult length {len(weather_arr)} != shift length {n_min}")
    if len(event_arr) != n_min:
        raise ValueError(f"event_mult length {len(event_arr)} != shift length {n_min}")

    day_type = day_type_for(day_of_week)
    profile_arr = np.array([temporal_profile(m % 1440, day_type) for m in minutes])
    combined_mult = profile_arr * weather_arr * event_arr

    restaurants = _load_restaurants(restaurants_path)
    population = _load_population(population_path)

    rngs = rng_streams(scenario_seed)
    orders_rng = rngs["orders"]
    kitchen_rng = rngs["kitchen"]
    competitors_rng = rngs["competitors"]

    model = _DemandModel(restaurants, population, kitchen_rng)

    rate = DEMAND_CALIBRATION["orders_per_weight_unit_per_min"]
    demand_by_cell: dict[str, np.ndarray] = {
        cell: model.cell_weight_sum[cell] * rate * combined_mult for cell in model.origin_cells
    }
    demand_full = {cell: demand_by_cell.get(cell, np.zeros(n_min)) for cell in model.full_cell_grid}

    surge_field = surge_mod.build_supply_and_surge(
        rng=competitors_rng,
        cells=model.full_cell_grid,
        minutes=minutes,
        demand_by_cell=demand_full,
        courier_supply_mult=courier_supply_mult,
        baseline_weights=model.baseline_weights,
    )

    fare = FARE_CALIBRATION
    prep_cal = PREP_TIME_CALIBRATION
    tip_cal = TIP_CALIBRATION
    jitter_radius = DEMAND_CALIBRATION["jitter_radius_km"]

    orders: list[OrderOffer] = []
    order_seq = 0

    lat_col = restaurants["lat"].to_numpy(dtype=float)
    lon_col = restaurants["lon"].to_numpy(dtype=float)
    denue_col = restaurants["denue_id"].to_numpy()
    cell_col = restaurants["cell"].to_numpy()

    for t_idx, minute in enumerate(minutes):
        for cell in model.origin_cells:
            lam = demand_by_cell[cell][t_idx]
            if lam <= 1e-12:
                continue
            count = orders_rng.poisson(lam)
            if count <= 0:
                continue

            local_idx = model.cell_restaurant_idx[cell]
            local_probs = model.cell_restaurant_probs[cell]
            chosen_local = orders_rng.choice(len(local_idx), size=count, p=local_probs)
            chosen_global = local_idx[chosen_local]

            surge_now = surge_field.at(cell, minute)

            for g_idx in chosen_global:
                origin_lat = float(lat_col[g_idx])
                origin_lon = float(lon_col[g_idx])
                denue_id = str(denue_col[g_idx])

                dest_idx = orders_rng.choice(len(model.pop_cells), p=model.dest_probs[g_idx])
                dest_cell = model.pop_cells[dest_idx]
                dest_centroid_lat, dest_centroid_lon = geo.cell_centroid(dest_cell)
                jitter_r = jitter_radius * np.sqrt(orders_rng.uniform(0.0, 1.0))
                jitter_bearing = orders_rng.uniform(0.0, 2 * np.pi)
                dest_lat, dest_lon = _offset_point_km(dest_centroid_lat, dest_centroid_lon, jitter_r, jitter_bearing)

                ref_km = geo.great_circle_km(origin_lat, origin_lon, dest_lat, dest_lon)
                ref_minutes = ref_km / fare["ref_speed_kmh"] * 60.0

                restaurant_row = restaurants.iloc[g_idx]
                mean_prep = model.restaurant_prep_mean[restaurant_row.name]
                prep_minutes = float(
                    np.clip(
                        kitchen_rng.normal(mean_prep, mean_prep * prep_cal["per_order_cv"]),
                        prep_cal["min_minutes"],
                        prep_cal["max_minutes"],
                    )
                )

                gross_payout = fare["base_mxn"] + fare["per_km_mxn"] * ref_km + fare["per_min_mxn"] * ref_minutes

                conditions_signal = abs(weather_arr[t_idx] - 1.0)
                tip_pct = float(
                    np.clip(
                        orders_rng.normal(
                            tip_cal["base_pct_mean"] + tip_cal["conditions_pct_gain"] * conditions_signal,
                            tip_cal["base_pct_std"],
                        ),
                        tip_cal["min_pct"],
                        tip_cal["max_pct"],
                    )
                )
                tip_mxn = round(gross_payout * tip_pct, 2)

                order_seq += 1
                orders.append(
                    OrderOffer(
                        order_id=f"ORD-{date.isoformat()}-{order_seq:06d}",
                        spawn_min=minute,
                        restaurant_denue_id=denue_id,
                        origin_cell=str(cell_col[g_idx]),
                        origin_lat=origin_lat,
                        origin_lon=origin_lon,
                        dest_cell=dest_cell,
                        dest_lat=dest_lat,
                        dest_lon=dest_lon,
                        ref_km=round(ref_km, 4),
                        ref_minutes=round(ref_minutes, 2),
                        base_mxn=fare["base_mxn"],
                        per_km_mxn=fare["per_km_mxn"],
                        per_min_mxn=fare["per_min_mxn"],
                        surge_at_spawn=surge_now,
                        prep_minutes=round(prep_minutes, 2),
                        tip_mxn=tip_mxn,
                    )
                )

    return orders
