"""The courier's traffic app: sparse, stale further out, wrong by design.

A courier does not have a live, citywide, perfectly accurate view of
congestion. This module returns a noisy read of `TrafficTick` for only a
handful of cells — the courier's own cell, its immediate H3 neighbours, and
a plausible corridor toward the day's activity centre (modelled as the
downtown landmark already used elsewhere in `src.world.events` for the demo
timeline) — with error AND staleness that both grow with distance from the
courier. Never the full 127-cell grid, and never the exact multiplier.
"""

from __future__ import annotations

import numpy as np

from src.core.ports import Estimate
from src.enrichment.calibration import TRAFFIC_NOISE_CALIBRATION
from src.world import geo
from src.world.events import DEMO_DOWNTOWN_LAT, DEMO_DOWNTOWN_LON
from src.world.timeline import TrafficTick


def _corridor_cells(courier_lat: float, courier_lon: float, sample_points: int) -> set[str]:
    """A handful of cells along the straight line from the courier toward
    the downtown landmark — a rough stand-in for "the corridor a traffic
    app bothers to show detail along", not a routed path."""
    cells: set[str] = set()
    for i in range(1, sample_points + 1):
        t = i / (sample_points + 1)
        lat = courier_lat + (DEMO_DOWNTOWN_LAT - courier_lat) * t
        lon = courier_lon + (DEMO_DOWNTOWN_LON - courier_lon) * t
        cells.add(geo.latlon_to_cell(lat, lon))
    return cells


def _sparse_cells(courier_lat: float, courier_lon: float, courier_cell: str) -> set[str]:
    cal = TRAFFIC_NOISE_CALIBRATION
    cells = {courier_cell}
    cells.update(geo.cell_neighbors(courier_cell, k=int(cal["neighbor_ring_k"]), include_self=True))
    cells.update(_corridor_cells(courier_lat, courier_lon, int(cal["corridor_sample_points"])))
    return cells


def estimate_traffic(
    minute: int,
    courier_lat: float,
    courier_lon: float,
    courier_cell: str,
    traffic_by_minute: dict[int, TrafficTick],
    shift_start_min: int,
    rng: np.random.Generator,
) -> dict[str, Estimate]:
    """Sparse, noisy, staleness-aware traffic-multiplier estimate.

    Iterates candidate cells in a fixed, deterministic order (sorted cell
    id) so `rng` is consumed identically given the same inputs, which is
    what makes replay byte-identical.
    """
    cal = TRAFFIC_NOISE_CALIBRATION
    out: dict[str, Estimate] = {}

    for cell in sorted(_sparse_cells(courier_lat, courier_lon, courier_cell)):
        lat, lon = geo.cell_centroid(cell)
        distance_km = geo.great_circle_km(courier_lat, courier_lon, lat, lon)

        age_minutes = cal["own_cell_age_minutes"] + cal["age_minutes_per_km"] * distance_km
        lookup_minute = max(shift_start_min, minute - int(round(age_minutes)))
        tick = traffic_by_minute.get(lookup_minute) or traffic_by_minute.get(minute)
        if tick is None:
            continue
        true_multiplier = tick.multiplier_for(cell)

        std_fraction = max(
            cal["base_error_std_fraction"] + cal["error_growth_per_km"] * distance_km,
            cal["min_error_std_fraction"],
        )
        noise = float(rng.normal(0.0, std_fraction * true_multiplier))
        value = max(true_multiplier + noise, 1.0)

        confidence = min(
            cal["base_confidence"],
            max(cal["base_confidence"] - cal["confidence_decay_per_km"] * distance_km, cal["min_confidence"]),
        )

        out[cell] = Estimate(value=value, confidence=confidence, age_minutes=age_minutes)

    return out
