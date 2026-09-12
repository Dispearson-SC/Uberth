"""The courier's sense of where demand is — never the ground-truth surge field.

`src.world.surge` and the per-order `surge_at_spawn` are ground truth and are
never imported here, directly or indirectly. Instead this module builds its
own independent estimate from two things a real courier's tools plausibly
give them:

1. The app's own lagged, coarse, quantised heatmap — modelled from real
   DENUE restaurant density (`fixtures/restaurants.parquet`, the same
   density signal `src.world.traffic` uses as its spatial congestion proxy)
   times the public-shape bimodal lunch/dinner curve in
   `src.world.demand.temporal_profile`. Both are legitimate things a courier
   or their app could plausibly know (where restaurants cluster, and that
   lunch/dinner are busy) without ever peeking at the live demand/supply
   mechanism.
2. Perceived SURGE_WINDOW events (via `perceived_events`, already filtered
   for detectability) bump the cells they affect — real, specific,
   corroborating information, on top of the generic rhythm baseline.

Because this is built from a *different, independent* signal than the
surge mechanism (no lag/migration dynamics, no live demand/supply ratio),
it structurally cannot leak the true field — the gap between what this
heatmap shows and what is actually happening is exactly the product being
demonstrated.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.core.ports import Estimate, PerceivedEvent
from src.enrichment.calibration import DEMAND_SENSE_CALIBRATION
from src.world import geo
from src.world.demand import temporal_profile
from src.world.timeline import EventType

RESTAURANTS_FIXTURE_PATH = geo.FIXTURES_DIR / "restaurants.parquet"


@lru_cache(maxsize=1)
def _restaurant_weight_by_cell(path: str) -> dict[str, float]:
    """Real DENUE restaurant weight summed per H3 cell. Same density signal
    `src.world.traffic` uses as its spatial congestion proxy — a legitimate,
    non-ground-truth-leaking thing for a courier's sense of "where the
    restaurants are" to be built from. Cached: a pure read of an on-disk
    fixture that never changes at runtime."""
    df = pd.read_parquet(path)
    return df.groupby("cell")["weight"].sum().to_dict()


def restaurant_weight_by_cell() -> dict[str, float]:
    return _restaurant_weight_by_cell(str(RESTAURANTS_FIXTURE_PATH))


@lru_cache(maxsize=1)
def _poi_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["lat", "lon", "weight", "cell"])


def poi_table() -> pd.DataFrame:
    """Food POIs as (lat, lon, weight, cell), and nothing else.

    Deliberately narrowed to those four columns: this is the shape an OSM
    POI query over a bounding box returns in any city, and it is the only
    shape the portable half of this package is allowed to see. The fixture
    behind it is this simulator's stand-in for that query — the simulator
    IS the world, and the world is allowed Mexico-only sources; the agent
    and the raw-source adapter are not.
    """
    return _poi_table(str(RESTAURANTS_FIXTURE_PATH))


def estimate_demand(
    minute: int,
    day_type: str,
    cell_order: list[str],
    perceived_events: tuple[PerceivedEvent, ...],
    rng: np.random.Generator,
) -> dict[str, Estimate]:
    """Sensed demand intensity per cell, in `cell_order` (deterministic RNG
    consumption order), quantised into a handful of heatmap-style buckets."""
    cal = DEMAND_SENSE_CALIBRATION
    weight_by_cell = restaurant_weight_by_cell()
    max_weight = max(weight_by_cell.values(), default=0.0) or 1.0

    lag_minute_of_day = (minute - cal["lag_minutes"]) % 1440
    rhythm = temporal_profile(lag_minute_of_day, day_type)

    buckets = max(int(cal["quantise_buckets"]), 2)
    out: dict[str, Estimate] = {}

    for cell in cell_order:
        weight = weight_by_cell.get(cell, 0.0)
        raw_score = (weight / max_weight) * rhythm

        value_noise = float(rng.normal(0.0, cal["value_noise_std"]))
        noisy_score = min(max(raw_score + value_noise, 0.0), 1.0)
        quantised = round(noisy_score * (buckets - 1)) / (buckets - 1)

        confidence_noise = float(rng.normal(0.0, cal["confidence_noise_std"]))
        confidence = min(max(cal["base_confidence"] + confidence_noise, 0.0), 1.0)

        out[cell] = Estimate(value=quantised, confidence=confidence, age_minutes=cal["age_minutes"])

    for event in perceived_events:
        if event.kind != EventType.SURGE_WINDOW.value:
            continue
        for cell in event.affects_cells:
            base = out.get(cell)
            bumped_value = min((base.value if base else 0.0) + cal["surge_event_value_bump"], 1.0)
            bumped_confidence = min(
                max(base.confidence if base else 0.0, event.confidence) + cal["surge_event_confidence_bump"],
                1.0,
            )
            out[cell] = Estimate(
                value=bumped_value,
                confidence=bumped_confidence,
                age_minutes=cal["surge_event_age_minutes"],
            )

    return out
