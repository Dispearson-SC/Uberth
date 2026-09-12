"""The courier's own external tools.

Public surface:

  - `RawSourceAdapter` implements `src.core.ports.RawSourcePort`: granular
    queries the AGENT pulls, each answerable from a lat/lon and an app
    screen. This is the one to use.
  - `CourierHistory` is the agent's accumulated experience — kitchen memory
    plus the tables fitted from its own completed trips. It is the only
    thing meant to outlive a shift, and handing it to the next shift's
    adapter is what makes a cold-start curve possible.
  - `TravelSkeleton` / `build_travel_skeleton` bootstrap the agent's own
    cell-to-cell free-flow matrix from OSM, offline and once per city.
  - `EnrichmentAdapter` implements the DEPRECATED `EnrichmentPort`, which
    pushes a finished `Observation`. Kept working so the migration had
    nothing broken in the middle of it; see `raw_source.py` for why push
    cannot be exported to a city with no engine in it.

Everything else in this package is an internal estimate builder.
"""

from __future__ import annotations

from src.enrichment.adapter import EnrichmentAdapter
from src.enrichment.history import CourierHistory
from src.enrichment.kitchen_memory import KitchenMemory
from src.enrichment.osm_travel import TravelSkeleton, build_travel_skeleton, poi_coordinates
from src.enrichment.raw_source import RawSourceAdapter

__all__ = [
    "CourierHistory",
    "EnrichmentAdapter",
    "KitchenMemory",
    "RawSourceAdapter",
    "TravelSkeleton",
    "build_travel_skeleton",
    "poi_coordinates",
]
