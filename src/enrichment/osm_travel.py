"""The agent's own cell-to-cell travel skeleton, bootstrapped from OSM.

This is the STRUCTURAL half of the agent's travel model. It answers "how far
apart are these two places, and how fast is the road between them" — geometry,
not conditions.

WHY IT IS PORTABLE, which is the only reason it exists. Every input is
planetary:

  - the drive graph, pulled from OpenStreetMap by bounding box;
  - the H3 grid, which is pure arithmetic on a lat/lon;
  - Dijkstra, which is a textbook;
  - the operating polygon, built by `geo.build_operating_polygon` from POI
    COORDINATES PASSED IN AS PARAMETERS. That function never loads a
    country-specific dataset itself; hand it food-POI coordinates from
    anywhere and it returns that city's working area.

Nothing here is Mexico-specific. Drop the agent in Guadalajara, hand it a
lat/lon and a POI pull, and this builds itself.

OFFLINE, NEVER ONLINE. One Dijkstra per operating cell over a ~95k-node
graph costs about 150 seconds. That is fine, because it happens ONCE on
arrival in a new city and is cached to disk forever after. It is the same
online/offline split the rest of the agent rests on: fitting is expensive,
lookup is free, and the seven-second decision budget only ever pays for the
lookup.

WHY THERE IS NO EXTERNAL TRAFFIC FEED HERE, and this is a design choice
rather than a budget constraint. A commercial traffic feed measures CAR
probe data. A courier on a motorcycle filters between lanes, takes gaps a
car cannot, and parks in thirty seconds. Their travel times are
systematically different from anything a car-derived feed reports, so buying
one would buy a precise measurement of the wrong vehicle. The agent's own
completed trips are not a portable-but-inferior substitute — they are the
CORRECT measurement for this purpose, taken on the actual vehicle over the
actual routes. A hundred of its own trips teach it more than a feed measured
on something else. The travel model is therefore exactly two layers:

    matrix   -> free-flow skeleton, from OSM, built once offline and cached
    learned  -> correction by zone and hour, from the agent's OWN trips

and nothing else. One fewer dependency, one fewer credential, one fewer
thing that can fail on stage, and nothing in the agent that a courier in any
city could not obtain for free.

SEPARATE FROM THE ENGINE'S ORACLE, and that separation is load-bearing.
`src.engine.travel.NetworkTravelOracle` is ground truth: it holds live
`TrafficTick` congestion and real street closures. This object holds neither.
The same published algorithm builds both, over different data, in different
objects — and this module never imports `src.engine`.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.enrichment.calibration import TRAVEL_SKELETON_CALIBRATION
from src.world import geo
from src.world.network import GRAPH_FIXTURE_PATH, TravelMatrix, get_or_build_graph

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Gitignored developer cache, exactly like the engine oracle's. Derived
# entirely from the OSM graph plus the POI coordinates handed in, so
# deleting it only ever costs time.
DEFAULT_CACHE_PATH = PROJECT_ROOT / "cache" / "agent_travel_skeleton.pkl"


@dataclass(frozen=True)
class TravelSkeleton:
    """Free-flow km and minutes between every pair of operating cells.

    Kilometres and minutes are SEPARATE arrays and stay separate all the way
    out of this class. They decouple exactly when a corridor is fast or slow
    for its length, and that decoupling is the road-class information this
    skeleton exists to carry.
    """

    cell_order: tuple[str, ...]
    distance_km: np.ndarray
    free_flow_minutes: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "_row", {cell: i for i, cell in enumerate(self.cell_order)})
        object.__setattr__(
            self, "_centroids", {cell: geo.cell_centroid(cell) for cell in self.cell_order}
        )
        object.__setattr__(self, "_nearest_memo", {})

    # -- placing a coordinate on the skeleton ----------------------------

    def nearest_cell(self, lat: float, lon: float) -> str:
        """The operating cell a coordinate falls in, or the nearest one.

        H3 answers the first question with arithmetic. The fallback matters
        at the edge of the working area: a drop-off just outside the
        polygon still has to be priced, and the nearest cell inside it is
        the honest answer rather than a refusal.
        """
        memo: dict[tuple[float, float], str] = self._nearest_memo  # type: ignore[attr-defined]
        # Rounded to about a metre: finer than that is below the resolution
        # of anything downstream, and the memo is what keeps a decision that
        # weighs a hundred candidate cells inside its budget.
        key = (round(lat, 5), round(lon, 5))
        cached = memo.get(key)
        if cached is not None:
            return cached
        cell = geo.latlon_to_cell(lat, lon)
        if cell in self._row:  # type: ignore[attr-defined]
            memo[key] = cell
            return cell
        best_cell = self.cell_order[0]
        best_km = float("inf")
        for candidate, (c_lat, c_lon) in self._centroids.items():  # type: ignore[attr-defined]
            km = geo.great_circle_km(lat, lon, c_lat, c_lon)
            if km < best_km:
                best_km = km
                best_cell = candidate
        memo[key] = best_cell
        return best_cell

    def known(self, cell: str) -> bool:
        return cell in self._row  # type: ignore[attr-defined]

    def cells(self) -> tuple[str, ...]:
        return self.cell_order

    def centroid(self, cell: str) -> tuple[float, float] | None:
        return self._centroids.get(cell)  # type: ignore[attr-defined]

    # -- the lookup itself -----------------------------------------------

    def leg(self, from_cell: str, to_cell: str) -> tuple[float, float] | None:
        """(km, free-flow minutes) between two cells, or None when the graph
        has no route between them. Never raises: an unroutable pair is a
        fact about the graph, and the caller has a straight-line fallback."""
        i = self._row.get(from_cell)  # type: ignore[attr-defined]
        j = self._row.get(to_cell)  # type: ignore[attr-defined]
        if i is None or j is None:
            return None
        km = float(self.distance_km[i, j])
        minutes = float(self.free_flow_minutes[i, j])
        if km != km or minutes != minutes:  # NaN: no route
            return None
        return km, minutes

    def corridor_speed_kmh(self, from_cell: str, to_cell: str) -> float | None:
        """Free-flow speed of the road between two cells, in km/h.

        This is the one number the skeleton contributes that a straight line
        cannot: a leg along a fast corridor and a leg of the same length
        through a residential grid are not the same trip, and the OSM
        `highway` tag knows it.
        """
        pair = self.leg(from_cell, to_cell)
        if pair is None:
            return None
        km, minutes = pair
        if minutes <= 0.0 or km <= 0.0:
            return None
        cal = TRAVEL_SKELETON_CALIBRATION
        return min(
            cal["max_corridor_speed_kmh"], max(cal["min_corridor_speed_kmh"], km / minutes * 60.0)
        )

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "cell_order": list(self.cell_order),
                    "distance_km": self.distance_km,
                    "free_flow_minutes": self.free_flow_minutes,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> "TravelSkeleton":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            cell_order=tuple(payload["cell_order"]),
            distance_km=payload["distance_km"],
            free_flow_minutes=payload["free_flow_minutes"],
        )


def build_travel_skeleton(
    poi_lats: Sequence[float],
    poi_lons: Sequence[float],
    *,
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    buffer_km: float | None = None,
) -> TravelSkeleton:
    """Bootstrap the skeleton for a city from POI coordinates alone.

    The argument is deliberately a bare pair of coordinate sequences: food
    POIs from an OSM pull, and nothing else. The operating area is their
    buffered hull, the cells are that polygon H3-filled, the graph is OSM
    over that polygon, and the matrix is one Dijkstra per cell.

    Cached on disk because the result is a pure function of the road network
    and those coordinates — nothing seed-, date- or shift-specific enters
    it, so a whole sweep pays for it at most once.
    """
    if cache_path is not None and cache_path.exists():
        return TravelSkeleton.load(cache_path)

    cal = TRAVEL_SKELETON_CALIBRATION
    polygon = geo.build_operating_polygon(
        poi_lats, poi_lons, buffer_km=cal["operating_buffer_km"] if buffer_km is None else buffer_km
    )
    cells = sorted(geo.polygon_to_cells(polygon))
    if not cells:
        raise ValueError("POI coordinates produced an empty operating area")
    cell_index = geo.build_cell_index(cells)
    graph = get_or_build_graph(polygon, cache_path=GRAPH_FIXTURE_PATH)
    logger.info(
        "Bootstrapping the agent's travel skeleton: %d cells over a %d-node OSM graph.",
        len(cells),
        graph.number_of_nodes(),
    )
    matrix = TravelMatrix.build(graph, cell_index)
    skeleton = TravelSkeleton(
        cell_order=tuple(matrix.cell_order),
        distance_km=matrix.distance_km,
        free_flow_minutes=matrix.time_min,
    )
    if cache_path is not None:
        skeleton.save(cache_path)
    return skeleton


def poi_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Food-POI coordinates for the city the simulator is standing in for.

    This is the ONE place the bootstrap touches the simulator's world, and
    it is a deliberate stand-in: in production these coordinates come from
    an OSM POI query over a bounding box, which works in any city. What
    comes back is a pair of coordinate arrays either way — the bootstrap
    above cannot tell the difference and has no other input.
    """
    from src.enrichment.demand_sense import poi_table

    table = poi_table()
    return table["lat"].to_numpy(), table["lon"].to_numpy()
