"""Geometry the agent is allowed to do for itself.

A courier can look at a map and judge that a pickup is "about four kilometres
north". That is all this module does: great-circle distance, and matching a
coordinate to the nearest cell of the in-app heatmap.

The agent's spatial index is whatever its own tools name and place: the app's
coarse heatmap cells, plus every cell `BeliefState.cell_coords` gives a
coordinate for. It knows no other grid, cannot enumerate the city, and never
places a cell it has not been told about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclasses_field

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres. Straight line, not street distance."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    h = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, h))))


@dataclass(frozen=True)
class CellIndex:
    """Cell ids to coordinates, built from the heatmap the app just showed.

    Cell ids are opaque strings to the agent: it never parses them, only
    compares and looks them up.
    """

    coordinates: dict[str, tuple[float, float]]
    fallback_cell: str
    # Quantised demand level per heatmap cell, straight off the app's map.
    levels: dict[str, int] = dataclasses_field(default_factory=dict)
    # Memo for `nearest`/`nearest_level`, keyed on a coordinate rounded to
    # about a metre. Not a micro-optimisation: `nearest` is a linear scan
    # over every cell the courier can place, and weighing every reachable
    # cell as a repositioning target calls it twice per candidate — which
    # is tens of thousands of great-circle computations inside one seven-
    # second decision. Pure function of the coordinate and this index, so
    # the memo cannot change an answer.
    _nearest: dict[tuple[float, float], str] = dataclasses_field(
        default_factory=dict, repr=False, compare=False
    )
    _nearest_level: dict[tuple[float, float], int | None] = dataclasses_field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def from_heatmap(cls, heatmap, fallback_cell: str, cell_coords=None) -> "CellIndex":
        """Build the agent's spatial vocabulary for this minute.

        `heatmap` is what the app just showed: cell ids with coordinates.
        They are, however, the app's own COARSE display cells, and the cell
        ids keyed in `BeliefState.traffic_by_cell` and
        `BeliefState.demand_by_cell` are a finer grid — so resolving a
        coordinate against the heatmap alone produces a cell id that those
        two dictionaries never contain, and every traffic and demand lookup
        silently falls through to its prior. Measured, that meant the policy
        was scoring every offer at default traffic and default demand for an
        entire shift while believing it was reasoning about both.

        `cell_coords` is `BeliefState.cell_coords`, which closes that gap
        outright: the courier's own tools name a cell and say where it is,
        for every cell in either belief map. Somebody looking at the zones
        on their own app knows where those zones are, so this is not
        privileged information — and without it the traffic and demand
        beliefs were unplaceable and therefore dead.

        This replaced a workaround that learned the fine grid one cell at a
        time, by remembering where the courier was standing each minute it
        was told which cell that was. It was honest but weak exactly when it
        mattered most: at the start of a shift the courier knew the
        coordinates of precisely one cell, so every offer was still scored
        against priors through the busiest part of the evening.
        """
        coordinates = {cell.cell: (cell.lat, cell.lon) for cell in heatmap}
        levels = {cell.cell: cell.level for cell in heatmap}
        if cell_coords:
            coordinates.update(cell_coords)
        return cls(coordinates=coordinates, fallback_cell=fallback_cell, levels=levels)

    def nearest(self, lat: float, lon: float) -> str:
        """The heatmap cell whose centroid is closest to a point.

        With an empty heatmap the agent has no spatial vocabulary at all, so it
        falls back to the only cell it knows: the one it is standing in.
        """
        key = (round(lat, 5), round(lon, 5))
        cached = self._nearest.get(key)
        if cached is not None:
            return cached
        best_cell = self.fallback_cell
        best_km = float("inf")
        for cell, (cell_lat, cell_lon) in self.coordinates.items():
            km = haversine_km(lat, lon, cell_lat, cell_lon)
            if km < best_km:
                best_km = km
                best_cell = cell
        self._nearest[key] = best_cell
        return best_cell

    def coords_of(self, cell: str) -> tuple[float, float] | None:
        return self.coordinates.get(cell)

    def nearest_level(self, lat: float, lon: float) -> int | None:
        """Heat level of the heatmap cell covering a point, or None if the
        app showed no map at all."""
        key = (round(lat, 5), round(lon, 5))
        if key in self._nearest_level:
            return self._nearest_level[key]
        best_level: int | None = None
        best_km = float("inf")
        for cell, level in self.levels.items():
            cell_lat, cell_lon = self.coordinates[cell]
            km = haversine_km(lat, lon, cell_lat, cell_lon)
            if km < best_km:
                best_km = km
                best_level = level
        self._nearest_level[key] = best_level
        return best_level

    def known_cells(self) -> list[str]:
        return sorted(self.coordinates)
