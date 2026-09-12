"""Geometry the agent is allowed to do for itself.

A courier can look at a map and judge that a pickup is "about four kilometres
north". That is all this module does: great-circle distance, and matching a
coordinate to the nearest cell of the in-app heatmap.

The heatmap is the agent's only spatial index. It knows no other grid, cannot
enumerate the city, and never learns a cell it has not been shown.
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

    @classmethod
    def from_heatmap(cls, heatmap, fallback_cell: str, learned=None) -> "CellIndex":
        """Build the agent's spatial vocabulary for this minute.

        `heatmap` is what the app just showed: cell ids with coordinates.
        They are, however, the app's own COARSE display cells, and the cell
        ids keyed in `Observation.traffic_by_cell` and
        `Observation.demand_by_cell` are a finer grid — so resolving a
        coordinate against the heatmap alone produces a cell id that those
        two dictionaries never contain, and every traffic and demand lookup
        silently falls through to its prior. Measured, that meant the policy
        was scoring every offer at default traffic and default demand for an
        entire shift while believing it was reasoning about both.

        `learned` closes that gap the only way the agent honestly can:
        every minute, `Observation` tells the courier which cell they are
        standing in AND where they are standing. Remembering those pairs
        builds up real coordinates for the fine grid over the shift — a
        courier learning their own city, one street at a time. Cells they
        have never been to stay unknown, which is correct.
        """
        coordinates = {cell.cell: (cell.lat, cell.lon) for cell in heatmap}
        levels = {cell.cell: cell.level for cell in heatmap}
        if learned:
            coordinates.update(learned)
        return cls(coordinates=coordinates, fallback_cell=fallback_cell, levels=levels)

    def nearest(self, lat: float, lon: float) -> str:
        """The heatmap cell whose centroid is closest to a point.

        With an empty heatmap the agent has no spatial vocabulary at all, so it
        falls back to the only cell it knows: the one it is standing in.
        """
        best_cell = self.fallback_cell
        best_km = float("inf")
        for cell, (cell_lat, cell_lon) in self.coordinates.items():
            km = haversine_km(lat, lon, cell_lat, cell_lon)
            if km < best_km:
                best_km = km
                best_cell = cell
        return best_cell

    def coords_of(self, cell: str) -> tuple[float, float] | None:
        return self.coordinates.get(cell)

    def nearest_level(self, lat: float, lon: float) -> int | None:
        """Heat level of the heatmap cell covering a point, or None if the
        app showed no map at all."""
        best_level: int | None = None
        best_km = float("inf")
        for cell, level in self.levels.items():
            cell_lat, cell_lon = self.coordinates[cell]
            km = haversine_km(lat, lon, cell_lat, cell_lon)
            if km < best_km:
                best_km = km
                best_level = level
        return best_level

    def known_cells(self) -> list[str]:
        return sorted(self.coordinates)
