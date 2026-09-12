"""Geometry the agent is allowed to do for itself.

A courier can look at a map and judge that a pickup is "about four kilometres
north". That is all this module does: great-circle distance, and matching a
coordinate to the nearest cell of the in-app heatmap.

The heatmap is the agent's only spatial index. It knows no other grid, cannot
enumerate the city, and never learns a cell it has not been shown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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

    @classmethod
    def from_heatmap(cls, heatmap, fallback_cell: str) -> "CellIndex":
        return cls(
            coordinates={cell.cell: (cell.lat, cell.lon) for cell in heatmap},
            fallback_cell=fallback_cell,
        )

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

    def known_cells(self) -> list[str]:
        return sorted(self.coordinates)
