"""Ground truth: spatial foundation for the operating area.

This module is ground truth. It defines the operating polygon, the H3
resolution-7 grid that discretizes it, and coordinate/geometry helpers used
by every other world module. `src/agent/` must never import this module
directly.

Operating area: Monterrey (039), San Pedro Garza García (019), San Nicolás
de los Garza (046), and Guadalupe (026), all in Nuevo León (entity 19). The
actual polygon is NOT hardcoded — it is derived from the convex hull of real
DENUE establishment coordinates in these four municipalities (see
`build_operating_polygon`), so the simulated area is grounded in where
businesses actually are, not a guessed bounding box.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# --- Constants -------------------------------------------------------------

ENTITY_CODE = "19"  # Nuevo León (INEGI cve_ent)

# Target municipalities: INEGI cve_mun -> name. This is real INEGI catalog
# data (municipality codes), not a calibration knob.
TARGET_MUNICIPALITIES: dict[str, str] = {
    "039": "Monterrey",
    "019": "San Pedro Garza García",
    "046": "San Nicolás de los Garza",
    "026": "Guadalupe",
}

H3_RESOLUTION = 7

EARTH_RADIUS_KM = 6371.0088

# CALIBRATION VALUE (not measured): margin added around the convex hull of
# real restaurant coordinates before filling with H3 cells, so cells right
# at the edge of the point cloud aren't clipped and the street graph has a
# bit of routing context beyond the outermost venues.
OPERATING_AREA_BUFFER_KM = 2.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
CELLS_FIXTURE_PATH = FIXTURES_DIR / "cells.parquet"


# --- Coordinate / H3 helpers ------------------------------------------------


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in kilometers between two points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# Mean distance between two points drawn uniformly at random inside a
# regular hexagon, as a fraction of its edge length. A standard closed-form
# result, not a fitted constant.
HEXAGON_MEAN_POINT_DISTANCE_PER_EDGE = 0.8262


def intra_cell_straight_line_km(resolution: int = H3_RESOLUTION) -> float:
    """Expected straight-line distance of a trip that begins and ends in the
    SAME cell.

    This exists because the cell-to-cell travel matrix has a zero diagonal:
    Dijkstra from a cell's centroid node to itself is 0 km and 0 minutes.
    Left unpatched, that turns every order whose restaurant and customer
    happen to fall in one cell into a free, instantaneous delivery — and a
    policy that scores expected net MXN per hour will find those and take
    nothing else, because a zero-cost trip has an unbounded rate.

    That is exactly what happened. Measured on seed 42 before this fix: 6 of
    the smart courier's 13 deliveries were 0.00 km and 1.0 minute, worth 29%
    of its earnings, all in the last third of the shift. The 55-peso payout
    floor took ZERO of them, because it never looks at geometry. So the
    agent's headline advantage was partly a modelling artefact it had
    learned to farm.

    A resolution-7 cell is about 5.16 km^2, with a mean edge of 1.41 km. Two
    random points inside it average 1.16 km apart. That is the honest
    diagonal, and it is pure geometry: H3 covers the planet, so an agent can
    derive this in any city without a local dataset.
    """
    import h3

    edge_km = float(h3.average_hexagon_edge_length(resolution, unit="km"))
    return HEXAGON_MEAN_POINT_DISTANCE_PER_EDGE * edge_km


def latlon_to_cell(lat: float, lon: float, resolution: int = H3_RESOLUTION) -> str:
    """Map a lat/lon point to its H3 cell index. Does not require the
    operating cell catalog to exist — any point maps to a cell directly."""
    return h3.latlng_to_cell(lat, lon, resolution)


def cell_centroid(cell: str) -> tuple[float, float]:
    """Return (lat, lon) of a cell's centroid."""
    return h3.cell_to_latlng(cell)


def cell_neighbors(cell: str, k: int = 1, include_self: bool = False) -> list[str]:
    """Return cells within k grid-rings of `cell` (h3.grid_disk)."""
    disk = h3.grid_disk(cell, k)
    if not include_self:
        disk = [c for c in disk if c != cell]
    return list(disk)


def cell_distance_km(cell_a: str, cell_b: str) -> float:
    """Great-circle distance in km between two cell centroids."""
    lat1, lon1 = cell_centroid(cell_a)
    lat2, lon2 = cell_centroid(cell_b)
    return great_circle_km(lat1, lon1, lat2, lon2)


# --- Operating polygon / cell set -------------------------------------------


def _buffer_degrees(buffer_km: float, at_lat: float) -> float:
    """Rough km -> degrees conversion for a buffer, adjusted for latitude
    (longitude degrees shrink with latitude; this is a simplification
    appropriate for a small local buffer, not a projected/geodesic buffer)."""
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(math.cos(math.radians(at_lat)), 0.1)
    return buffer_km / min(km_per_deg_lat, km_per_deg_lon)


def build_operating_polygon(
    lats: Sequence[float],
    lons: Sequence[float],
    buffer_km: float = OPERATING_AREA_BUFFER_KM,
) -> Polygon:
    """Convex hull of real establishment points, buffered outward.

    Simpler than a true concave/administrative boundary, but grounds the
    operating area in real DENUE data rather than hardcoded coordinates.
    """
    if len(lats) < 3:
        raise ValueError("Need at least 3 points to build a convex hull")
    points = np.column_stack([np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)])
    hull = Polygon(points).convex_hull
    if buffer_km > 0:
        mean_lat = float(np.mean(lats))
        hull = hull.buffer(_buffer_degrees(buffer_km, mean_lat))
    return hull


def polygon_to_cells(polygon: Polygon, resolution: int = H3_RESOLUTION) -> set[str]:
    """Fill a shapely polygon (lon, lat coordinates) with H3 cells."""
    exterior = list(polygon.exterior.coords)
    if exterior[0] == exterior[-1]:
        exterior = exterior[:-1]
    # shapely coords are (lon, lat); h3.LatLngPoly wants (lat, lon).
    latlng_ring = [(lat, lon) for lon, lat in exterior]
    h3_poly = h3.LatLngPoly(latlng_ring)
    return set(h3.polygon_to_cells(h3_poly, resolution))


def build_cell_index(cells: Iterable[str], resolution: int = H3_RESOLUTION) -> pd.DataFrame:
    """Build the cell catalog dataframe: cell id, centroid lat/lon, resolution."""
    rows = []
    for cell in cells:
        lat, lon = cell_centroid(cell)
        rows.append({"cell": cell, "lat": lat, "lon": lon, "resolution": resolution})
    df = pd.DataFrame(rows).sort_values("cell").reset_index(drop=True)
    return df


def save_cell_index(df: pd.DataFrame, path: Path = CELLS_FIXTURE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_cell_index(path: Path = CELLS_FIXTURE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cell index not found at {path}. Run `python scripts/build_fixtures.py cells` first."
        )
    return pd.read_parquet(path)


def get_or_build_cell_index(
    lats: Sequence[float],
    lons: Sequence[float],
    path: Path = CELLS_FIXTURE_PATH,
    resolution: int = H3_RESOLUTION,
    buffer_km: float = OPERATING_AREA_BUFFER_KM,
    force: bool = False,
) -> pd.DataFrame:
    """Load the cached cell catalog, or build it from real points and persist it.

    Building from points is what makes the operating area data-grounded;
    persisting it is what makes it stable across runs (same file, same
    cells, every time the fixtures aren't rebuilt).
    """
    if path.exists() and not force:
        return load_cell_index(path)
    polygon = build_operating_polygon(lats, lons, buffer_km=buffer_km)
    cells = polygon_to_cells(polygon, resolution=resolution)
    df = build_cell_index(cells, resolution=resolution)
    save_cell_index(df, path)
    return df
