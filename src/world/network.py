"""Ground truth: street network and cell-to-cell travel matrix.

This module is ground truth. It builds/loads the OSMnx drive graph for the
operating polygon, assigns free-flow speeds by road class, precomputes the
H3 cell-to-cell distance (km) and free-flow time (min) matrices used for
O(1) lookups in the simulation loop, and implements the street-closure
mechanic (`TravelMatrix.close_streets`). `src/agent/` must never import
this module directly.

Two representations of the same city on purpose:
- The real OSMnx graph: used to precompute the matrix once, to draw route
  polylines on the map, and to recompute rows after a street closure.
- The cell-to-cell matrix: what the simulation loop actually queries every
  tick, so a courier's ETA/route cost isn't a full Dijkstra per tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "fixtures"
GRAPH_FIXTURE_PATH = FIXTURES_DIR / "monterrey_graph.graphml"
MATRIX_FIXTURE_PATH = FIXTURES_DIR / "travel_matrix.npz"

# CALIBRATION VALUE (not measured): extra buffer applied to the operating
# polygon when pulling the drivable street graph, so routes near the edge
# of the operating cells still have real street context to route through
# instead of dead-ending at the polygon boundary.
GRAPH_BUFFER_KM = 3.0

# CALIBRATION VALUES (not measured): free-flow speed assumptions per OSM
# `highway` class, for the Monterrey metro area. These are the theoretical
# uncongested speed used to seed the ground-truth travel-time matrix.
# Real-world congestion is applied later by the exogenous traffic layer
# (a later slice) as a multiplier on top of these free-flow times — it is
# never baked into this table. Deliberately NOT sourced from OSM `maxspeed`
# tags: those are sparse/unreliable for this region, and mixing sources
# would make the ground truth non-auditable and non-reproducible.
HIGHWAY_SPEED_KMH: dict[str, float] = {
    "motorway": 90.0,
    "motorway_link": 50.0,
    "trunk": 80.0,
    "trunk_link": 45.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 35.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
    "unclassified": 30.0,
    "default": 25.0,  # fallback for any highway class not listed above
}


# --- Graph build/load -------------------------------------------------------


def get_or_build_graph(
    polygon: Polygon,
    cache_path: Path = GRAPH_FIXTURE_PATH,
    force: bool = False,
) -> nx.MultiDiGraph:
    """Load the cached drive graph, or download it from OSM and cache it.

    Downloading happens lazily on first call only — never at import time.
    """
    if cache_path.exists() and not force:
        return ox.io.load_graphml(cache_path)
    graph = ox.graph.graph_from_polygon(polygon, network_type="drive", simplify=True)
    graph = assign_edge_speeds(graph)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.io.save_graphml(graph, cache_path)
    return graph


def assign_edge_speeds(
    graph: nx.MultiDiGraph,
    speed_by_class: dict[str, float] = HIGHWAY_SPEED_KMH,
) -> nx.MultiDiGraph:
    """Assign `speed_kph` to every edge directly from the highway-class
    lookup table (see HIGHWAY_SPEED_KMH), then derive `travel_time` via
    osmnx. This intentionally overrides/ignores any OSM `maxspeed` tag so
    the ground-truth free-flow speed stays fully auditable from this file
    alone.
    """
    default_kmh = speed_by_class.get("default", 25.0)
    for _, _, _, data in graph.edges(keys=True, data=True):
        highway = data.get("highway", "unclassified")
        highway_class = highway[0] if isinstance(highway, list) else highway
        data["speed_kph"] = speed_by_class.get(highway_class, default_kmh)
    graph = ox.routing.add_edge_travel_times(graph)
    return graph


def snap_cells_to_nodes(graph: nx.MultiDiGraph, cell_index: pd.DataFrame) -> dict[str, int]:
    """Nearest graph node for each cell centroid.

    Uses a plain euclidean k-d tree in lat/lon space (scipy, already a core
    dependency) rather than `osmnx.distance.nearest_nodes`: that helper
    needs scikit-learn's haversine BallTree for an unprojected graph, and
    pulling in a new dependency just for node-snapping isn't worth it.
    Euclidean-in-degrees distortion is negligible at metro scale and this
    is only used to pick the nearest node, never to measure a real
    distance (that always comes from the graph's own edge lengths).
    """
    node_ids = np.array(list(graph.nodes))
    xs = np.array([graph.nodes[n]["x"] for n in node_ids])
    ys = np.array([graph.nodes[n]["y"] for n in node_ids])
    tree = cKDTree(np.column_stack([xs, ys]))
    query_points = np.column_stack([cell_index["lon"].to_numpy(), cell_index["lat"].to_numpy()])
    _, nearest_idx = tree.query(query_points)
    nearest_nodes = node_ids[nearest_idx]
    return dict(zip(cell_index["cell"], nearest_nodes))


def _multigraph_edge_length_m(graph: nx.MultiDiGraph, u: int, v: int) -> float:
    """Length (m) of the shortest parallel edge between u and v.

    Simplification: when parallel edges exist between the same node pair,
    we take the minimum-length one for distance bookkeeping, matching how
    networkx's Dijkstra picks the minimum-weight parallel edge for time.
    """
    edge_data = graph.get_edge_data(u, v)
    return min(d.get("length", 0.0) for d in edge_data.values())


def _path_length_km(graph: nx.MultiDiGraph, path: list[int]) -> float:
    total_m = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total_m += _multigraph_edge_length_m(graph, u, v)
    return total_m / 1000.0


# --- Travel matrix -----------------------------------------------------------


@dataclass
class TravelMatrix:
    """Cell-to-cell distance (km) and free-flow time (min) matrices, with
    the graph and per-pair node paths kept around so `close_streets` can
    invalidate and recompute only the affected rows instead of the whole
    matrix.
    """

    graph: nx.MultiDiGraph
    cell_order: list[str]
    cell_to_node: dict[str, int]
    distance_km: np.ndarray
    time_min: np.ndarray
    _paths: dict[tuple[int, int], list[int]] = field(default_factory=dict, repr=False)

    # -- construction --

    @classmethod
    def build(cls, graph: nx.MultiDiGraph, cell_index: pd.DataFrame) -> "TravelMatrix":
        cell_order = list(cell_index["cell"])
        cell_to_node = snap_cells_to_nodes(graph, cell_index)
        n = len(cell_order)
        distance_km = np.full((n, n), np.nan)
        time_min = np.full((n, n), np.nan)
        paths: dict[tuple[int, int], list[int]] = {}

        for i, origin_cell in enumerate(cell_order):
            source = cell_to_node[origin_cell]
            times_sec, node_paths = nx.single_source_dijkstra(graph, source, weight="travel_time")
            for j, dest_cell in enumerate(cell_order):
                target = cell_to_node[dest_cell]
                if target not in node_paths:
                    continue
                path = node_paths[target]
                time_min[i, j] = times_sec[target] / 60.0
                distance_km[i, j] = _path_length_km(graph, path)
                paths[(i, j)] = path

        return cls(
            graph=graph,
            cell_order=cell_order,
            cell_to_node=cell_to_node,
            distance_km=distance_km,
            time_min=time_min,
            _paths=paths,
        )

    # -- persistence (matrices + index only; graph is cached separately) --

    def save(self, path: str | Path = MATRIX_FIXTURE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            distance_km=self.distance_km,
            time_min=self.time_min,
            cell_order=np.array(self.cell_order),
        )

    @staticmethod
    def load_matrices(path: str | Path = MATRIX_FIXTURE_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load just the matrices + cell order (no graph, no paths). Enough
        for the simulation loop to query travel km/min; NOT enough to call
        `close_streets` (use `TravelMatrix.build` for that)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Travel matrix not found at {path}. Run scripts/build_fixtures.py matrix.")
        data = np.load(path, allow_pickle=False)
        return data["distance_km"], data["time_min"], list(data["cell_order"])

    # -- street closures --

    def close_streets(self, edges_to_remove: Iterable[tuple[int, int, int]]) -> list[int]:
        """Remove the given (u, v, key) edges from the working graph and
        recompute ONLY the matrix rows whose cached shortest path actually
        used one of those edges. Mutates this TravelMatrix in place and
        returns the list of affected origin-cell row indices.
        """
        removed = set(edges_to_remove)
        removed_pairs = {(u, v) for u, v, _ in removed}

        affected_rows: set[int] = set()
        for (i, _j), path in self._paths.items():
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) in removed_pairs:
                    affected_rows.add(i)
                    break

        working_graph = self.graph.copy()
        for u, v, key in removed:
            if working_graph.has_edge(u, v, key):
                working_graph.remove_edge(u, v, key)
        self.graph = working_graph

        n = len(self.cell_order)
        for i in sorted(affected_rows):
            source = self.cell_to_node[self.cell_order[i]]
            for j in range(n):
                self._paths.pop((i, j), None)
            try:
                times_sec, node_paths = nx.single_source_dijkstra(working_graph, source, weight="travel_time")
            except nx.NodeNotFound:
                times_sec, node_paths = {}, {}
            for j, dest_cell in enumerate(self.cell_order):
                target = self.cell_to_node[dest_cell]
                if target not in node_paths:
                    self.distance_km[i, j] = np.nan
                    self.time_min[i, j] = np.nan
                    continue
                path = node_paths[target]
                self.time_min[i, j] = times_sec[target] / 60.0
                self.distance_km[i, j] = _path_length_km(working_graph, path)
                self._paths[(i, j)] = path

        return sorted(affected_rows)

    # -- rendering --

    def route_polyline(self, cell_a: str, cell_b: str) -> list[tuple[float, float]]:
        """Real (lat, lon) coordinate sequence for the fastest route between
        two cells, for map rendering."""
        i = self.cell_order.index(cell_a)
        j = self.cell_order.index(cell_b)
        path = self._paths.get((i, j))
        if path is None:
            source = self.cell_to_node[cell_a]
            target = self.cell_to_node[cell_b]
            path = nx.shortest_path(self.graph, source, target, weight="travel_time")
        return [(self.graph.nodes[n]["y"], self.graph.nodes[n]["x"]) for n in path]
