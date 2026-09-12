"""Characterization tests: the hand-placed corridor closure.

CHARACTERIZATION, NOT UNIT TESTS -- see tests/conftest.py.

These run on SYNTHETIC graphs, not the 120MB Monterrey fixture, and
are deliberately not marked `slow`. The behaviour under test is not
Monterrey-specific: it is the rule that a closure must lengthen roads
rather than delete them, and a 9x9 lattice proves or disproves that just as
well as a real city while running in milliseconds.

Why this module exists at all: the first version of the demo closure was
measured to be completely inert. It closed a 2-hop blob around the
courier's own cell, which swallowed the access edges of the cell-centroid
nodes. `close_streets` dutifully recomputed 126 of 127 rows and every one
of the 14,288 still-routable pairs came back changed by exactly 0.000 km,
while 1,590 pairs went unroutable instead. The forked replay was
byte-identical to the reference replay -- same 737.31 MXN, same 82.189 km,
same 13 deliveries, zero diverging minutes out of 490.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.world.events import (
    CORRIDOR_MIN_EDGES,
    _corridor_candidate_edges,
    _protected_nodes,
    _reachable_cells,
    corridor_closure,
    corridor_from_occupancy,
)
from src.world.network import TravelMatrix
from src.world.timeline import EventType

GRID = 15  # wide enough that a blocked middle has a real detour, AND that a
           # corridor path is long enough to leave more than CORRIDOR_MIN_EDGES
           # closable once every centroid neighbourhood is protected


def _node(row: int, col: int) -> int:
    return row * GRID + col


def _grid_matrix() -> TravelMatrix:
    """A 9x9 street lattice with four cells snapped to four corners of the
    interior, so every corridor path crosses the middle of the grid."""
    graph = nx.MultiDiGraph()
    for row in range(GRID):
        for col in range(GRID):
            graph.add_node(_node(row, col), x=float(col), y=float(row))
    for row in range(GRID):
        for col in range(GRID):
            for d_row, d_col in ((0, 1), (1, 0)):
                nr, nc = row + d_row, col + d_col
                if nr < GRID and nc < GRID:
                    # Both directions, so a closure has to block a road
                    # rather than one direction of it.
                    graph.add_edge(_node(row, col), _node(nr, nc), travel_time=60.0, length=1000.0)
                    graph.add_edge(_node(nr, nc), _node(row, col), travel_time=60.0, length=1000.0)

    cell_order = ["cell-nw", "cell-ne", "cell-sw", "cell-se"]
    cell_to_node = {
        "cell-nw": _node(0, 0),
        "cell-ne": _node(0, GRID - 1),
        "cell-sw": _node(GRID - 1, 0),
        "cell-se": _node(GRID - 1, GRID - 1),
    }
    n = len(cell_order)
    distance_km = np.full((n, n), np.nan)
    time_min = np.full((n, n), np.nan)
    paths: dict[tuple[int, int], list[int]] = {}
    for i, origin in enumerate(cell_order):
        times_sec, node_paths = nx.single_source_dijkstra(
            graph, cell_to_node[origin], weight="travel_time"
        )
        for j, dest in enumerate(cell_order):
            target = cell_to_node[dest]
            path = node_paths[target]
            time_min[i, j] = times_sec[target] / 60.0
            distance_km[i, j] = max(len(path) - 1, 0) * 1.0
            paths[(i, j)] = path
    return TravelMatrix(
        graph=graph,
        cell_order=cell_order,
        cell_to_node=cell_to_node,
        distance_km=distance_km,
        time_min=time_min,
        _paths=paths,
    )


CORRIDOR = ["cell-nw", "cell-se", "cell-ne", "cell-sw"]


def test_closure_never_disconnects_a_cell():
    """The defining property. An unroutable pair does not take a longer
    road -- it falls through to `NetworkTravelOracle`'s synthetic fallback,
    which is an invented number, so a closure that disconnects has stopped
    simulating the street network it claims to simulate."""
    matrix = _grid_matrix()
    unroutable_before = int(np.isnan(matrix.time_min).sum())

    event = corridor_closure("probe", 100, 45, CORRIDOR, matrix, protect_hops=1)
    matrix.close_streets(
        [(u, v, key) for u, v in event.edges for key in matrix.graph.get_edge_data(u, v)]
    )

    assert int(np.isnan(matrix.time_min).sum()) == unroutable_before


def test_closure_makes_the_corridor_measurably_slower():
    matrix = _grid_matrix()
    before = matrix.time_min.copy()

    event = corridor_closure("probe", 100, 45, CORRIDOR, matrix, protect_hops=1)
    matrix.close_streets(
        [(u, v, key) for u, v in event.edges for key in matrix.graph.get_edge_data(u, v)]
    )

    slower = (matrix.time_min > before + 1e-9).sum()
    assert slower > 0, "a closure that changes no travel time is inert, which is the bug this guards"


def test_closure_carries_real_graph_edges_and_the_right_type():
    matrix = _grid_matrix()
    event = corridor_closure("probe", 100, 45, CORRIDOR, matrix, protect_hops=1)

    assert event.type is EventType.STREET_CLOSURE
    assert event.close_edges is True
    assert event.cells is None, "an edge-scoped closure must not also carry a cell locator"
    assert event.edges
    for u, v in event.edges:
        assert matrix.graph.has_edge(u, v)


def test_closure_is_learned_about_late_not_announced():
    """`detect_offset_min > 0` is what makes "it did not see this coming"
    an honest sentence on stage."""
    matrix = _grid_matrix()
    event = corridor_closure("probe", 100, 45, CORRIDOR, matrix, protect_hops=1)

    assert event.detect_offset_min > 0
    assert event.detectable_from_min == 100 + event.detect_offset_min


def test_over_protection_raises_instead_of_returning_an_inert_closure():
    """Protecting the whole lattice leaves nothing closable. Returning a
    tiny or empty closure here is exactly how the demo ended up with a fork
    that changed nothing, so this must fail loudly."""
    matrix = _grid_matrix()
    with pytest.raises(ValueError, match=str(CORRIDOR_MIN_EDGES)):
        corridor_closure("probe", 100, 45, CORRIDOR, matrix, protect_hops=GRID)


def test_corridor_needs_at_least_two_cells():
    matrix = _grid_matrix()
    with pytest.raises(ValueError, match="at least an origin"):
        corridor_closure("probe", 100, 45, ["cell-nw"], matrix)


def test_corridor_from_occupancy_ranks_by_minutes_and_drops_unvisited():
    ranked = corridor_from_occupancy({"a": 12, "b": 203, "c": 0, "d": 45}, limit=3)
    assert ranked == ["b", "d", "a"]


# ==========================================================================
# The regression the uniform lattice above CANNOT catch
# ==========================================================================
#
# Every test above passed while the real implementation was still cutting
# 125 cell pairs off the Monterrey road network. A 9x9 lattice is uniform:
# two hops of protection around a centroid always reaches a redundant
# through-street, so hop-based protection looks sufficient when it is not.
#
# Monterrey is not uniform. In the sparse parts of the city a cell's only
# viable artery is further out than two hops, and closing the middle of a
# corridor path that runs down it isolates the cell entirely. The fixture
# below reproduces that shape deliberately: one cell hanging off the
# lattice by a single long access chain.


SPUR_CHAIN_LEN = 7  # long enough that its middle is >2 hops from both ends
SPUR_ANCHOR = (7, 7)  # joins the lattice at the centre, on every corridor path


def _spur_matrix() -> TravelMatrix:
    """The lattice, plus one cell reachable ONLY down a long single chain.

    This is the non-uniform shape a lattice cannot express, and it is the
    shape that broke the real graph."""
    matrix = _grid_matrix()
    graph = matrix.graph
    anchor = _node(*SPUR_ANCHOR)

    chain = [10_000 + i for i in range(SPUR_CHAIN_LEN)]
    previous = anchor
    for offset, node in enumerate(chain, start=1):
        graph.add_node(node, x=float(SPUR_ANCHOR[1]), y=float(SPUR_ANCHOR[0] + offset))
        graph.add_edge(previous, node, travel_time=60.0, length=1000.0)
        graph.add_edge(node, previous, travel_time=60.0, length=1000.0)
        previous = node

    cell_order = list(matrix.cell_order) + ["cell-spur"]
    cell_to_node = dict(matrix.cell_to_node)
    cell_to_node["cell-spur"] = chain[-1]

    n = len(cell_order)
    distance_km = np.full((n, n), np.nan)
    time_min = np.full((n, n), np.nan)
    paths: dict[tuple[int, int], list[int]] = {}
    for i, origin in enumerate(cell_order):
        times_sec, node_paths = nx.single_source_dijkstra(
            graph, cell_to_node[origin], weight="travel_time"
        )
        for j, dest in enumerate(cell_order):
            path = node_paths[cell_to_node[dest]]
            time_min[i, j] = times_sec[cell_to_node[dest]] / 60.0
            distance_km[i, j] = max(len(path) - 1, 0) * 1.0
            paths[(i, j)] = path
    return TravelMatrix(
        graph=graph,
        cell_order=cell_order,
        cell_to_node=cell_to_node,
        distance_km=distance_km,
        time_min=time_min,
        _paths=paths,
    )


SPUR_CORRIDOR = ["cell-nw", "cell-spur", "cell-se", "cell-ne"]


def test_the_spur_fixture_really_does_break_hop_only_protection():
    """The test's own teeth. If a bare two-hop candidate set does NOT
    isolate the spur here, this fixture has stopped reproducing the real
    failure and every assertion below it is worthless."""
    matrix = _spur_matrix()
    protected = _protected_nodes(
        matrix.graph.to_undirected(as_view=False), matrix.cell_to_node, matrix.cell_order,
        {cell: 2 for cell in matrix.cell_order},
    )
    pairs = [("cell-nw", "cell-spur"), ("cell-spur", "cell-nw")]
    naive_edges = _corridor_candidate_edges(matrix, pairs, protected)

    scratch = matrix.graph.copy()
    scratch.remove_edges_from(
        (u, v, key) for u, v in naive_edges for key in matrix.graph.get_edge_data(u, v)
    )
    assert "cell-spur" not in _reachable_cells(scratch, matrix), (
        "the spur fixture no longer reproduces the real bug: two-hop protection kept it reachable"
    )


def test_connectivity_repair_saves_the_spur():
    matrix = _spur_matrix()
    reachable_before = _reachable_cells(matrix.graph, matrix)

    event = corridor_closure("probe", 100, 45, SPUR_CORRIDOR, matrix, protect_hops=2)
    matrix.close_streets(
        [(u, v, key) for u, v in event.edges for key in matrix.graph.get_edge_data(u, v)]
    )

    assert _reachable_cells(matrix.graph, matrix) >= reachable_before
    assert int(np.isnan(matrix.time_min).sum()) == 0


def test_repair_does_not_neuter_the_closure():
    """Repairing connectivity must not quietly reduce the closure to
    nothing -- that trades one silent failure for another."""
    matrix = _spur_matrix()
    before = matrix.time_min.copy()

    event = corridor_closure("probe", 100, 45, SPUR_CORRIDOR, matrix, protect_hops=2)
    matrix.close_streets(
        [(u, v, key) for u, v in event.edges for key in matrix.graph.get_edge_data(u, v)]
    )

    assert len(event.edges) >= CORRIDOR_MIN_EDGES
    assert (matrix.time_min > before + 1e-9).sum() > 0
