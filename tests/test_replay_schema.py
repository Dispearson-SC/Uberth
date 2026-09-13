"""The replay document's schema-3 additions: closure geometry and surge.

These are the fields the dashboard draws the hot zones and the closed roads
from, and they are the first thing in `eval/replay.py` under test at all --
which is worth saying plainly, because the reason the trip-length ceiling
leaked for a whole session was that nobody was checking the shape of what got
written, only that it got written.

What is deliberately NOT asserted here: any calibrated number. These tests
are about the CONTRACT -- that a closure knows its own roads, that a surge
series cannot silently disagree with its own minute axis, and that a
recording made before any of this existed still builds.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.core.ports import CourierActivity, CourierSnapshot, ShiftResult, TickRecord
from src.eval.replay import (
    REPLAY_SCHEMA_VERSION,
    SurgeGridRecord,
    WorldEventRecord,
    build_replay,
)
from src.world import events as events_mod
from src.world.timeline import Event, EventType


# ---------------------------------------------------------------------------
# closure geometry
# ---------------------------------------------------------------------------

def _two_node_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node(1, y=25.670, x=-100.310)
    graph.add_node(2, y=25.675, x=-100.300)
    graph.add_node(3, y=25.680, x=-100.290)
    graph.add_edge(1, 2, 0, length=1200.0)
    graph.add_edge(2, 3, 0, length=1100.0)
    return graph


def _closure(edges: list[tuple[int, int]]) -> Event:
    return Event(
        event_id="closure-test",
        type=EventType.STREET_CLOSURE,
        start_min=900,
        duration_min=60,
        edges=edges,
        close_edges=True,
    )


def _covered(segments: list[list[tuple[float, float]]]) -> set[tuple]:
    """Every consecutive point pair the polylines actually draw over."""
    pairs = set()
    for seg in segments:
        for a, b in zip(seg, seg[1:]):
            pairs.add((tuple(round(c, 6) for c in a), tuple(round(c, 6) for c in b)))
    return pairs


def test_closure_segments_draws_every_closed_edge() -> None:
    """Every closed edge is covered, not one pin for the whole corridor.

    This is the defect the field exists to fix: a corridor closure holds a
    list of edges, and rendering it from its first node alone reported a
    multi-kilometre closure as a dot. Asserted as COVERAGE rather than as a
    segment count, because adjacent edges are legitimately merged into one
    polyline -- see `_merge_chains`.
    """
    graph = _two_node_graph()
    segments = events_mod.closure_segments(_closure([(1, 2), (2, 3)]), graph)

    covered = _covered(segments)
    assert ((25.670, -100.310), (25.675, -100.300)) in covered
    assert ((25.675, -100.300), (25.680, -100.290)) in covered
    assert all(len(seg) >= 2 for seg in segments)


def test_adjacent_edges_merge_into_one_polyline() -> None:
    """Same ink, fewer objects. A corridor closure covers hundreds of edges
    and each one drawn separately becomes its own map vector -- five live at
    once was several thousand paths for what reads as a few closed avenues."""
    graph = _two_node_graph()
    segments = events_mod.closure_segments(_closure([(1, 2), (2, 3)]), graph)

    assert len(segments) == 1, "two edges sharing node 2 should be one run"
    assert segments[0] == [
        pytest.approx((25.670, -100.310)),
        pytest.approx((25.675, -100.300)),
        pytest.approx((25.680, -100.290)),
    ]


def test_disconnected_edges_stay_separate() -> None:
    """Merging joins runs that genuinely touch; it must never weld two roads
    that only pass near each other."""
    graph = _two_node_graph()
    graph.add_node(4, y=25.700, x=-100.250)
    graph.add_node(5, y=25.705, x=-100.245)
    graph.add_edge(4, 5, 0, length=800.0)

    segments = events_mod.closure_segments(_closure([(1, 2), (4, 5)]), graph)
    assert len(segments) == 2


def test_closure_segments_uses_real_edge_geometry_when_the_graph_has_it() -> None:
    """A curved avenue draws as a curve, not as a chord across the blocks it
    bends around."""
    shapely = pytest.importorskip("shapely.geometry")
    graph = _two_node_graph()
    # OSMnx stores edge geometry in (x, y) = (lon, lat) order.
    graph[1][2][0]["geometry"] = shapely.LineString(
        [(-100.310, 25.670), (-100.306, 25.674), (-100.300, 25.675)]
    )

    segments = events_mod.closure_segments(_closure([(1, 2)]), graph)

    assert len(segments) == 1
    assert len(segments[0]) == 3, "the bend in the middle was dropped"
    assert segments[0][1] == pytest.approx((25.674, -100.306))


def test_closure_segments_is_empty_without_a_graph_or_edges() -> None:
    """Fails quiet, not loud: a degraded run with no graph loaded still
    records the event, it just cannot draw it."""
    assert events_mod.closure_segments(_closure([(1, 2)]), None) == []

    point_event = Event(
        event_id="crash-test", type=EventType.CRASH, start_min=900, duration_min=30,
        point_lat=25.67, point_lon=-100.31, point_radius_km=0.5,
    )
    assert events_mod.closure_segments(point_event, _two_node_graph()) == []


def test_closure_segments_skips_edges_the_graph_does_not_have() -> None:
    graph = _two_node_graph()
    segments = events_mod.closure_segments(_closure([(1, 2), (99, 100)]), graph)
    assert len(segments) == 1


# ---------------------------------------------------------------------------
# the surge record
# ---------------------------------------------------------------------------

def _grid(n_minutes: int = 3, n_cells: int = 2) -> SurgeGridRecord:
    ring = ((25.67, -100.31), (25.67, -100.30), (25.68, -100.30))
    return SurgeGridRecord(
        minutes=tuple(range(900, 900 + n_minutes)),
        cells=tuple(f"cell-{i}" for i in range(n_cells)),
        boundaries=tuple(ring for _ in range(n_cells)),
        values=tuple(tuple(1.0 + i * 0.1 for _ in range(n_minutes)) for i in range(n_cells)),
    )


def test_surge_grid_rejects_a_series_that_disagrees_with_its_minute_axis() -> None:
    """A column-wise store has no way to notice a short series at read time --
    it just silently reads the wrong minute, or nothing, for the rest of the
    shift. So it is rejected at construction, where the caller still knows
    what it meant."""
    with pytest.raises(ValueError, match="recorded minutes"):
        SurgeGridRecord(
            minutes=(900, 901, 902),
            cells=("a",),
            boundaries=(((25.67, -100.31),),),
            values=((1.0, 1.2),),  # two values, three minutes
        )


def test_surge_grid_rejects_mismatched_column_counts() -> None:
    with pytest.raises(ValueError, match="same length"):
        SurgeGridRecord(
            minutes=(900,),
            cells=("a", "b"),
            boundaries=(((25.67, -100.31),),),  # one boundary, two cells
            values=((1.0,), (1.1,)),
        )


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------

def _minimal_result() -> ShiftResult:
    courier = CourierSnapshot(
        minute=900, lat=25.67, lon=-100.31, cell="cell-0", activity=CourierActivity.IDLE,
        earnings_mxn=0.0, deliveries_completed=0, km_traveled=0.0, minutes_elapsed=1.0,
        minutes_idle=1.0, carrying_order_ids=(), offers_seen=0, offers_accepted=0,
        fuel_minutes_remaining=300.0, home_lat=25.67, home_lon=-100.31,
        minutes_left_in_shift=420,
    )
    tick = TickRecord(minute=900, courier=courier, decision=None, offers_shown=())
    return ShiftResult(
        policy_name="smart", seed=42, shift_start_min=900, shift_end_min=901,
        earnings_mxn=0.0, deliveries_completed=0, km_traveled=0.0, minutes_elapsed=1.0,
        minutes_idle=1.0, offers_seen=0, offers_accepted=0, ticks=[tick],
    )


def test_document_carries_the_surge_grid_when_one_is_supplied() -> None:
    doc = build_replay(_minimal_result(), surge=_grid())

    assert doc["schema_version"] == REPLAY_SCHEMA_VERSION
    grid = doc["surge_grid"]
    assert grid["minutes"] == [900, 901, 902]
    assert len(grid["cells"]) == len(grid["values"]) == len(grid["boundaries"]) == 2


def test_a_recording_without_surge_says_so_rather_than_omitting_the_key() -> None:
    """`null` and "absent" read the same to a careless consumer and very
    differently to a careful one. The key is always present."""
    doc = build_replay(_minimal_result())
    assert "surge_grid" in doc
    assert doc["surge_grid"] is None


def test_world_event_geometry_defaults_keep_older_callers_working() -> None:
    """`build_days.py` builds `WorldEventRecord`s with no geometry at all and
    must keep doing so -- the new fields are optional on purpose."""
    record = WorldEventRecord(
        event_id="e1", kind="crash", lat=25.67, lon=-100.31,
        affects_cells=(), ground_truth_minute=900,
    )
    doc = build_replay(_minimal_result(), world_events=[record])

    written = doc["world_events"][0]
    assert written["segments"] == []
    assert written["ends_minute"] is None
    assert written["jam_radius_km"] is None
    # The honest field is still computed here and never accepted from a caller.
    assert written["perceived_minute"] is None


def test_closure_geometry_survives_into_the_document() -> None:
    record = WorldEventRecord(
        event_id="c1", kind="street_closure", lat=25.67, lon=-100.31,
        affects_cells=(), ground_truth_minute=900, ends_minute=960,
        segments=(((25.670, -100.310), (25.675, -100.300)),),
        jam_radius_km=events_mod.CLOSURE_JAM_RADIUS_KM,
    )
    written = build_replay(_minimal_result(), world_events=[record])["world_events"][0]

    assert written["ends_minute"] == 960
    assert written["segments"] == [[[25.67, -100.31], [25.675, -100.3]]]
    assert written["jam_radius_km"] == pytest.approx(events_mod.CLOSURE_JAM_RADIUS_KM)
