"""Characterization tests: the disruptive-event timeline (src/world/events.py).

CHARACTERIZATION, NOT UNIT TESTS — see tests/conftest.py's module docstring.

Every test in this module depends on the `loaded_graph` fixture (see
conftest.py), which parses the real ~120MB OSMnx drive-graph fixture ONCE
for the whole module (~20s) and patches `events._try_load_graph` to reuse
it — without that patch, each `build_events_timeline` call below would
independently re-parse the same file. This whole module is marked `slow`
for that reason: run `pytest -m "not slow"` for a fast dev loop.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.world import events as events_mod
from src.world.timeline import EventType

pytestmark = pytest.mark.slow

SEED = 42
DATE = date(2026, 7, 10)  # Friday
SHIFT_START_MIN = 840
SHIFT_END_MIN = 1320


def test_events_timeline_is_deterministic_for_same_seed(loaded_graph):
    events_a = events_mod.build_events_timeline(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    events_b = events_mod.build_events_timeline(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    assert [e.model_dump() for e in events_a] == [e.model_dump() for e in events_b]


def test_events_timeline_differs_for_different_seed(loaded_graph):
    events_a = events_mod.build_events_timeline(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    events_c = events_mod.build_events_timeline(SEED + 1, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    assert [e.model_dump() for e in events_a] != [e.model_dump() for e in events_c]


def test_demo_events_places_event_at_exactly_the_requested_minute():
    """The whole point of demo_events (per its own docstring): being able to
    place a street closure at an exact minute of the shift, rather than
    hoping a random draw lands there. No graph fixture needed — demo_events
    is hand-authored, not RNG- or graph-derived."""
    demo = events_mod.demo_events(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    closure = next(e for e in demo if e.event_id == "demo-street_closure-mid_route")
    assert closure.start_min == SHIFT_START_MIN + 143


def test_crash_detectability_is_honest(loaded_graph):
    """The honesty property of the whole project: a courier standing
    exactly at a crash's anchor coordinate cannot perceive it before
    `start_min + detect_offset_min`, and can from that minute on. If this
    ever breaks, the agent has become clairvoyant.
    """
    events = events_mod.build_events_timeline(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    crash = next(e for e in events if e.type == EventType.CRASH)
    anchor = events_mod._event_anchor_latlon(crash, loaded_graph)
    assert anchor is not None
    lat, lon = anchor

    before = events_mod.perceivable_events(events, crash.detectable_from_min - 1, lat, lon, loaded_graph)
    at_detection = events_mod.perceivable_events(events, crash.detectable_from_min, lat, lon, loaded_graph)

    assert crash not in before
    assert crash in at_detection


def test_checkpoints_absent_on_a_daytime_shift(loaded_graph):
    """Checkpoints are modelled Thursday-Sunday, ~22:00-04:00 (see
    events.py's CHECKPOINT_WEEKDAYS / CHECKPOINT_WINDOW_*). A 14:00-22:00
    shift never overlaps that window, so this must be exactly zero
    regardless of seed — a structural fact about the window, not a
    statistical draw that merely tends toward zero."""
    events = events_mod.build_events_timeline(SEED, DATE, SHIFT_START_MIN, SHIFT_END_MIN)
    checkpoints = [e for e in events if e.type == EventType.CHECKPOINT]
    assert checkpoints == []


def test_checkpoints_present_on_a_night_shift(loaded_graph):
    """Symmetric to the daytime-absence test above: a shift that actually
    overlaps the 22:00-04:00 checkpoint window can and does produce
    checkpoints. Checkpoint count is itself a Poisson draw
    (`CHECKPOINT_RATE_PER_NIGHT`), so this uses a specific seed (1) verified
    by hand to land on a non-zero draw for this window — the point is that
    the mechanism CAN fire on a qualifying night, not that every seed does.
    """
    night_start, night_end = 1080, 1560  # 18:00-02:00, crosses midnight
    events = events_mod.build_events_timeline(1, DATE, night_start, night_end)
    checkpoints = [e for e in events if e.type == EventType.CHECKPOINT]
    assert len(checkpoints) > 0
