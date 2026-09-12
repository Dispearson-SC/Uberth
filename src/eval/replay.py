"""The replay recording the live demo plays back.

Why this exists: a live-computing demo can stall on stage, and "go back to
the minute it rejected that order" cannot be answered by re-simulating on
the spot. So a shift runs headless once, every tick is recorded to a JSON
file, and the UI becomes a dumb player over static data — zero compute on
stage, instant scrubbing.

Two responsibilities, kept separate:

1. `ShiftRecorder` implements `RecorderPort` exactly as declared in
   `core/ports.py` (`record(tick: TickRecord) -> None`). This is what an
   engine/agent integration plugs in while a shift actually runs.
2. `build_replay` assembles the full JSON document from a finished
   `ShiftResult` (or several, for a multi-courier demo). Routes and
   perceived-event minutes are read straight off the ports contract —
   `ShiftResult.routes`, `TickRecord.route_id`/`route_progress`, and
   `TickRecord.perceived_event_ids` — no injected side-channel data.

The one thing this module still cannot get from `ports.py` is the
ground-truth DESCRIPTION of a world event (its kind, location, affected
cells, and the minute it actually started) — there is no event catalog
type in the ports contract, because that lives in `world/`, which `eval/`
must not import from. `build_replay` therefore still takes that as an
optional external input (`WorldEventRecord`), but the one field that must
never come from ground truth — `perceived_minute` — is no longer accepted
from the caller at all: it is always computed here, from
`TickRecord.perceived_event_ids`, via `resolve_perceived_minute`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.core.ports import (
    Decision,
    DecisionTrace,
    DeliveryRecord,
    RecorderPort,
    ShiftResult,
    TickRecord,
)

REPLAY_SCHEMA_VERSION = 2

# One street polyline, a list of (lat, lon) points, as stored in
# ShiftResult.routes and referenced by id from each tick.
Polyline = Sequence[tuple[float, float]]


class ShiftRecorder:
    """`RecorderPort` implementation: buffers ticks in the order recorded.

    This is intentionally dumb — it satisfies the port contract
    (`record(tick) -> None`) and nothing else. Assembling the replay
    document happens afterwards, in `build_replay`, from the finished
    buffer (or from a `ShiftResult.ticks` list built some other way).
    """

    def __init__(self) -> None:
        self._ticks: list[TickRecord] = []

    def record(self, tick: TickRecord) -> None:
        self._ticks.append(tick)

    @property
    def ticks(self) -> list[TickRecord]:
        return list(self._ticks)


# Compile-time check that ShiftRecorder actually satisfies the port.
_: RecorderPort = ShiftRecorder()


@dataclass(frozen=True)
class WorldEventRecord:
    """Ground-truth description of one world event, as input to
    `build_replay`. There is deliberately no `perceived_minute` field here:
    that is computed by `build_replay` from `TickRecord.perceived_event_ids`
    and can never be supplied by the caller, which is what keeps this
    honest — the recorder can only ever under-claim what the courier saw,
    never leak ground truth into the replay.
    """

    event_id: str
    kind: str
    lat: float | None
    lon: float | None
    affects_cells: tuple[str, ...]
    ground_truth_minute: int


def resolve_perceived_minute(event_id: str, ticks: Sequence[TickRecord]) -> int | None:
    """The first minute, across a tick stream, at which `event_id` appears
    in `TickRecord.perceived_event_ids` — filled by the engine from the
    agent's own enrichment observation, never from ground truth. `None` if
    no tick ever perceives it."""
    for tick in sorted(ticks, key=lambda t: t.minute):
        if event_id in tick.perceived_event_ids:
            return tick.minute
    return None


def _round(value: float, ndigits: int) -> float:
    return round(value, ndigits)


def _score_factor_to_dict(factor: object) -> dict:
    return {
        "label": factor.label,  # type: ignore[attr-defined]
        "delta_mxn": _round(factor.delta_mxn, 2) if factor.delta_mxn is not None else None,  # type: ignore[attr-defined]
        "delta_minutes": _round(factor.delta_minutes, 2) if factor.delta_minutes is not None else None,  # type: ignore[attr-defined]
        "note": factor.note,  # type: ignore[attr-defined]
    }


def _decision_to_dict(decision: Decision | None) -> dict | None:
    if decision is None:
        return None
    trace: DecisionTrace = decision.trace
    return {
        "action": decision.action.value,
        "order_id": decision.order_id,
        "target_cell": decision.target_cell,
        "trace": {
            "minute": trace.minute,
            "considered": [
                {
                    "order_id": e.order_id,
                    "expected_net_mxn": _round(e.expected_net_mxn, 2),
                    "expected_minutes": _round(e.expected_minutes, 2),
                    "expected_km": _round(e.expected_km, 3),
                    "expected_mxn_per_hour": _round(e.expected_mxn_per_hour, 2),
                    "surge_flag": e.surge_flag,
                    "factors": [_score_factor_to_dict(f) for f in e.factors],
                    "rejected_because": e.rejected_because,
                }
                for e in trace.considered
            ],
            "chosen_order_id": trace.chosen_order_id,
            "threshold_mxn_per_hour": _round(trace.threshold_mxn_per_hour, 2),
            "binding_constraint": trace.binding_constraint,
            "summary": trace.summary,
        },
    }


def _delivery_to_dict(delivery: DeliveryRecord | None) -> dict | None:
    if delivery is None:
        return None
    return {
        "order_id": delivery.order_id,
        "accepted_at_min": delivery.accepted_at_min,
        "delivered_at_min": delivery.delivered_at_min,
        "payout_mxn": _round(delivery.payout_mxn, 2),
        "tip_mxn": _round(delivery.tip_mxn, 2),
        "surge_locked": _round(delivery.surge_locked, 3),
        "km": _round(delivery.km, 3),
        "minutes": _round(delivery.minutes, 2),
        "kitchen_wait_minutes": _round(delivery.kitchen_wait_minutes, 2),
    }


def _courier_frame(tick: TickRecord, courier_id: str, routes_table: dict[str, Polyline]) -> dict:
    c = tick.courier
    if tick.route_id is not None and tick.route_id not in routes_table:
        raise ValueError(f"tick at minute {tick.minute} references route_id {tick.route_id!r} not in routes")

    return {
        "courier_id": courier_id,
        "lat": _round(c.lat, 6),
        "lon": _round(c.lon, 6),
        "cell": c.cell,
        "activity": c.activity.value,
        "earnings_mxn": _round(c.earnings_mxn, 2),
        "km_traveled": _round(c.km_traveled, 3),
        "minutes_elapsed": _round(c.minutes_elapsed, 2),
        "minutes_idle": _round(c.minutes_idle, 2),
        "deliveries_completed": c.deliveries_completed,
        "route_ref": tick.route_id,
        "route_progress": _round(tick.route_progress, 4),
        "delivered": _delivery_to_dict(tick.delivered),
    }


def _merge_routes(results: Sequence[ShiftResult]) -> dict[str, Polyline]:
    """Union every courier's `ShiftResult.routes`. Raises if two couriers
    disagree on the polyline for the same route id — that would be an
    engine bug, not something to paper over silently."""
    merged: dict[str, Polyline] = {}
    for result in results:
        for route_id, points in result.routes.items():
            points_tuple = tuple(points)
            if route_id in merged and merged[route_id] != points_tuple:
                raise ValueError(f"route_id {route_id!r} maps to different polylines across ShiftResults")
            merged[route_id] = points_tuple
    return merged


def build_replay(
    results: ShiftResult | Sequence[ShiftResult],
    *,
    world_events: Sequence[WorldEventRecord] | None = None,
    courier_ids: Sequence[str] | None = None,
) -> dict:
    """Assemble the replay document from one or several `ShiftResult`s
    (several, for a multi-courier demo run on the same window and seed).

    Routes come from `ShiftResult.routes` (merged across couriers) and are
    referenced from each frame by `TickRecord.route_id`/`route_progress`.
    World events' `perceived_minute` is always resolved here from
    `TickRecord.perceived_event_ids`, never accepted from the caller.

    Returns a plain JSON-serializable dict with deterministic field order
    and rounded floats, so `json.dumps` on it is byte-identical for
    identical inputs. Use `write_replay` to write it to disk.
    """
    result_list = [results] if isinstance(results, ShiftResult) else list(results)
    if not result_list:
        raise ValueError("build_replay requires at least one ShiftResult")

    ids = list(courier_ids) if courier_ids is not None else [r.policy_name for r in result_list]
    if len(ids) != len(result_list):
        raise ValueError("courier_ids must have the same length as results")

    seeds = {r.seed for r in result_list}
    starts = {r.shift_start_min for r in result_list}
    ends = {r.shift_end_min for r in result_list}
    if len(seeds) != 1 or len(starts) != 1 or len(ends) != 1:
        raise ValueError("all ShiftResults in one replay must share seed, shift_start_min and shift_end_min")

    routes_table = _merge_routes(result_list)

    # Union of all minutes across couriers, in order, so a courier idle-out
    # at a minute another courier still has a tick for still gets a frame.
    ticks_by_courier: list[dict[int, TickRecord]] = [{t.minute: t for t in r.ticks} for r in result_list]
    all_minutes = sorted({minute for ticks in ticks_by_courier for minute in ticks})
    all_ticks_flat: list[TickRecord] = [t for ticks in ticks_by_courier for t in ticks.values()]

    frames: list[dict] = []
    for minute in all_minutes:
        couriers_at_minute = []
        decisions_at_minute = []
        for courier_id, ticks in zip(ids, ticks_by_courier):
            tick = ticks.get(minute)
            if tick is None:
                continue
            couriers_at_minute.append(_courier_frame(tick, courier_id, routes_table))
            decision_dict = _decision_to_dict(tick.decision)
            if decision_dict is not None:
                decisions_at_minute.append({"courier_id": courier_id, **decision_dict})
        frames.append(
            {
                "minute": minute,
                "couriers": couriers_at_minute,
                "decisions": decisions_at_minute,
            }
        )

    events_sorted = sorted(world_events or (), key=lambda e: (e.ground_truth_minute, e.event_id))

    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "policy_names": ids,
        "seed": next(iter(seeds)),
        "shift_start_min": next(iter(starts)),
        "shift_end_min": next(iter(ends)),
        "routes": {
            route_id: [[_round(lat, 6), _round(lon, 6)] for lat, lon in points]
            for route_id, points in sorted(routes_table.items())
        },
        "frames": frames,
        "world_events": [
            {
                "event_id": e.event_id,
                "kind": e.kind,
                "lat": _round(e.lat, 6) if e.lat is not None else None,
                "lon": _round(e.lon, 6) if e.lon is not None else None,
                "affects_cells": list(e.affects_cells),
                "ground_truth_minute": e.ground_truth_minute,
                "perceived_minute": resolve_perceived_minute(e.event_id, all_ticks_flat),
            }
            for e in events_sorted
        ],
    }


def write_replay(path: str | Path, document: dict) -> int:
    """Write the replay document as compact, deterministic JSON. Returns
    the file size in bytes."""
    text = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    Path(path).write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


__all__ = [
    "REPLAY_SCHEMA_VERSION",
    "Polyline",
    "ShiftRecorder",
    "WorldEventRecord",
    "build_replay",
    "resolve_perceived_minute",
    "write_replay",
]
