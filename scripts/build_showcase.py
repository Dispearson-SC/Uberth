"""Record two shifts for the demo: an ordinary afternoon, and a chaotic one.

This is the SHOWCASE pair, not a controlled experiment, and the difference
matters enough to say up front. The two scenarios differ in two ways at once
-- the weather and the road closures -- so nothing here isolates the effect of
either. What each scenario DOES hold fixed is the thing the demo is actually
about: inside one scenario both couriers work the same city, the same order
stream, the same weather and the same closed roads. The A/B is smart against
the payout floor, within a day; the two days are two worlds to look at, not a
measurement of what chaos costs.

WHAT MAKES THE CHAOTIC DAY CHAOTIC. Five road closures ACTIVE AT THE SAME
TIME, not five taken in turn. `build_days.py` spaced its five an hour apart,
which meant the city never had more than one closed road at a time and the
courier could always route around the single obstacle. Here the windows
overlap on purpose, so there is a stretch of the dinner peak with all five
shut at once and detours that have to cross each other. On top of that the
day itself is the harshest in the weather archive.

FAIRNESS, WHICH THIS PROJECT HAD TO FIX ONCE ALREADY. The closures alternate
between the two couriers' own corridors, because a shared event timeline is
not a shared experience: a closed road only costs the courier who needed it.
The script counts how many each courier actually perceived and says so
loudly if either met none.

WHAT IS NEW IN THE RECORDING (replay schema 3):
  - the surge map over the whole shift, per cell per minute, so the UI can
    draw where the city is paying and when;
  - each closure's real closed-road POLYLINES rather than one pin at the
    first node of the corridor, plus the short jam radius around them;
  - when each event ENDS, not only when it starts.

Usage:
    .venv\\Scripts\\python.exe scripts/build_showcase.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date as Date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_shift as rs  # noqa: E402
from src.agent.calibration import BASELINE_CALIBRATION  # noqa: E402
from src.eval.replay import (  # noqa: E402
    SurgeGridRecord,
    WorldEventRecord,
    build_replay,
    write_replay,
)
from src.world import events as events_mod  # noqa: E402
from src.world import geo  # noqa: E402

# The ordinary afternoon: the reference Friday every other figure in this
# project is quoted on.
CALM_DAY = ("2026-07-10", "Friday", "Ordinary Friday",
            "34.3 C, trace rain, no road closures — the shift every other figure is quoted on")
# The harshest day in the weather archive, and then five roads shut at once.
CHAOS_DAY = ("2026-06-18", "Thursday", "Chaotic Thursday",
             "41.9 C, above 40 for three hours, rain at the dinner peak, five roads closed together")

# (offset from shift start, duration). Chosen to OVERLAP: every one of these
# is open at minute +320 and all five are still shut at +340, which puts the
# whole pile-up inside the dinner peak where the orders are.
CLOSURE_PLAN: tuple[tuple[int, int], ...] = (
    (240, 180),
    (260, 170),
    (280, 160),
    (300, 150),
    (320, 140),
)

# How many of the courier's busiest cells a closure spans. `corridor_closure`
# shuts the shortest paths between the hot cell and its partners, so this is
# the dial between "an avenue is shut" and "a district is shut". The default
# of 7 closed 484 to 893 road edges apiece -- five of those at once painted
# most of the metropolitan area red, which is not a road closure, it is a
# curfew. Three keeps each one a corridor a courier would recognise.
CORRIDOR_CELLS = 3

# The surge layer is recorded every SURGE_STRIDE_MIN minutes rather than every
# minute. Surge moves on a several-minute feedback loop -- it physically
# cannot carry per-minute detail -- and storing it per minute quadrupled the
# file for a curve the eye cannot tell apart.
SURGE_STRIDE_MIN = 4

MANIFEST_NAME = "showcase.json"


def _cell_minutes(result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tick in result.ticks:
        counts[tick.courier.cell] = counts.get(tick.courier.cell, 0) + 1
    return counts


def _world_events(scenario, graph) -> list[WorldEventRecord]:
    """Ground-truth descriptions for the replay, WITH the geometry a map can
    actually draw. `perceived_minute` is deliberately absent -- `build_replay`
    computes that itself from what the courier's own ticks perceived."""
    records: list[WorldEventRecord] = []
    for event in scenario.events_timeline:
        anchor = events_mod._event_anchor_latlon(event, graph)
        lat, lon = anchor if anchor is not None else (None, None)
        segments = events_mod.closure_segments(event, graph)
        records.append(WorldEventRecord(
            event_id=event.event_id,
            kind=event.type.value,
            lat=lat,
            lon=lon,
            affects_cells=tuple(event.cells or ()),
            ground_truth_minute=event.start_min,
            ends_minute=event.start_min + event.duration_min,
            segments=tuple(tuple(seg) for seg in segments),
            jam_radius_km=events_mod.CLOSURE_JAM_RADIUS_KM if segments else None,
        ))
    return records


def _surge_grid(surge_field, start_min: int, end_min: int) -> SurgeGridRecord | None:
    """The scenario's surge map, thinned in time and trimmed to the shift.

    Only cells the grid actually has a series for are kept, and a cell that
    never leaves 1.00 for the whole shift is dropped: a flat cell paints a
    uniform nothing on the map and is pure file weight.
    """
    if surge_field is None:
        return None
    minutes = [m for m in surge_field.minutes
               if start_min <= m < end_min and (m - start_min) % SURGE_STRIDE_MIN == 0]
    if not minutes:
        return None

    cells: list[str] = []
    boundaries: list[tuple[tuple[float, float], ...]] = []
    values: list[tuple[float, ...]] = []
    for cell in surge_field.cells:
        series = tuple(surge_field.at(cell, m) for m in minutes)
        if max(series) <= 1.005:
            continue
        cells.append(cell)
        boundaries.append(tuple(geo.h3.cell_to_boundary(cell)))
        values.append(series)
    if not cells:
        return None
    return SurgeGridRecord(
        minutes=tuple(minutes),
        cells=tuple(cells),
        boundaries=tuple(boundaries),
        values=tuple(values),
    )


def _summary(outcome) -> dict:
    result = outcome.result
    last = max(result.ticks, key=lambda t: t.minute).courier
    payout = sum(d.payout_mxn for d in result.deliveries)
    tips = sum(d.tip_mxn for d in result.deliveries)
    hours = last.minutes_elapsed / 60.0
    n = max(len(result.deliveries), 1)
    return {
        "take_home": round(payout + tips, 1),
        "mxn_h": round((payout + tips) / hours, 1) if hours else 0.0,
        "mxn_km": round((payout + tips) / last.km_traveled, 2) if last.km_traveled else 0.0,
        "km": round(last.km_traveled, 1),
        "deliveries": len(result.deliveries),
        "per_hour": round(len(result.deliveries) / hours, 2) if hours else 0.0,
        "mxn_per_trip": round((payout + tips) / n, 1),
    }


def _closures_met(result, ids: set[str]) -> int:
    met: set[str] = set()
    for tick in result.ticks:
        met.update(e for e in tick.perceived_event_ids if e in ids)
    return len(met)


def _conditions(scenario, window) -> dict:
    weather = [w for w in scenario.weather_timeline
               if window.start_min <= w.minute < window.end_min]
    return {
        "apparent_min": round(min(w.apparent_c for w in weather), 1),
        "apparent_max": round(max(w.apparent_c for w in weather), 1),
        "minutes_over_40c": sum(1 for w in weather if w.is_extreme_heat),
        "raining_minutes": sum(1 for w in weather if w.is_raining),
        "peak_rain_mm_per_hour": round(max(w.precip_mm for w in weather) * 60, 1),
    }


def _peak_concurrent(closures) -> tuple[int, int]:
    """(most closures shut at the same minute, the minute it happens)."""
    best = (0, 0)
    if not closures:
        return best
    lo = min(c.start_min for c in closures)
    hi = max(c.start_min + c.duration_min for c in closures)
    for minute in range(lo, hi + 1):
        n = sum(1 for c in closures if c.is_active(minute))
        if n > best[0]:
            best = (n, minute)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record the demo pair: an ordinary afternoon and a chaotic one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", default="reference", choices=sorted(rs.SHIFT_WINDOWS))
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "replays")
    args = parser.parse_args(argv)

    window = rs.SHIFT_WINDOWS[args.window]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("Building the travel oracle once ...", flush=True)
    oracle = rs.load_base_oracle(use_cache=True)
    graph = oracle.matrix.graph
    skeleton = rs.load_travel_skeleton()
    original_loader = events_mod._try_load_graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph

    scenarios: list[dict] = []
    try:
        # ------------------------------------------------------------------
        # 1. The ordinary afternoon
        # ------------------------------------------------------------------
        iso, weekday, label, note = CALM_DAY
        print(f"\n=== calm — {iso} ({weekday}) {label} ===", flush=True)
        calm_built = rs.build_scenario(
            args.seed, Date.fromisoformat(iso), window.start_min, window.end_min, weekday
        )
        calm_conditions = _conditions(calm_built.scenario, window)
        print(f"  {calm_conditions['apparent_min']}-{calm_conditions['apparent_max']} C, "
              f"rain on {calm_conditions['raining_minutes']} min", flush=True)

        calm_runs: dict[str, object] = {}
        for policy, tag in (("smart", "smart"), ("fixed_threshold", "floor")):
            outcome, _ = rs.run_one(
                rs.POLICY_FACTORIES[policy](), calm_built, oracle, window, graph, skeleton
            )
            calm_runs[tag] = outcome
            s = _summary(outcome)
            print(f"  calm  {tag:<6} {s['mxn_h']:>6.1f} MXN/h  {s['per_hour']:>4.2f}/h  "
                  f"{s['km']:>6.1f} km  {s['mxn_per_trip']:>5.1f} MXN/trip", flush=True)

        # ------------------------------------------------------------------
        # 2. The chaotic day: harshest weather, five roads shut together
        # ------------------------------------------------------------------
        c_iso, c_weekday, c_label, c_note = CHAOS_DAY
        print(f"\n=== chaos — {c_iso} ({c_weekday}) {c_label} ===", flush=True)
        chaos_built = rs.build_scenario(
            args.seed, Date.fromisoformat(c_iso), window.start_min, window.end_min, c_weekday
        )
        chaos_conditions = _conditions(chaos_built.scenario, window)
        print(f"  {chaos_conditions['apparent_min']}-{chaos_conditions['apparent_max']} C, "
              f"{chaos_conditions['minutes_over_40c']} min above 40 C, "
              f"rain on {chaos_conditions['raining_minutes']} min "
              f"(peak {chaos_conditions['peak_rain_mm_per_hour']} mm/h)", flush=True)

        # Probe runs on the clean version of the chaotic day, purely to learn
        # where each courier actually rides, so the closures can be put on
        # roads they were going to use.
        print("  probing both couriers' corridors on the clean day ...", flush=True)
        corridors: dict[str, list[str]] = {}
        for policy, tag in (("smart", "smart"), ("fixed_threshold", "floor")):
            probe, _ = rs.run_one(
                rs.POLICY_FACTORIES[policy](), chaos_built, oracle, window, graph, skeleton
            )
            corridors[tag] = events_mod.corridor_from_occupancy(
                _cell_minutes(probe.result), limit=CORRIDOR_CELLS
            )

        closures = []
        for index, (offset, duration) in enumerate(CLOSURE_PLAN, start=1):
            tag = "smart" if index % 2 else "floor"
            try:
                closures.append(events_mod.corridor_closure(
                    f"chaos-closure-{index:02d}-{tag}", window.start_min + offset,
                    duration, corridors[tag], oracle.matrix,
                ))
            except ValueError as exc:
                print(f"    closure {index} on {tag} skipped: {exc}", flush=True)
        if not closures:
            raise SystemExit("no closure could be placed; the chaotic day would be a calm day")

        peak_n, peak_min = _peak_concurrent(closures)
        print(f"  {len(closures)} closures placed; {peak_n} shut at once at minute {peak_min}",
              flush=True)
        if peak_n < 2:
            print("    WARNING: the closures never overlap — this is not a chaotic day, it is "
                  "five ordinary ones in a row.", flush=True)

        chaos_scenario = chaos_built.scenario.model_copy(update={"events_timeline": sorted(
            list(chaos_built.scenario.events_timeline) + closures, key=lambda e: e.start_min)})
        chaos_with_closures = rs.BuiltScenario(
            scenario=chaos_scenario,
            platform=chaos_built.platform,
            surge_field=chaos_built.surge_field,
        )
        closure_ids = {c.event_id for c in closures}

        chaos_runs: dict[str, object] = {}
        met: dict[str, int] = {}
        for policy, tag in (("smart", "smart"), ("fixed_threshold", "floor")):
            outcome, _ = rs.run_one(
                rs.POLICY_FACTORIES[policy](), chaos_with_closures, oracle, window, graph, skeleton
            )
            chaos_runs[tag] = outcome
            met[tag] = _closures_met(outcome.result, closure_ids)
            s = _summary(outcome)
            print(f"  chaos {tag:<6} {s['mxn_h']:>6.1f} MXN/h  {s['per_hour']:>4.2f}/h  "
                  f"{s['km']:>6.1f} km  {s['mxn_per_trip']:>5.1f} MXN/trip  "
                  f"met {met[tag]}/{len(closures)} closures", flush=True)
        if min(met.values()) == 0:
            print("    WARNING: one courier met NO closure — that scenario compares a hard shift "
                  "against an easy one and must not be shown as like-for-like.", flush=True)

        # ------------------------------------------------------------------
        # 3. Write
        # ------------------------------------------------------------------
        plan = (
            ("calm", CALM_DAY, calm_built.scenario, calm_built.surge_field,
             calm_conditions, calm_runs, 0, {}),
            ("chaos", CHAOS_DAY, chaos_scenario, chaos_built.surge_field,
             chaos_conditions, chaos_runs, len(closures), met),
        )
        for key, day, scenario, surge_field, conditions, runs, n_closures, met_map in plan:
            day_iso, day_weekday, day_label, day_note = day
            events = _world_events(scenario, graph)
            grid = _surge_grid(surge_field, window.start_min, window.end_min)
            drawn = sum(len(e.segments) for e in events)
            print(f"\n  {key}: {len(events)} world events ({drawn} drawable road segments), "
                  f"surge grid {0 if grid is None else len(grid.cells)} cells x "
                  f"{0 if grid is None else len(grid.minutes)} samples", flush=True)

            files = {}
            for tag, outcome in runs.items():
                courier = "smart" if tag == "smart" else "fixed_threshold"
                name = f"showcase_{key}_{tag}.json"
                size = write_replay(
                    args.out_dir / name,
                    build_replay(
                        outcome.result,
                        world_events=events,
                        courier_ids=[courier],
                        surge=grid,
                    ),
                )
                files[tag] = name
                print(f"    wrote {name}  ({size / 1024:.0f} KB)", flush=True)

            scenarios.append({
                "key": key,
                "date": day_iso,
                "weekday": day_weekday,
                "label": day_label,
                "note": day_note,
                "conditions": conditions,
                "closures": n_closures,
                "closures_met": met_map,
                "peak_concurrent_closures": peak_n if key == "chaos" else 0,
                "files": files,
                "summary": {tag: _summary(out) for tag, out in runs.items()},
            })

        payload = {
            "schema_version": 1,
            "seed": args.seed,
            "window": {"name": args.window, "label": window.label_with_hours},
            "payout_floor_mxn": BASELINE_CALIBRATION["fixed_payout_floor_mxn"],
            "surge_stride_min": SURGE_STRIDE_MIN,
            "scenarios": scenarios,
        }
        path = args.out_dir / MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {path} — {len(scenarios)} scenarios, {len(scenarios) * 2} replay files",
              flush=True)
    finally:
        events_mod._try_load_graph = original_loader

    print(f"Total wall time: {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
