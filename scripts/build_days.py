"""Record several real days, each one twice: as it was, and with roads closed.

Why several days rather than one good one: a single recording cannot tell you
whether a result is a property of the agent or of the weather it happened to
get. Four days of measured Monterrey weather, each run clean and then again
with five street closures, gives eight side-by-side shifts over the same
policies -- and any claim that survives all eight is worth more than one that
needs a particular afternoon.

THE DAYS ARE REAL. Every one is a weekday drawn from
`fixtures/raw/openmeteo_mty_2026-06-01_2026-09-05.json`, chosen for contrast
across the 14:00-22:00 window and nothing else. Weekdays only, because the
traffic profile is 100% measured Monterrey on weekdays and a borrowed weekend
ratio otherwise (see `world/traffic.py`).

WHAT "HOSTILE" ADDS, AND WHAT IT DOES NOT. The day supplies its own weather;
hostile does not make it hotter or wetter. It adds five one-hour road
closures, split between BOTH couriers' own corridors and alternating, so
neither gets the easier half of the shift. That split is not a detail: a
shared event timeline is not a shared experience, because a closed road only
costs the courier who needed it. The script counts how many each courier
actually perceived and says so.

Usage:
    .venv\\Scripts\\python.exe scripts/build_days.py
    .venv\\Scripts\\python.exe scripts/build_days.py --days 2026-06-18,2026-07-15
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
from src.eval.replay import WorldEventRecord, build_replay, write_replay  # noqa: E402
from src.world import events as events_mod  # noqa: E402

# (date, weekday name, short label, what makes it worth recording)
DAYS: list[tuple[str, str, str, str]] = [
    ("2026-07-15", "Wednesday", "Mild and dry", "33.2 C, no rain — the easy afternoon"),
    ("2026-07-10", "Friday", "Reference Friday", "34.3 C, trace rain — the shift every other figure is quoted on"),
    ("2026-08-04", "Tuesday", "Rain all shift", "38.1 C, 15.5 mm across 8 of 8 hours"),
    ("2026-06-18", "Thursday", "Extreme heat", "41.9 C, above 40 for three hours, rain at the dinner peak"),
]

CLOSURE_OFFSETS = (55, 120, 185, 250, 315)
CLOSURE_DURATION_MIN = 60
MANIFEST_NAME = "days.json"


def _cell_minutes(result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tick in result.ticks:
        counts[tick.courier.cell] = counts.get(tick.courier.cell, 0) + 1
    return counts


def _world_events(scenario, graph) -> list[WorldEventRecord]:
    records: list[WorldEventRecord] = []
    for event in scenario.events_timeline:
        anchor = events_mod._event_anchor_latlon(event, graph)
        lat, lon = anchor if anchor is not None else (None, None)
        records.append(WorldEventRecord(
            event_id=event.event_id, kind=event.type.value, lat=lat, lon=lon,
            affects_cells=tuple(event.cells or ()), ground_truth_minute=event.start_min,
        ))
    return records


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record several real days, each clean and with road closures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", default="reference", choices=sorted(rs.SHIFT_WINDOWS))
    parser.add_argument("--days", default="", help="comma-separated ISO dates; default is all four")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "replays")
    args = parser.parse_args(argv)

    wanted = {d.strip() for d in args.days.split(",") if d.strip()}
    days = [d for d in DAYS if not wanted or d[0] in wanted]
    if not days:
        raise SystemExit(f"no known day matched {sorted(wanted)}; known: {[d[0] for d in DAYS]}")

    window = rs.SHIFT_WINDOWS[args.window]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    floor_mxn = BASELINE_CALIBRATION["fixed_payout_floor_mxn"]

    started = time.time()
    print("Building the travel oracle once ...", flush=True)
    oracle = rs.load_base_oracle(use_cache=True)
    graph = oracle.matrix.graph
    skeleton = rs.load_travel_skeleton()
    original_loader = events_mod._try_load_graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph

    manifest: list[dict] = []
    try:
        for iso, weekday, label, note in days:
            print(f"\n=== {iso} ({weekday}) — {label} ===", flush=True)
            built = rs.build_scenario(
                args.seed, Date.fromisoformat(iso), window.start_min, window.end_min, weekday
            )
            weather = [w for w in built.scenario.weather_timeline
                       if window.start_min <= w.minute < window.end_min]
            conditions = {
                "apparent_min": round(min(w.apparent_c for w in weather), 1),
                "apparent_max": round(max(w.apparent_c for w in weather), 1),
                "minutes_over_40c": sum(1 for w in weather if w.is_extreme_heat),
                "raining_minutes": sum(1 for w in weather if w.is_raining),
                "peak_rain_mm_per_hour": round(max(w.precip_mm for w in weather) * 60, 1),
            }
            print(f"  {conditions['apparent_min']}-{conditions['apparent_max']} C, "
                  f"{conditions['minutes_over_40c']} min above 40 C, "
                  f"rain on {conditions['raining_minutes']} min "
                  f"(peak {conditions['peak_rain_mm_per_hour']} mm/h)", flush=True)

            entry = {
                "date": iso, "weekday": weekday, "label": label, "note": note,
                "conditions": conditions, "variants": {},
            }

            # --- the day as it was -------------------------------------
            normal: dict[str, object] = {}
            for policy, tag in (("smart", "smart"), ("fixed_threshold", "floor")):
                outcome, _ = rs.run_one(
                    rs.POLICY_FACTORIES[policy](), built, oracle, window, graph, skeleton
                )
                normal[tag] = outcome
                s = _summary(outcome)
                print(f"  normal  {tag:<6} {s['mxn_h']:>6.1f} MXN/h  {s['per_hour']:>4.2f}/h  "
                      f"{s['km']:>6.1f} km  {s['mxn_per_trip']:>5.1f} MXN/trip", flush=True)

            # --- the same day with roads closed ------------------------
            # Corridors come from the runs just completed, so no extra
            # shifts are paid for: each courier's closures land on the roads
            # that courier actually worked THIS day.
            corridors = {
                tag: events_mod.corridor_from_occupancy(_cell_minutes(out.result))
                for tag, out in normal.items()
            }
            closures = []
            for index, offset in enumerate(CLOSURE_OFFSETS, start=1):
                tag = "smart" if index % 2 else "floor"
                try:
                    closures.append(events_mod.corridor_closure(
                        f"day-closure-{index:02d}-{tag}", window.start_min + offset,
                        CLOSURE_DURATION_MIN, corridors[tag], oracle.matrix,
                    ))
                except ValueError as exc:
                    print(f"    closure {index} on {tag} skipped: {exc}", flush=True)
            if not closures:
                print("    no closure could be placed; skipping the hostile variant", flush=True)
                continue

            hostile_scenario = built.scenario.model_copy(update={"events_timeline": sorted(
                list(built.scenario.events_timeline) + closures, key=lambda e: e.start_min)})
            hostile_built = rs.BuiltScenario(scenario=hostile_scenario, platform=built.platform)
            closure_ids = {c.event_id for c in closures}

            hostile: dict[str, object] = {}
            met: dict[str, int] = {}
            for policy, tag in (("smart", "smart"), ("fixed_threshold", "floor")):
                outcome, _ = rs.run_one(
                    rs.POLICY_FACTORIES[policy](), hostile_built, oracle, window, graph, skeleton
                )
                hostile[tag] = outcome
                met[tag] = _closures_met(outcome.result, closure_ids)
                s = _summary(outcome)
                print(f"  hostile {tag:<6} {s['mxn_h']:>6.1f} MXN/h  {s['per_hour']:>4.2f}/h  "
                      f"{s['km']:>6.1f} km  {s['mxn_per_trip']:>5.1f} MXN/trip  "
                      f"met {met[tag]}/{len(closures)} closures", flush=True)
            if min(met.values()) == 0:
                print("    WARNING: one courier met NO closure — that variant compares a hard "
                      "shift against an easy one and must not be shown as like-for-like.",
                      flush=True)

            # --- write ---------------------------------------------------
            for variant, runs, scenario in (("normal", normal, built.scenario),
                                            ("hostile", hostile, hostile_scenario)):
                events = _world_events(scenario, graph)
                files = {}
                for tag, outcome in runs.items():
                    courier = "smart" if tag == "smart" else "fixed_threshold"
                    name = f"day_{iso}_{variant}_{tag}.json"
                    write_replay(
                        args.out_dir / name,
                        build_replay(outcome.result, world_events=events, courier_ids=[courier]),
                    )
                    files[tag] = name
                entry["variants"][variant] = {
                    "files": files,
                    "summary": {tag: _summary(out) for tag, out in runs.items()},
                    "closures": len(closures) if variant == "hostile" else 0,
                    "closures_met": met if variant == "hostile" else {},
                }
            manifest.append(entry)

        payload = {
            "schema_version": 1,
            "seed": args.seed,
            "window": {"name": args.window, "label": window.label_with_hours},
            "payout_floor_mxn": floor_mxn,
            "days": manifest,
        }
        path = args.out_dir / MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {path} — {len(manifest)} day(s), "
              f"{len(manifest) * 4} replay files", flush=True)
    finally:
        events_mod._try_load_graph = original_loader

    print(f"Total wall time: {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
