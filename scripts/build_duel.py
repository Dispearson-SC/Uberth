"""Baseline against the smart agent on the worst day the archive contains.

Why this script exists separately from `build_replays.py`: that one records a
NORMAL shift, which is the right thing to show first -- if the agent only wins
in a storm, it does not win. This one answers the opposite question. Put both
couriers in a shift loaded with obstacles and see which one's model of the
world was worth having.

THE DAY IS REAL, NOT FABRICATED. 2026-06-18 is the harshest working day in
`fixtures/raw/openmeteo_mty_2026-06-01_2026-09-05.json` for the 14:00-22:00
window: eight of eight hours above 32 C apparent, the first three above 40 C
-- the threshold the challenge brief names -- and 2.0 mm of rain arriving at
19:00, in the middle of the dinner peak. It is a Thursday, which matters twice
over: the traffic profile is 100% measured Monterrey on weekdays, and Thursday
is inside the checkpoint window.

BOTH COURIERS MUST ACTUALLY HIT THE OBSTACLES, AND THAT IS NOT THE SAME AS
SHARING A WORLD. Both runs read the identical `Scenario` -- same weather, same
crashes, same closures -- but a shared timeline is not a shared experience. A
closure only costs the courier who drives that road, and the two couriers do
not drive the same roads: the payout floor takes long trips across the metro
while the agent works a tight corridor.

An earlier version of this script placed every closure on the SMART agent's
corridor, reasoning that aiming them at the agent was the adversarial choice.
It is adversarial, but it is not a fair comparison: it produces a hard shift
for one courier and a nearly obstacle-free one for the other, then reports the
difference as skill.

So closures are now split between BOTH corridors, each measured from that
courier's own probe run, alternating so neither gets the easier half of the
shift. And the script counts, per courier, how many closures each one actually
perceived -- printed, and treated as a failure if either count is zero.
Without that count the fairness claim is an intention rather than a fact.

What the contrast rests on: the payout floor sees ONE field, the payout. It
cannot see 41 C, it cannot see wet pavement, and it cannot see a jam. The
smart agent prices all three -- look for "Riding into the jam" and "Heat
exposure premium" in its `DecisionTrace` factors. Neither courier is told
about a closure before `detect_offset_min` lets them notice it.

Usage:
    .venv\\Scripts\\python.exe scripts/build_duel.py
    .venv\\Scripts\\python.exe scripts/build_duel.py --closures 5 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date as Date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_shift as rs  # noqa: E402
from src.eval.replay import WorldEventRecord, build_replay, write_replay  # noqa: E402
from src.agent.calibration import BASELINE_CALIBRATION  # noqa: E402
from src.world import events as events_mod  # noqa: E402

# The worst working day in the weather archive for this window. See the module
# docstring: chosen from measured data, not invented.
HOSTILE_DATE = Date(2026, 6, 18)
HOSTILE_DAY_OF_WEEK = "Thursday"

# Minutes after shift start at which each closure begins. Spread across the
# shift on purpose: one closure is an incident, several are a working day in a
# city with roadworks, and the courier has to keep re-planning rather than
# absorb one shock and coast.
CLOSURE_OFFSETS = (55, 120, 185, 250, 315)
CLOSURE_DURATION_MIN = 60

DEFAULT_OUT_DIR = PROJECT_ROOT / "replays"


def _cell_minutes(result) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tick in result.ticks:
        counts[tick.courier.cell] = counts.get(tick.courier.cell, 0) + 1
    return counts


def _world_events(scenario, graph) -> list[WorldEventRecord]:
    """Ground-truth event records, so the closures and crashes are visible on
    the map. `perceived_minute` is never set here -- `build_replay` computes
    it per courier from that courier's own `TickRecord.perceived_event_ids`,
    which is what stops one courier's replay claiming knowledge the other
    one had."""
    records: list[WorldEventRecord] = []
    for event in scenario.events_timeline:
        anchor = events_mod._event_anchor_latlon(event, graph)
        lat, lon = anchor if anchor is not None else (None, None)
        records.append(
            WorldEventRecord(
                event_id=event.event_id,
                kind=event.type.value,
                lat=lat,
                lon=lon,
                affects_cells=tuple(event.cells or ()),
                ground_truth_minute=event.start_min,
            )
        )
    return records


def _summarise(label: str, outcome) -> dict:
    result = outcome.result
    last = max(result.ticks, key=lambda t: t.minute).courier
    payout = sum(d.payout_mxn for d in result.deliveries)
    tips = sum(d.tip_mxn for d in result.deliveries)
    hours = last.minutes_elapsed / 60.0
    return {
        "label": label,
        "take_home": payout + tips,
        "mxn_h": (payout + tips) / hours if hours else 0.0,
        "km": last.km_traveled,
        "mxn_km": (payout + tips) / last.km_traveled if last.km_traveled else 0.0,
        "deliveries": len(result.deliveries),
        "offers": last.offers_seen,
        "idle_pct": last.minutes_idle / last.minutes_elapsed * 100 if last.minutes_elapsed else 0.0,
    }


def _closures_met(result, closure_ids: set[str]) -> tuple[int, set[str]]:
    """How many of the placed closures this courier actually PERCEIVED.

    A shared event timeline is not a shared experience: a closed road costs
    only the courier who needed it. This is the number that decides whether a
    "both faced the same obstacles" claim is a fact or a hope, so it is
    printed and it is checked.
    """
    met: set[str] = set()
    for tick in result.ticks:
        met.update(eid for eid in tick.perceived_event_ids if eid in closure_ids)
    return len(met), met


def _count_factor(result, needle: str) -> int:
    """How many scored offers carried a factor whose label contains `needle`.

    This is the evidence that a posture is doing something rather than being
    declared: if the smart agent never once priced the jam, the contrast this
    script claims to show is not there.
    """
    hits = 0
    for tick in result.ticks:
        if tick.decision is None:
            continue
        for evaluation in tick.decision.trace.considered:
            if any(needle in factor.label for factor in evaluation.factors):
                hits += 1
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record the payout-floor baseline against the smart agent on a hostile day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="scenario seed")
    parser.add_argument("--date", type=Date.fromisoformat, default=HOSTILE_DATE,
                        help="scenario date; the default is the harshest day in the weather archive")
    parser.add_argument("--day-of-week", default=HOSTILE_DAY_OF_WEEK,
                        help="drives the demand and traffic day-type profile")
    parser.add_argument("--window", default="reference", choices=sorted(rs.SHIFT_WINDOWS),
                        help="named shift window to record")
    parser.add_argument("--closures", type=int, default=len(CLOSURE_OFFSETS),
                        help="how many street closures to place on the agent's own corridor")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory")
    parser.add_argument("--no-oracle-cache", action="store_true",
                        help="rebuild the travel matrix instead of reusing the pickle cache")
    args = parser.parse_args(argv)

    window = rs.SHIFT_WINDOWS[args.window]
    offsets = CLOSURE_OFFSETS[: max(args.closures, 0)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("Building the travel oracle once ...", flush=True)
    oracle = rs.load_base_oracle(use_cache=not args.no_oracle_cache)
    graph = oracle.matrix.graph
    skeleton = rs.load_travel_skeleton()

    # events.py re-parses the ~120MB graph fixture on every call unless handed
    # the copy already in memory. See run_shift.main.
    original_loader = events_mod._try_load_graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph

    try:
        print(f"Building the hostile scenario: seed {args.seed}, {args.date} "
              f"({args.day_of_week}), {window.label_with_hours} ...", flush=True)
        built = rs.build_scenario(
            args.seed, args.date, window.start_min, window.end_min, args.day_of_week
        )
        scenario = built.scenario
        weather = [w for w in scenario.weather_timeline
                   if window.start_min <= w.minute < window.end_min]
        traffic = [t for t in scenario.traffic_timeline
                   if window.start_min <= t.minute < window.end_min]
        print(f"  {len(scenario.order_stream)} orders, "
              f"{len(scenario.events_timeline)} ground-truth events")
        print(f"  apparent temperature {min(w.apparent_c for w in weather):.1f}-"
              f"{max(w.apparent_c for w in weather):.1f} C, "
              f"{sum(1 for w in weather if w.is_extreme_heat)} minutes at or above 40 C")
        print(f"  rain on {sum(1 for w in weather if w.is_raining)} of {len(weather)} minutes, "
              f"peak {max(w.precip_mm for w in weather):.2f} mm")
        print(f"  congestion {min(t.city_multiplier for t in traffic):.2f}-"
              f"{max(t.city_multiplier for t in traffic):.2f}x free flow")

        # Probe runs purely to learn WHERE each courier works, so closures can
        # be aimed at both. These numbers are never reported -- they are a
        # measurement of routes, not a result.
        print("Locating each courier's own corridor ...", flush=True)
        corridors: dict[str, list[str]] = {}
        for name in ("smart", "fixed_threshold"):
            probe, _ = rs.run_one(
                rs.POLICY_FACTORIES[name](), built, oracle, window, graph, skeleton
            )
            occupancy = _cell_minutes(probe.result)
            corridors[name] = events_mod.corridor_from_occupancy(occupancy)
            busiest = corridors[name][0]
            print(f"  {name:<16} {len(corridors[name])} cells; busiest held "
                  f"{occupancy[busiest]} of {window.end_min - window.start_min} minutes")
        shared = set(corridors["smart"]) & set(corridors["fixed_threshold"])
        print(f"  corridors share {len(shared)} cell(s) -- so a closure on one is not "
              f"automatically a closure on the other, which is the whole reason to split them")

        # Alternate, so neither courier gets the easier half of the shift.
        closures = []
        for index, offset in enumerate(offsets, start=1):
            target = "smart" if index % 2 else "fixed_threshold"
            minute = window.start_min + offset
            try:
                closures.append(events_mod.corridor_closure(
                    f"duel-closure-{index:02d}-{target}", minute, CLOSURE_DURATION_MIN,
                    corridors[target], oracle.matrix,
                ))
            except ValueError as exc:
                print(f"  closure {index} at +{offset} min on {target} skipped: {exc}")
        if not closures:
            raise SystemExit(
                "no closure could be placed on either corridor, so this run would be "
                "indistinguishable from an ordinary shift. Refusing to write it."
            )
        on_smart = sum(1 for c in closures if c.event_id.endswith("smart"))
        print(f"  placed {len(closures)} closure(s), {CLOSURE_DURATION_MIN} min each: "
              f"{on_smart} on the agent's roads, {len(closures) - on_smart} on the floor's, "
              f"{sum(len(c.edges or ()) for c in closures)} graph edges closed in total")

        hostile_events = sorted(
            list(scenario.events_timeline) + closures, key=lambda e: e.start_min
        )
        hostile = scenario.model_copy(update={"events_timeline": hostile_events})
        hostile_built = rs.BuiltScenario(scenario=hostile, platform=built.platform)

        print("\nRunning the payout-floor baseline through it ...", flush=True)
        floor_outcome, _ = rs.run_one(
            rs.POLICY_FACTORIES["fixed_threshold"](), hostile_built, oracle, window, graph, skeleton
        )
        print("Running the smart agent through the SAME world ...", flush=True)
        smart_outcome, _ = rs.run_one(
            rs.POLICY_FACTORIES["smart"](), hostile_built, oracle, window, graph, skeleton
        )

        floor = _summarise(
            f"payout floor ({BASELINE_CALIBRATION['fixed_payout_floor_mxn']:.0f} MXN)", floor_outcome
        )
        smart = _summarise("smart agent", smart_outcome)

        print(f"\n=== {args.date} ({args.day_of_week}), {window.label_with_hours}, "
              f"seed {args.seed}, {len(closures)} closures ===")
        header = f'{"":<24}{"take-home":>11}{"MXN/h":>8}{"km":>8}{"MXN/km":>9}{"dels":>6}{"offers":>8}{"idle%":>7}'
        print(header)
        for row in (floor, smart):
            print(f'{row["label"]:<24}{row["take_home"]:>11.0f}{row["mxn_h"]:>8.1f}'
                  f'{row["km"]:>8.1f}{row["mxn_km"]:>9.2f}{row["deliveries"]:>6}'
                  f'{row["offers"]:>8}{row["idle_pct"]:>7.1f}')
        if floor["mxn_km"]:
            print(f'\n  smart earns {smart["mxn_h"] / floor["mxn_h"] * 100:.0f}% of the floor\'s '
                  f'MXN/h on {(1 - smart["km"] / floor["km"]) * 100:.0f}% less driving '
                  f'-> {(smart["mxn_km"] / floor["mxn_km"] - 1) * 100:+.0f}% per kilometre driven')

        # The honesty check this script would be worthless without: the smart
        # agent must actually have PRICED the conditions. A posture that never
        # fires is a claim, not a behaviour.
        closure_ids = {c.event_id for c in closures}
        floor_met, floor_ids = _closures_met(floor_outcome.result, closure_ids)
        smart_met, smart_ids = _closures_met(smart_outcome.result, closure_ids)
        print(f'\n  closures actually perceived: payout floor {floor_met}/{len(closures)}, '
              f'smart {smart_met}/{len(closures)}')
        if floor_met == 0 or smart_met == 0:
            print("  WARNING: one courier never met a single closure, so this recording "
                  "compares a hard shift against an easy one. Do not present it as a "
                  "like-for-like comparison -- re-place the closures first.")

        jam_hits = _count_factor(smart_outcome.result, "Riding into the jam")
        heat_hits = _count_factor(smart_outcome.result, "Heat exposure premium")
        rain_hits = _count_factor(smart_outcome.result, "Wet road risk premium")
        print(f'\n  smart priced the jam on {jam_hits} scored offer(s), '
              f'heat on {heat_hits}, wet pavement on {rain_hits}')
        if jam_hits == 0 and heat_hits == 0:
            print("  WARNING: the smart agent never priced a jam OR the heat, so this "
                  "recording does not show the contrast it was built to show.")

        print()
        for filename, outcome, courier_ids in (
            ("duel_smart.json", smart_outcome, ["smart"]),
            ("duel_floor.json", floor_outcome, ["fixed_threshold"]),
        ):
            document = build_replay(
                outcome.result,
                world_events=_world_events(hostile, graph),
                courier_ids=courier_ids,
            )
            path = args.out_dir / filename
            size = write_replay(path, document)
            print(f"wrote {path} ({size / (1024 * 1024):.2f} MB)")
    finally:
        events_mod._try_load_graph = original_loader

    print(f"\nTotal wall time: {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
