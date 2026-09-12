"""Generate the recorded replay files the live-demo dashboard plays back.

Why recorded rather than live: nothing computes on stage, so nothing can
stall in front of a judge, and scrubbing to any minute is instant instead
of a re-simulation. An 8-hour shift already runs in a few seconds, so
recording it once and shipping the JSON is pure upside.

This script is deliberately outside `src/` — it is wiring, exactly like
`scripts/run_shift.py`, and it reuses that script's scenario-building code
directly (`build_scenario`, `run_one`, `load_base_oracle`) rather than
duplicating it, because the two hardest facts in this project ("one order
list, threaded into both sides" and "the oracle is built once, forked per
run") are already solved there.

Three files come out of one run of this script:

  replays/reference_smart.json   -- SmartPolicy on the reference scenario
  replays/reference_fixed.json   -- FixedPayoutThresholdPolicy, SAME scenario
  replays/reference_fork.json    -- SmartPolicy on the SAME scenario plus one
                                     injected STREET_CLOSURE at --fork-offset-min
                                     minutes into the shift

`reference_smart` and `reference_fixed` share one `BuiltScenario` object, so
they face byte-identical weather, traffic, events, supply and orders --
that identity is what makes the dashboard's side-by-side comparison
hermetic rather than approximate.

`reference_fork` reuses the exact same `BuiltScenario` too, with one extra
`STREET_CLOSURE` event appended to `events_timeline` via
`Scenario.model_copy(update=...)`. A `STREET_CLOSURE` event carries
`demand_mult=1.0` and `courier_supply_mult=1.0` by default (see
`src/world/timeline.py`), so it never touches the demand field or the order
stream -- its only effect is a real travel-matrix mutation applied once, at
its start minute, inside `run_shift` (`NetworkTravelOracle.apply_closure`).
That is exactly why the fork run is identical to `reference_smart` up to
the fork minute and diverges only after: same world, same policy, same
RNG-free engine, one street that closes.

The closure is scoped to real graph EDGES along the courier's OWN corridor
-- the cells it actually worked, ranked by how many shift minutes it spent
in each (`_cell_minutes` -> `events.corridor_from_occupancy` ->
`events.corridor_closure`) -- never a fixed landmark cell, and never a blob
around one instant's position. Both of those were tried and both failed,
for two different reasons worth keeping written down:

  * A hardcoded downtown cell is a coin flip on whether this seed's route
    ever passes through it. On the reference seed it does not.
  * A 2-hop blob around the courier's cell at the fork minute swallows the
    cell-centroid nodes' own access edges, so `close_streets` pushes those
    pairs to NaN -- UNROUTABLE, not longer. An unroutable pair answers from
    `NetworkTravelOracle`'s fallback constants instead of from the street
    graph. Measured: 1,590 pairs went NaN and all 14,288 still-routable
    pairs came back changed by exactly 0.000 km, and reference_fork shipped
    byte-identical to reference_smart.

`corridor_closure` protects every centroid's neighbourhood for that reason,
so the router is always left a real, longer road to find. The build now
REFUSES to write the replay set if the fork run comes out identical to the
reference run.

    .venv\\Scripts\\python.exe scripts/build_replays.py
    .venv\\Scripts\\python.exe scripts/build_replays.py --fork-offset-min 200 --seed 7
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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_shift as rs  # noqa: E402 -- the wiring script; see module docstring

from src.eval.replay import WorldEventRecord, build_replay, write_replay  # noqa: E402
from src.eval.runner import SHIFT_WINDOWS  # noqa: E402
from src.world import events as events_mod  # noqa: E402
from src.world.timeline import Event  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "replays"
FILE_SIZE_WARN_BYTES = 2 * 1024 * 1024  # ~2 MB, per the demo build brief


def _cell_minutes(result) -> dict[str, int]:
    """How many minutes of the recorded shift the courier spent in each cell.

    This is what a closure has to be aimed at. Measured on the reference
    seed, the smart courier spent 203 of 480 minutes inside ONE cell and
    entered only seven all shift, so a closure placed anywhere else cannot
    physically touch it -- which is exactly what happened: the first
    version of this fork anchored a 2-hop blob on whichever cell the
    courier occupied at the single fork minute, and the resulting
    reference_fork came out byte-identical to reference_smart (737.31 MXN,
    82.189 km, 13 deliveries, zero diverging minutes out of 490).

    See `src.world.events.corridor_closure` for the other half of that bug:
    the blob closed the cell-centroid nodes' own access edges, so the
    pairs it touched went UNROUTABLE rather than longer, and an unroutable
    pair answers from the travel oracle's fallback constants instead of
    from the street graph.
    """
    counts: dict[str, int] = {}
    for tick in result.ticks:
        counts[tick.courier.cell] = counts.get(tick.courier.cell, 0) + 1
    return counts


def _world_events(scenario, graph) -> list[WorldEventRecord]:
    """Ground-truth event records for `build_replay`. `perceived_minute` is
    never set here -- `build_replay` computes it itself from each
    `ShiftResult`'s own `TickRecord.perceived_event_ids`, which is what
    keeps one policy's replay from ever claiming another policy's
    knowledge."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the three recorded replay files the live demo dashboard plays back.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="scenario seed")
    parser.add_argument("--date", type=Date.fromisoformat, default=rs.DEFAULT_DATE, help="scenario date, ISO format")
    parser.add_argument("--day-of-week", default="Friday", help="drives the demand/traffic day-type profile")
    parser.add_argument("--window", default="reference", choices=sorted(SHIFT_WINDOWS),
                         help="named shift window to record")
    parser.add_argument("--fork-offset-min", type=int, default=143,
                         help="minutes after shift start at which the fork's street closure starts")
    parser.add_argument("--fork-duration-min", type=int, default=45,
                         help="how long the fork's street closure stays in place")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="output directory for the replay files")
    parser.add_argument("--no-oracle-cache", action="store_true",
                         help="rebuild the travel matrix instead of reusing the pickle cache")
    args = parser.parse_args(argv)

    window = SHIFT_WINDOWS[args.window]
    fork_minute_abs = window.start_min + args.fork_offset_min
    if not (window.start_min <= fork_minute_abs < window.end_min):
        raise SystemExit(
            f"--fork-offset-min {args.fork_offset_min} places the closure at absolute minute "
            f"{fork_minute_abs}, outside the shift window {window.start_min}-{window.end_min}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("Building the travel oracle once ...", flush=True)
    base_oracle = rs.load_base_oracle(use_cache=not args.no_oracle_cache)
    graph = base_oracle.matrix.graph

    print("Bootstrapping the agent's own OSM travel skeleton ...", flush=True)
    skeleton = rs.load_travel_skeleton()

    # See run_shift.main: events.py re-parses the ~120MB graph fixture from
    # disk on every call unless handed the copy already in memory.
    original_loader = events_mod._try_load_graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph

    try:
        print(
            f"Building the reference scenario (seed {args.seed}, {window.label_with_hours}) ..."
        )
        built = rs.build_scenario(args.seed, args.date, window.start_min, window.end_min, args.day_of_week)
        print(
            f"  {len(built.scenario.order_stream)} orders, "
            f"{len(built.scenario.events_timeline)} ground-truth events, "
            f"{len(built.scenario.supply_timeline)} supply ticks"
        )

        print("Running SmartPolicy on the reference scenario ...")
        smart_outcome, _smart_sources = rs.run_one(rs.POLICY_FACTORIES["smart"](), built, base_oracle, window, graph, skeleton)
        print(
            f"  {smart_outcome.metrics.mxn_per_hour:.1f} MXN/h, "
            f"{smart_outcome.metrics.deliveries} deliveries, "
            f"{smart_outcome.metrics.km_traveled:.1f} km"
        )

        print("Running FixedPayoutThresholdPolicy on the SAME reference scenario ...")
        fixed_outcome, _fixed_sources = rs.run_one(rs.POLICY_FACTORIES["fixed_threshold"](), built, base_oracle, window, graph, skeleton)
        print(
            f"  {fixed_outcome.metrics.mxn_per_hour:.1f} MXN/h, "
            f"{fixed_outcome.metrics.deliveries} deliveries, "
            f"{fixed_outcome.metrics.km_traveled:.1f} km"
        )

        km_savings = 1.0 - (smart_outcome.metrics.km_traveled / fixed_outcome.metrics.km_traveled) \
            if fixed_outcome.metrics.km_traveled else 0.0
        print(
            f"  -> smart earns {smart_outcome.metrics.mxn_per_hour / fixed_outcome.metrics.mxn_per_hour * 100:.0f}% "
            f"of fixed_threshold's MXN/h on {km_savings * 100:.0f}% less driving"
        )

        corridor = events_mod.corridor_from_occupancy(_cell_minutes(smart_outcome.result))
        busiest_minutes = _cell_minutes(smart_outcome.result)[corridor[0]]
        print(
            f"Forking the scenario: injecting a street closure at absolute minute {fork_minute_abs} "
            f"(shift start + {args.fork_offset_min} min) across the courier's OWN corridor -- "
            f"{len(corridor)} cells, busiest one occupied {busiest_minutes} of "
            f"{window.end_min - window.start_min} shift minutes ..."
        )
        closure_event = events_mod.corridor_closure(
            "fork-street_closure",
            fork_minute_abs,
            args.fork_duration_min,
            corridor,
            base_oracle.matrix,
        )
        print(f"  resolved to {len(closure_event.edges or ())} real graph edge(s) closed")

        fork_events = sorted(
            list(built.scenario.events_timeline) + [closure_event], key=lambda e: e.start_min
        )
        # `model_copy` shallow-copies every field not named in `update`, so
        # `order_stream` and `supply_timeline` remain the SAME list objects
        # `built.platform` was already constructed against -- the fork run
        # sees the exact same world as reference_smart, plus one more event.
        fork_scenario = built.scenario.model_copy(update={"events_timeline": fork_events})
        fork_built = rs.BuiltScenario(scenario=fork_scenario, platform=built.platform)

        print("Running SmartPolicy on the forked (closure) scenario ...")
        fork_outcome, _fork_sources = rs.run_one(rs.POLICY_FACTORIES["smart"](), fork_built, base_oracle, window, graph, skeleton)
        print(
            f"  {fork_outcome.metrics.mxn_per_hour:.1f} MXN/h, "
            f"{fork_outcome.metrics.deliveries} deliveries, "
            f"{fork_outcome.metrics.km_traveled:.1f} km"
        )

        # Sanity check the "identical up to the fork minute" claim the demo
        # rests on: every tick strictly before the closure activates must
        # match between reference_smart and reference_fork, minute for
        # minute. A mismatch here means the two scenarios silently diverged
        # somewhere upstream of the closure and the fork story is a lie.
        smart_ticks = {t.minute: t for t in smart_outcome.result.ticks}
        fork_ticks = {t.minute: t for t in fork_outcome.result.ticks}
        mismatches = []
        for minute in range(window.start_min, fork_minute_abs):
            a, b = smart_ticks.get(minute), fork_ticks.get(minute)
            if a is None or b is None:
                continue
            if (a.courier.lat, a.courier.lon, a.courier.earnings_mxn, a.courier.km_traveled) != (
                b.courier.lat, b.courier.lon, b.courier.earnings_mxn, b.courier.km_traveled
            ):
                mismatches.append(minute)
        if mismatches:
            print(
                f"  WARNING: reference_smart and reference_fork diverge BEFORE the fork minute "
                f"at {len(mismatches)} tick(s), first at minute {mismatches[0]} -- the fork is not hermetic."
            )
        else:
            print(f"  verified: identical to reference_smart for all {fork_minute_abs - window.start_min} pre-fork minutes")

        # The other half of the claim: the closure must actually have CHANGED
        # something. A pre-fork match plus zero post-fork divergence would
        # mean the injected closure never touched the courier's real route
        # (the silent failure mode described in the module docstring).
        post_fork_diff = any(
            (a.courier.lat, a.courier.lon, a.courier.earnings_mxn, a.courier.km_traveled)
            != (b.courier.lat, b.courier.lon, b.courier.earnings_mxn, b.courier.km_traveled)
            for m in range(fork_minute_abs, min(smart_outcome.result.shift_end_min, max(smart_ticks, default=fork_minute_abs)) + 1)
            for a, b in [(smart_ticks.get(m), fork_ticks.get(m))]
            if a is not None and b is not None
        )
        if post_fork_diff:
            print("  verified: reference_fork visibly diverges from reference_smart after the fork minute")
        else:
            # This was a warning once. It fired, it scrolled past in the
            # build log, and a replay set whose fork changed nothing
            # shipped anyway. A demo artifact that cannot support the claim
            # it is built to make is not a valid artifact, so refuse to
            # write it.
            raise SystemExit(
                "reference_fork is IDENTICAL to reference_smart for the whole shift: the injected "
                "closure had no effect, so the fork button would show two identical runs. Refusing "
                "to write the replay set. Check that the closure resolved real graph edges and that "
                "those edges lie on the courier's own corridor, then retry with a different "
                "--fork-offset-min or --seed."
            )

        # ------------------------------------------------------------------
        # Assemble and write the three replay documents.
        # ------------------------------------------------------------------
        outputs = [
            ("reference_smart.json", smart_outcome.result, built.scenario, ["smart"]),
            ("reference_fixed.json", fixed_outcome.result, built.scenario, ["fixed_threshold"]),
            ("reference_fork.json", fork_outcome.result, fork_scenario, ["smart"]),
        ]
        for filename, result, scenario, courier_ids in outputs:
            document = build_replay(result, world_events=_world_events(scenario, graph), courier_ids=courier_ids)
            path = args.out_dir / filename
            size_bytes = write_replay(path, document)
            size_mb = size_bytes / (1024 * 1024)
            flag = "  <-- exceeds ~2 MB budget, see build brief" if size_bytes > FILE_SIZE_WARN_BYTES else ""
            print(f"wrote {path} ({size_mb:.2f} MB, {size_bytes} bytes){flag}")
    finally:
        events_mod._try_load_graph = original_loader

    print(f"\nTotal wall time: {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
