"""Run one or more policies through one or more real shifts, end to end.

This is the wiring: the REAL adapters (`src.platform.PlatformAdapter`,
`src.enrichment.RawSourceAdapter`) over a fully-populated `Scenario` built
from the real `src.world` producers, driven by `src.engine.run_shift`.
`src.engine.stubs` is deliberately not imported — the stubs exist so the
engine can run standalone before the adapters land, not so a demo runs on
them.

    .venv\\Scripts\\python.exe scripts/run_shift.py --window reference
    .venv\\Scripts\\python.exe scripts/run_shift.py --window all --seeds 42,7,13
    .venv\\Scripts\\python.exe scripts/run_shift.py --seeds 42,7,13 --diagnose --probe
    .venv\\Scripts\\python.exe scripts/run_shift.py --start 840 --end 1320 --trace 10
    .venv\\Scripts\\python.exe scripts/run_shift.py --cold-start 3 --seeds 42,7,13

THE COLD-START CURVE (`--cold-start N`) is a different question from the
comparison table above, and the stronger one. Instead of running each policy
once, it runs the smart policy through N CONSECUTIVE shifts as the same
courier: a fresh scenario each time (a different day in the same city — new
orders, new weather, new events) with the agent's `CourierHistory` carried
across and `refit()` called between shifts. Nothing else survives.

It runs a CONTROL arm alongside, over the identical days in the identical
order, with the history thrown away between shifts. That arm is necessary
rather than tidy: each shift is a different day, so a rising line on its own
cannot be told apart from day three simply being busier. Only the difference
between the two arms is attributable to what the agent learned.

Shift one should be visibly worse than shift three, and if it is not that is
a finding rather than a success: it would mean something city-specific leaked
into the agent's priors, because an agent that starts out already calibrated
never learned anything. See `Docs/architecture/AGENT_MODEL.md` section 6.

The first run pays ~120 s to build the engine's travel matrix and another
~120 s to bootstrap the agent's own OSM travel skeleton, both cached under
`cache/` (gitignored); later runs start in about a second, and
`--no-oracle-cache` forces a rebuild of the engine's.

TWO WIRING FACTS THAT COST REAL HOURS, STATED UP FRONT:

1. ONE ORDER LIST, THREADED INTO BOTH SIDES. `run_shift` resolves an
   accepted order's ground truth (restaurant location, prep time, fare,
   surge, tip) out of `scenario.order_stream`, keyed by the minute the
   platform surfaced the card — NEVER out of the `OfferCard`, which
   deliberately omits all of it. So the exact same list object must go into
   `Scenario(order_stream=...)` AND into `PlatformAdapter(order_stream=...)`.
   Pass two different collections and every ACCEPT fails its lookup, which
   used to look exactly like "the courier never leaves IDLE, and every
   number is zero". The engine now raises loudly on it instead; see
   `build_scenario` below, which returns one list and hands it to both.

2. THE TRAVEL ORACLE IS BUILT ONCE. `NetworkTravelOracle.from_fixtures` runs
   one Dijkstra per operating cell over the full Monterrey drive graph —
   about 150 seconds. Nothing in it depends on the seed, the date or the
   window, so a sweep pays it exactly once and `fork()`s a cheap,
   independent copy per run (the fork matters: street closures mutate the
   matrix in place and are never reopened, so a shared oracle would leak
   one run's closures into the next and silently destroy the A/B
   comparison).

Everything printed is derived from `src.eval.metrics.compute_metrics` plus
`src.engine.self_check`, so the table cannot drift away from the evaluation
layer's own definitions.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.agent import AcceptAllPolicy, FixedPayoutThresholdPolicy, SmartPolicy  # noqa: E402
from src.core.ports import Action, Policy, ShiftResult  # noqa: E402
from src.engine import NetworkTravelOracle, run_shift, self_check  # noqa: E402
from src.enrichment import (  # noqa: E402
    CourierHistory,
    RawSourceAdapter,
    TravelSkeleton,
    build_travel_skeleton,
    poi_coordinates,
)
from src.eval.metrics import ShiftMetrics, compute_metrics  # noqa: E402
from src.eval.runner import SHIFT_WINDOWS, ShiftWindow  # noqa: E402
from src.platform import PlatformAdapter  # noqa: E402
from src.world import demand as demand_mod  # noqa: E402
from src.world import events as events_mod  # noqa: E402
from src.world import surge as surge_mod  # noqa: E402
from src.world import weather as weather_mod  # noqa: E402
from src.world.demand import build_order_stream  # noqa: E402
from src.world.events import build_events_timeline  # noqa: E402
from src.world.scenario import Scenario, rng_streams  # noqa: E402
from src.world.timeline import Event  # noqa: E402
from src.world.traffic import build_traffic_timeline  # noqa: E402
from src.world.weather import build_weather_timeline  # noqa: E402

DEFAULT_DATE = Date(2026, 7, 10)  # a Friday — the reference scenario's date
DEFAULT_SEEDS = (42,)
DEFAULT_POLICIES = ("accept_all", "fixed_threshold", "smart")

POLICY_FACTORIES = {
    "accept_all": AcceptAllPolicy,
    # The real bar. `nearest_first` used to sit here and was removed, not
    # renamed: with one offer per minute it picked the nearest of one card,
    # so its every row was byte-identical to `accept_all`. See
    # `src.agent.baseline` for why a payout floor is the right shape of
    # baseline for a flow.
    "fixed_threshold": FixedPayoutThresholdPolicy,
    "smart": SmartPolicy,
}

# Pickled `TravelMatrix`. Purely a developer-loop cache for the ~120 s
# `from_fixtures` build — derived entirely from `fixtures/monterrey_graph.
# graphml` + `fixtures/cells.parquet`, so deleting it only costs time.
ORACLE_CACHE_PATH = PROJECT_ROOT / "cache" / "travel_matrix_oracle.pkl"


def load_base_oracle(use_cache: bool = True) -> NetworkTravelOracle:
    """Build (or reload) the one expensive travel oracle for the process."""
    if use_cache and ORACLE_CACHE_PATH.exists():
        started = time.time()
        with ORACLE_CACHE_PATH.open("rb") as handle:
            matrix = pickle.load(handle)
        oracle = NetworkTravelOracle(matrix=matrix, traffic_by_minute={})
        print("  ... loaded from cache in %.0f s" % (time.time() - started), flush=True)
        return oracle
    started = time.time()
    oracle = NetworkTravelOracle.from_fixtures([])
    print("  ... built in %.0f s" % (time.time() - started), flush=True)
    if use_cache:
        ORACLE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ORACLE_CACHE_PATH.open("wb") as handle:
            pickle.dump(oracle.matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return oracle


class FreeSlotProbePolicy:
    """Measurement instrument, not a competitor.

    Accepts the first offer it is shown but ONLY while genuinely idle, so it
    never fills its queue and the app never stops offering. Polled every
    minute from a realistic working trajectory, its `offers_seen` per hour is
    the platform's true push rate for this calibration — the number the
    brief's ~7.5 offers per courier-hour anchor is about.
    """

    name = "probe_free_slot"

    def decide(self, view, sources, courier):
        from src.core.ports import Action, Decision, DecisionTrace

        if view.offers and courier.activity.value == "idle":
            chosen = view.offers[0]
            return Decision(
                action=Action.ACCEPT,
                order_id=chosen.order_id,
                target_cell=None,
                trace=DecisionTrace(view.minute, (), chosen.order_id, 0.0, "none",
                                    "probe: free, so take the first card"),
            )
        return Decision(
            action=Action.REJECT,
            order_id=None,
            target_cell=None,
            trace=DecisionTrace(view.minute, (), None, 0.0, "none",
                                "probe: busy, keeping the slot open"),
        )


# ---------------------------------------------------------------------------
# Scenario assembly
# ---------------------------------------------------------------------------


def _city_wide_event_multipliers(
    events: list[Event], minutes: range
) -> tuple[list[float], list[float]]:
    """Per-minute city-wide (demand, courier-supply) multipliers from the
    events timeline.

    `build_order_stream` accepts city-wide scalars per minute — it never
    imports `events.py` and has no per-cell event channel — so an active
    event's `demand_mult` / `courier_supply_mult` are composed
    multiplicatively across whatever is active that minute. That is a
    deliberate simplification of a spatially-scoped effect into the only
    shape the demand producer's API offers; the event's real, located
    effect still reaches the courier through the enrichment layer's
    perception of it and (for street closures) through the travel oracle.
    """
    demand_mult: list[float] = []
    supply_mult: list[float] = []
    for minute in minutes:
        d = 1.0
        s = 1.0
        for event in events:
            if event.is_active(minute):
                d *= event.demand_mult
                s *= event.courier_supply_mult
        demand_mult.append(d)
        supply_mult.append(s)
    return demand_mult, supply_mult


def _supply_timeline(
    seed: int,
    minutes: list[int],
    day_of_week: str,
    weather_demand_mult: list[float],
    event_demand_mult: list[float],
    courier_supply_mult: list[float],
):
    """Rebuild the ground-truth `SurgeField` that `build_order_stream` used
    internally, so its `supply_ticks` can be handed to the platform.

    `build_order_stream` builds this field to stamp `surge_at_spawn` on
    every order but returns only the orders, and `PlatformAdapter` needs
    the supply timeline (local competing couriers per cell per minute) to
    compute offer reach. Rebuilding it here is exact rather than
    approximate: `rng_streams(seed)` spawns each named generator fresh from
    the seed on every call, so the `competitors` stream replays identically,
    and every other input below is reconstructed from the same fixtures and
    the same multipliers. Same inputs, same stream, same field.

    (The alternative — having `build_order_stream` return the field — is a
    change to `src/world/`, which this script is not allowed to make. It
    would be the cleaner fix: see the report.)
    """
    n_min = len(minutes)
    restaurants = demand_mod._load_restaurants()
    population = demand_mod._load_population()
    workplaces = demand_mod._load_workplaces()
    rngs = rng_streams(seed)
    model = demand_mod._DemandModel(restaurants, population, workplaces, rngs["kitchen"])

    day_type = demand_mod.day_type_for(day_of_week)
    profile = np.array([demand_mod.temporal_profile(m % 1440, day_type) for m in minutes])
    combined = profile * np.asarray(weather_demand_mult) * np.asarray(event_demand_mult)
    rate = demand_mod.DEMAND_CALIBRATION["orders_per_weight_unit_per_min"]

    demand_by_cell = {cell: model.cell_weight_sum[cell] * rate * combined for cell in model.origin_cells}
    demand_full = {c: demand_by_cell.get(c, np.zeros(n_min)) for c in model.full_cell_grid}

    return surge_mod.build_supply_and_surge(
        rng=rngs["competitors"],
        cells=model.full_cell_grid,
        minutes=minutes,
        demand_by_cell=demand_full,
        courier_supply_mult=courier_supply_mult,
        baseline_weights=model.baseline_weights,
    )


@dataclass
class BuiltScenario:
    """A `Scenario` plus the platform built from THE SAME order list.

    Bundled in one object on purpose: the whole point of fact (1) in the
    module docstring is that these two must never be constructed
    independently, so they are never handed out independently either.
    """

    scenario: Scenario
    platform: PlatformAdapter
    # The surge map this scenario was built with, kept rather than dropped.
    # `Scenario` only carries `supply_timeline` (couriers per cell), and the
    # multiplier itself was recomputed nowhere and therefore unrecordable --
    # which is why the replay had no surge layer to draw. Optional so the
    # callers that assemble a `BuiltScenario` by hand (a scenario copied with
    # closures spliced in, say) keep working unchanged.
    surge_field: surge_mod.SurgeField | None = None


def build_scenario(
    seed: int,
    date: Date,
    shift_start_min: int,
    shift_end_min: int,
    day_of_week: str,
) -> BuiltScenario:
    """Build one fully-populated `Scenario` and its platform adapter.

    Producer order is a dependency chain, not a preference: weather feeds
    traffic's rain congestion and the demand/supply multipliers; events feed
    the demand/supply multipliers too; the order stream needs both; the
    supply timeline is reconstructed from the same inputs the order stream
    used.
    """
    minutes = list(range(shift_start_min, shift_end_min))

    weather_timeline = build_weather_timeline(date, shift_start_min, shift_end_min)
    traffic_timeline = build_traffic_timeline(date, shift_start_min, shift_end_min, weather_timeline)
    events_timeline = build_events_timeline(seed, date, shift_start_min, shift_end_min)

    weather_demand = [weather_mod.demand_factor(t) for t in weather_timeline]
    weather_supply = [weather_mod.courier_supply_factor(t) for t in weather_timeline]
    event_demand, event_supply = _city_wide_event_multipliers(events_timeline, range(shift_start_min, shift_end_min))
    courier_supply = [w * e for w, e in zip(weather_supply, event_supply)]

    order_stream = build_order_stream(
        seed,
        date,
        shift_start_min,
        shift_end_min,
        day_of_week=day_of_week,
        weather_mult=weather_demand,
        event_mult=event_demand,
        courier_supply_mult=courier_supply,
    )

    surge_field = _supply_timeline(
        seed, minutes, day_of_week, weather_demand, event_demand, courier_supply
    )

    scenario = Scenario(
        seed=seed,
        date=date,
        day_of_week=day_of_week,
        shift_start_min=shift_start_min,
        shift_end_min=shift_end_min,
        weather_timeline=weather_timeline,
        traffic_timeline=traffic_timeline,
        events_timeline=events_timeline,
        supply_timeline=surge_field.supply_ticks,
        order_stream=order_stream,
    )

    # THE SAME LIST, both sides. `scenario.order_stream` is what an ACCEPT
    # is resolved against; this is what the offer cards are projected from.
    platform = PlatformAdapter(
        order_stream=scenario.order_stream,
        supply_timeline=scenario.supply_timeline,
        scenario_seed=seed,
    )
    return BuiltScenario(scenario=scenario, platform=platform, surge_field=surge_field)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _fuel_stops(result: ShiftResult) -> int:
    return sum(
        1
        for tick in result.ticks
        if tick.decision is not None and tick.decision.action == Action.REFUEL
    )


@dataclass
class RunOutcome:
    metrics: ShiftMetrics
    result: ShiftResult
    window: ShiftWindow
    self_check_error: str | None

    @property
    def offers_per_hour(self) -> float:
        hours = self.result.minutes_elapsed / 60.0
        return self.result.offers_seen / hours if hours else 0.0

    @property
    def fuel_stops(self) -> int:
        return _fuel_stops(self.result)


def load_travel_skeleton() -> TravelSkeleton:
    """The agent's OWN cell-to-cell free-flow matrix, bootstrapped from OSM.

    Pure geometry — no seed, no date, no window enters it — so the whole
    sweep shares one instance and the disk cache means only the very first
    run in a checkout pays for building it. It is a DIFFERENT object over
    DIFFERENT data from the engine's `NetworkTravelOracle`, which carries
    live congestion and street closures; see `src.enrichment.osm_travel`.
    """
    started = time.time()
    skeleton = build_travel_skeleton(*poi_coordinates())
    print(
        "  ... agent travel skeleton ready in %.0f s (%d cells)"
        % (time.time() - started, len(skeleton.cells())),
        flush=True,
    )
    return skeleton


def run_one(
    policy: Policy,
    built: BuiltScenario,
    base_oracle: NetworkTravelOracle,
    window: ShiftWindow,
    graph,
    skeleton: TravelSkeleton,
    history: CourierHistory | None = None,
) -> tuple[RunOutcome, RawSourceAdapter]:
    """One policy, one scenario, one shift.

    A fresh `RawSourceAdapter` per run is mandatory, not tidiness: the
    courier's history is stateful and per-courier, so reusing one across
    policies would hand the second policy everything the first one learned.
    The platform, by contrast, is a pure read and is shared, and the travel
    skeleton is pure geometry and is shared too.

    `history` is the one thing a caller may deliberately carry across runs —
    that is what makes the cold-start curve a curve. Left out, this is a
    courier who has never worked the city.

    The adapter is returned alongside the outcome so the caller can `refit()`
    it between shifts.
    """
    sources = RawSourceAdapter(
        built.scenario, graph=graph, history=history, skeleton=skeleton
    )
    travel = base_oracle.fork(built.scenario.traffic_timeline)
    result = run_shift(
        built.scenario,
        policy,
        built.platform,
        sources,
        travel=travel,
    )
    error: str | None = None
    try:
        self_check(result)
    except AssertionError as exc:
        error = str(exc)
    outcome = RunOutcome(
        metrics=compute_metrics(result), result=result, window=window, self_check_error=error
    )
    return outcome, sources


# ---------------------------------------------------------------------------
# The cold-start curve
# ---------------------------------------------------------------------------

# Offset between one courier-run's consecutive shift seeds. Each shift has to
# be a DIFFERENT day — a new order stream, new weather, new events — or the
# agent would be re-running the same day and "learning" would just be
# memorising it. Same city, different day, same courier.
COLD_START_SEED_STRIDE = 1000


@dataclass
class ColdStartRun:
    """One courier's N consecutive shifts in a city they arrived in cold."""

    seed: int
    mxn_per_hour: tuple[float, ...]
    deliveries: tuple[int, ...]
    km: tuple[float, ...]
    fit_reports: tuple[dict[str, int], ...]


@dataclass
class ColdStartReport:
    """Both arms of the experiment, over the identical sequence of days.

    A curve on its own proves nothing, and this is the trap the first
    version of this measurement fell into. Each shift is a DIFFERENT day, so
    a rising line can just mean day three was busier than day one — and
    measured that way the per-seed lines ranged from 90 -> 88 -> 155 to
    104 -> 80 -> 48, which is not a learning curve, it is weather.

    So there is a control: the same seeds, the same days, in the same order,
    with the agent's history THROWN AWAY between shifts. That arm measures
    day difficulty and nothing else. The difference between the two arms is
    the only thing attributable to what the agent learned.
    """

    learning: list[ColdStartRun]
    control: list[ColdStartRun]


def run_cold_start(
    policy_name: str,
    seeds: tuple[int, ...],
    shifts: int,
    date: Date,
    window: ShiftWindow,
    day_of_week: str,
    base_oracle: NetworkTravelOracle,
    graph,
    skeleton: TravelSkeleton,
) -> ColdStartReport:
    """Run each seed as one courier working `shifts` consecutive days.

    The ONLY thing carried from one shift to the next is the
    `CourierHistory`: kitchen memory, the trip log, and the tables fitted
    off it. A fresh policy object each shift makes that explicit — none of
    the learning lives in the policy, all of it lives in what the agent
    measured and re-fitted.

    `refit()` runs BETWEEN shifts, never inside one, which is the whole
    online/offline split: fitting is expensive and happens on the agent's
    own time; using a fitted table is a dict lookup inside seven seconds.
    """
    factory = POLICY_FACTORIES.get(policy_name, FreeSlotProbePolicy)
    learning: list[ColdStartRun] = []
    control: list[ColdStartRun] = []
    for seed in seeds:
        for arm, runs in (("learning", learning), ("control", control)):
            # The learning arm carries one history across every shift. The
            # control arm gets a brand-new courier each time, over the exact
            # same days in the same order, so its line IS the day-difficulty
            # profile and nothing else.
            history = CourierHistory() if arm == "learning" else None
            rates: list[float] = []
            deliveries: list[int] = []
            km: list[float] = []
            reports: list[dict[str, int]] = []
            for shift in range(shifts):
                shift_seed = seed + shift * COLD_START_SEED_STRIDE
                built = build_scenario(
                    shift_seed, date, window.start_min, window.end_min, day_of_week
                )
                outcome, sources = run_one(
                    factory(),
                    built,
                    base_oracle,
                    window,
                    graph,
                    skeleton,
                    history=history if arm == "learning" else CourierHistory(),
                )
                rates.append(outcome.metrics.mxn_per_hour)
                deliveries.append(outcome.metrics.deliveries)
                km.append(outcome.metrics.km_traveled)
                print(
                    "  seed %d %-8s shift %d (scenario seed %d): %.1f MXN/h, "
                    "%d deliveries, %.0f km"
                    % (
                        seed,
                        arm,
                        shift + 1,
                        shift_seed,
                        outcome.metrics.mxn_per_hour,
                        outcome.metrics.deliveries,
                        outcome.metrics.km_traveled,
                    ),
                    flush=True,
                )
                # Between shifts, never inside one. Harmless on the control
                # arm, whose history is discarded either way.
                reports.append(sources.refit())
            runs.append(
                ColdStartRun(
                    seed=seed,
                    mxn_per_hour=tuple(rates),
                    deliveries=tuple(deliveries),
                    km=tuple(km),
                    fit_reports=tuple(reports),
                )
            )
    return ColdStartReport(learning=learning, control=control)


def _arm_means(runs: list[ColdStartRun], shifts: int) -> list[float]:
    return [
        _mean([run.mxn_per_hour[i] for run in runs if i < len(run.mxn_per_hour)])
        for i in range(shifts)
    ]


def _arm_block(label: str, runs: list[ColdStartRun], shifts: int, width: int) -> list[str]:
    lines = [label]
    for run in runs:
        lines.append(
            "  ".join(
                ["%-10s" % ("seed %d" % run.seed)]
                + ["%9.1f" % rate for rate in run.mxn_per_hour]
            )
        )
    lines.append("-" * width)
    lines.append(
        "  ".join(["%-10s" % "MEAN"] + ["%9.1f" % v for v in _arm_means(runs, shifts)])
    )
    return lines


def render_cold_start(report: ColdStartReport, shifts: int) -> str:
    """Both arms, then the difference, which is the only honest number here.

    Reported per seed as well as on the mean because one seed climbing is a
    lucky draw. And reported against a control because each shift is a
    different day: without it a rising line cannot be told apart from a
    busier Friday.
    """
    header = "  ".join(
        ["%-10s" % "seed"] + ["%9s" % ("shift %d" % (i + 1)) for i in range(shifts)]
    )
    width = len(header)
    lines: list[str] = [header, "-" * width]
    lines += _arm_block(
        "MXN/h, history PERSISTING between shifts (refit between each):",
        report.learning,
        shifts,
        width,
    )
    lines.append("")
    lines += _arm_block(
        "MXN/h, CONTROL: same days, same order, history thrown away each shift:",
        report.control,
        shifts,
        width,
    )

    learned = _arm_means(report.learning, shifts)
    controlled = _arm_means(report.control, shifts)
    lines.append("")
    lines.append("Learning minus control, per shift (the part attributable to memory):")
    lines.append(
        "  ".join(
            ["%-10s" % "DELTA"]
            + ["%+9.1f" % (learned[i] - controlled[i]) for i in range(shifts)]
        )
    )
    if shifts >= 2:
        first = learned[0] - controlled[0]
        last = learned[-1] - controlled[-1]
        lines.append("")
        lines.append(
            "shift 1 %+.1f MXN/h -> shift %d %+.1f MXN/h against the same day run cold"
            % (first, shifts, last)
        )
        ahead = sum(
            1
            for learn, ctrl in zip(report.learning, report.control)
            if learn.mxn_per_hour[-1] > ctrl.mxn_per_hour[-1]
        )
        lines.append(
            "  seeds where the experienced courier beat the cold one on the last day: "
            "%d of %d" % (ahead, len(report.learning))
        )

    lines.append("")
    lines.append("What the agent had evidence for, after each shift (learning arm):")
    for run in report.learning:
        for i, fit in enumerate(run.fit_reports):
            lines.append(
                "  seed %d after shift %d: %s"
                % (run.seed, i + 1, ", ".join("%s=%d" % kv for kv in sorted(fit.items())))
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HEADERS = (
    ("policy", 15, "<"),
    ("MXN/h", 8, ">"),
    ("deliv", 6, ">"),
    ("deliv/h", 8, ">"),
    ("offers", 7, ">"),
    ("offers/h", 9, ">"),
    ("accept", 7, ">"),
    ("km", 7, ">"),
    ("unpaid km", 10, ">"),
    ("idle", 6, ">"),
    ("fuel", 5, ">"),
    ("home km", 8, ">"),
    ("check", 6, ">"),
)


def _row_cells(outcome: RunOutcome) -> list[str]:
    m = outcome.metrics
    home = m.end_of_shift_distance_from_home_km
    return [
        m.policy_name,
        "%.1f" % m.mxn_per_hour,
        "%d" % m.deliveries,
        "%.2f" % m.deliveries_per_hour,
        "%d" % m.offers_seen,
        "%.1f" % outcome.offers_per_hour,
        "%.0f%%" % (m.acceptance_rate * 100.0),
        "%.1f" % m.km_traveled,
        "%.1f" % m.unpaid_km,
        "%.0f%%" % (m.idle_fraction * 100.0),
        "%d" % outcome.fuel_stops,
        "n/a" % () if home is None else "%.2f" % home,
        "PASS" if outcome.self_check_error is None else "FAIL",
    ]


def render_table(rows: list[list[str]]) -> str:
    header = "  ".join(f"{label:{align}{width}}" for label, width, align in _HEADERS)
    lines = [header, "-" * len(header)]
    for cells in rows:
        lines.append(
            "  ".join(f"{cell:{align}{width}}" for cell, (_, width, align) in zip(cells, _HEADERS))
        )
    return "\n".join(lines)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_cycle_diagnosis(outcome: RunOutcome) -> str:
    """Where the minutes of one delivery cycle actually go.

    Throughput is the metric everything else hangs off, so when it is
    capped it is worth naming the component that caps it rather than
    asserting that one does.
    """
    deliveries = outcome.result.deliveries
    if not deliveries:
        return "  %s: no deliveries to decompose." % outcome.metrics.policy_name

    cycles = [float(d.delivered_at_min - d.accepted_at_min) for d in deliveries]
    kitchen = [d.kitchen_wait_minutes for d in deliveries]
    trip = [d.minutes for d in deliveries]
    # Whatever the accept-to-delivered span is not kitchen wait and not the
    # pickup->customer leg: the ride out to the restaurant plus both
    # handling stops.
    other = [c - k - t for c, k, t in zip(cycles, kitchen, trip)]
    km = [d.km for d in deliveries]

    idle_per_delivery = outcome.result.minutes_idle / len(deliveries)
    return (
        "  %-15s cycle %.1f min = kitchen wait %.1f + ride to customer %.1f + "
        "ride to restaurant & handling %.1f | idle %.1f min/delivery | trip %.2f km"
        % (
            outcome.metrics.policy_name,
            _mean(cycles),
            _mean(kitchen),
            _mean(trip),
            _mean(other),
            idle_per_delivery,
            _mean(km),
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.replace(" ", "").split(",") if part)


def _parse_policies(raw: str) -> tuple[str, ...]:
    names = tuple(part for part in raw.replace(" ", "").split(",") if part)
    unknown = [n for n in names if n not in POLICY_FACTORIES]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown policy %s; choose from %s" % (unknown, sorted(POLICY_FACTORIES))
        )
    return names


def _windows_for(args: argparse.Namespace) -> list[ShiftWindow]:
    """The four named windows, or one explicit start/end pair.

    The Night window runs 1080-1560 and crosses midnight. Minutes are a
    continuous counter from the scenario's reference start rather than a
    clock that resets, so `end_min > start_min` always holds and no shift
    needs wraparound arithmetic to run. The wrap is handled deliberately in
    exactly two places, both of them display or lookup rather than
    duration: `ShiftWindow.wall_clock` formats 1560 as "02:00" instead of
    "26:00", and every world producer indexes its time-of-day profiles by
    `minute % 1440`.
    """
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise SystemExit("--start and --end must be given together")
        if args.end <= args.start:
            raise SystemExit("--end must be strictly after --start")
        return [ShiftWindow("Custom", args.start, args.end)]
    if args.window == "all":
        return list(SHIFT_WINDOWS.values())
    return [SHIFT_WINDOWS[args.window]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run policies through real shifts with the real adapters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS,
                        help="comma-separated scenario seeds, e.g. 42,7,13")
    parser.add_argument("--date", type=Date.fromisoformat, default=DEFAULT_DATE,
                        help="scenario date, ISO format")
    parser.add_argument("--day-of-week", default="Friday",
                        help="drives the demand/traffic day-type profile")
    parser.add_argument("--window", default="reference",
                        choices=sorted(SHIFT_WINDOWS) + ["all"],
                        help="named shift window (night crosses midnight)")
    parser.add_argument("--start", type=int, default=None,
                        help="explicit shift start minute; overrides --window")
    parser.add_argument("--end", type=int, default=None,
                        help="explicit shift end minute; overrides --window")
    parser.add_argument("--policies", type=_parse_policies, default=DEFAULT_POLICIES,
                        help="comma-separated policy names")
    parser.add_argument("--baseline", default=None,
                        help="policy the smart margin is reported against "
                             "(default: the best-earning non-smart policy)")
    parser.add_argument("--diagnose", action="store_true",
                        help="also print where each delivery cycle's minutes go")
    parser.add_argument("--probe", action="store_true",
                        help="also run the free-slot probe, which measures the "
                             "platform's true offers-per-courier-hour push rate")
    parser.add_argument("--trace", type=int, default=0, metavar="N",
                        help="print the first N decision summaries per policy")
    parser.add_argument("--no-oracle-cache", action="store_true",
                        help="rebuild the travel matrix instead of reusing the pickle cache")
    parser.add_argument("--cold-start", type=int, default=0, metavar="N",
                        help="instead of the comparison table, run N consecutive "
                             "shifts per seed with the agent's history persisting "
                             "and refit() between them, and print MXN/h per shift")
    parser.add_argument("--cold-start-policy", default="smart",
                        help="policy the cold-start curve is measured on")
    args = parser.parse_args(argv)

    policy_names = list(args.policies)
    if args.probe:
        policy_names.append("probe_free_slot")

    windows = _windows_for(args)

    started = time.time()
    print("Building the travel oracle once (one Dijkstra per operating cell) ...", flush=True)
    # Any traffic timeline will do here: `fork` re-indexes the real one per
    # run, and this base oracle never runs a shift itself (a base that had
    # applied a closure would hand that closure to every fork).
    base_oracle = load_base_oracle(use_cache=not args.no_oracle_cache)
    graph = base_oracle.matrix.graph
    print("Bootstrapping the agent's own OSM travel skeleton ...", flush=True)
    skeleton = load_travel_skeleton()

    # `build_events_timeline` re-parses the ~120MB drive graph from disk on
    # every call (~20 s). It only ever reads from whatever `_try_load_graph`
    # returns, so handing it the copy already in memory is the same graph,
    # not a different one — the same caching `tests/conftest.py` does.
    original_loader = events_mod._try_load_graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph

    try:
        if args.cold_start:
            for window in windows:
                print()
                print(
                    "=== %s (minutes %d-%d) ==="
                    % (window.label_with_hours, window.start_min, window.end_min)
                )
                print(
                    render_cold_start(
                        run_cold_start(
                            args.cold_start_policy,
                            args.seeds,
                            args.cold_start,
                            args.date,
                            window,
                            args.day_of_week,
                            base_oracle,
                            graph,
                            skeleton,
                        ),
                        args.cold_start,
                    )
                )
            print()
            print("Total wall time: %.0f s" % (time.time() - started))
            return 0

        all_outcomes: dict[tuple[str, int], list[RunOutcome]] = {}
        for window in windows:
            print()
            print("=== %s (minutes %d-%d) ===" % (window.label_with_hours, window.start_min, window.end_min))
            rows: list[list[str]] = []
            diagnostics: list[str] = []
            for seed in args.seeds:
                built = build_scenario(
                    seed, args.date, window.start_min, window.end_min, args.day_of_week
                )
                print(
                    "seed %d: %d orders, %d events, %d supply ticks"
                    % (
                        seed,
                        len(built.scenario.order_stream),
                        len(built.scenario.events_timeline),
                        len(built.scenario.supply_timeline),
                    )
                )
                for name in policy_names:
                    factory = POLICY_FACTORIES.get(name, FreeSlotProbePolicy)
                    policy = factory()
                    outcome, _sources = run_one(
                        policy, built, base_oracle, window, graph, skeleton
                    )
                    all_outcomes.setdefault((window.label, seed), []).append(outcome)
                    rows.append(["seed %d %s" % (seed, name)] + _row_cells(outcome)[1:])
                    if args.diagnose:
                        diagnostics.append(render_cycle_diagnosis(outcome))
                    if args.trace:
                        _print_traces(outcome, args.trace)
            print()
            print(render_table(rows))
            if diagnostics:
                print()
                print("Where a delivery cycle's minutes go:")
                for line in diagnostics:
                    print(line)
            _print_failures(all_outcomes, window.label, args.seeds)
            _print_margin(all_outcomes, window.label, args.seeds, args.baseline)
    finally:
        events_mod._try_load_graph = original_loader

    print()
    print("Total wall time: %.0f s" % (time.time() - started))
    return 0


def _print_traces(outcome: RunOutcome, limit: int) -> None:
    """The policy's own words, in order. The cheapest way to see WHY a
    policy is doing something odd: the reasoning is returned in the trace,
    not reconstructed afterwards."""
    print()
    print("first %d decisions by %s:" % (limit, outcome.metrics.policy_name))
    shown = 0
    for tick in outcome.result.ticks:
        if tick.decision is None:
            continue
        trace = tick.decision.trace
        print("  min %d [%s] offers=%d bar=%.0f %s"
              % (tick.minute, tick.decision.action.value, len(trace.considered),
                 trace.threshold_mxn_per_hour, trace.summary))
        for evaluation in trace.considered[:3]:
            print("      %s: %.0f MXN net over %.0f min = %.0f MXN/h%s"
                  % (evaluation.order_id, evaluation.expected_net_mxn,
                     evaluation.expected_minutes, evaluation.expected_mxn_per_hour,
                     "" if evaluation.rejected_because is None
                     else " -- " + evaluation.rejected_because))
        shown += 1
        if shown >= limit:
            break


def _print_failures(
    all_outcomes: dict[tuple[str, int], list[RunOutcome]], label: str, seeds: tuple[int, ...]
) -> None:
    failures = [
        (seed, o)
        for seed in seeds
        for o in all_outcomes.get((label, seed), [])
        if o.self_check_error is not None
    ]
    if not failures:
        return
    print()
    print("self_check failures (bounds are the honesty gate; they are not to be loosened):")
    for seed, outcome in failures:
        print("  seed %d %s: %s" % (seed, outcome.metrics.policy_name, outcome.self_check_error))


def _print_margin(
    all_outcomes: dict[tuple[str, int], list[RunOutcome]],
    label: str,
    seeds: tuple[int, ...],
    baseline_name: str | None,
) -> None:
    """The smart policy's margin over the better baseline, per seed and in
    aggregate. Reported per seed as well as on the mean on purpose: a policy
    that wins on the mean but loses on a seed is one unlucky draw away from
    losing on stage."""
    by_policy: dict[str, dict[int, float]] = {}
    for seed in seeds:
        for outcome in all_outcomes.get((label, seed), []):
            by_policy.setdefault(outcome.metrics.policy_name, {})[seed] = outcome.metrics.mxn_per_hour
    if "smart" not in by_policy:
        return
    baselines = {n: v for n, v in by_policy.items() if n != "smart"}
    if not baselines:
        return
    if baseline_name is not None:
        best_name = baseline_name
    else:
        best_name = max(baselines, key=lambda n: _mean(list(baselines[n].values())))

    smart = by_policy["smart"]
    base = baselines[best_name]
    print()
    per_seed = []
    for seed in seeds:
        if seed in smart and seed in base and base[seed]:
            per_seed.append((seed, (smart[seed] - base[seed]) / base[seed] * 100.0))
    smart_mean = _mean([smart[s] for s in seeds if s in smart])
    base_mean = _mean([base[s] for s in seeds if s in base])
    margin = (smart_mean - base_mean) / base_mean * 100.0 if base_mean else 0.0
    print(
        "smart vs %s (MXN/h): %.1f vs %.1f = %+.1f%% on the mean of %d seed(s)"
        % (best_name, smart_mean, base_mean, margin, len(seeds))
    )
    if per_seed:
        print(
            "  per seed: "
            + ", ".join("seed %d %+.1f%%" % (seed, delta) for seed, delta in per_seed)
        )
        deltas = [d for _, d in per_seed]
        print("  spread: %+.1f%% to %+.1f%%" % (min(deltas), max(deltas)))


if __name__ == "__main__":
    raise SystemExit(main())
