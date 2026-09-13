"""Re-fit the smart agent's constants to the corrected world.

WHY THIS EXISTS. The world was recalibrated against a working courier's own
figures -- short, frequent, cheap trips -- and the agent's constants were not.
They were fitted to a world of few, long, expensive trips, and that world was
an artefact. In the corrected one the agent LOSES to a flat 40 MXN payout
floor on every scenario measured. Being selective costs more than it earns
when the trips are cheap and keep coming.

WHAT IT COSTS, MEASURED, BECAUSE THE FIRST ESTIMATE WAS WRONG BY 300x.

  building a scenario        18 s calm, 57 s hostile
  one shift on a calm day     0.62 s
  one shift on a HOSTILE day  ~180 s

The last number is the one that shapes this whole script. It is not the agent
thinking; it is `TravelMatrix.close_streets`, which re-runs Dijkstra over the
Monterrey road graph for every matrix row a closure touches, five times a
shift. A hostile shift costs about as much as three hundred calm ones.

So the search runs on CALM days only, where scenarios are built once per
worker and each candidate then costs 0.62 seconds, and the hostile day is
spent only on the short-list -- enough to answer "does this collapse under
closures and 42 degrees", which is the question the hostile day is for. A
search that spent its budget on hostile days would have explored a few dozen
candidates instead of a few hundred, which is the worse trade.

HOW OVERFITTING IS KEPT OUT. With hundreds of candidates over ten dimensions,
SOMETHING will win the search set by luck. So the seeds are split: the search
never sees the hold-out seeds, and the winner is reported on both. A result
that needs its own search set is not a result -- this project has already had
to retract one headline for exactly that reason.

THE CLOSURES ARE FROZEN. `build_days.py` and `build_duel.py` place closures on
corridors measured from the policies' own probe runs, which would make the
world depend on the config being tuned. Here they are computed once, from the
DEFAULT agent and the floor, and reused for every candidate. Arbitrary, but
identical for everyone, which is the property that matters.

Usage:
    .venv\\Scripts\\python.exe scripts/tune_smart.py --stage explore
    .venv\\Scripts\\python.exe scripts/tune_smart.py --stage refine
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# What a candidate is
# ---------------------------------------------------------------------------

# Every knob here is a NUMBER THE AGENT BELIEVES OR PREFERS, never a fact
# about the world. Nothing in this file touches `src/world/`: tuning an agent
# by moving the city is not tuning an agent.
#
# (dict name, key, low, high). `None` as the dict name means a constructor
# argument rather than a calibration entry.
SEARCH_SPACE: list[tuple[str | None, str, float, float]] = [
    # How fussy to be, as a multiplier on the computed reservation price.
    # Below 1.0 deliberately accepts work the bar refuses -- which is not
    # cheating, it is paying for the offer flow the bar cannot see (see
    # SmartPolicy.__init__ on the acceptance-rate feedback it omits).
    (None, "bar_factor", 0.35, 1.15),
    # What exposure COSTS the courier: night, rain, heat, riding into a jam.
    # A preference, not a belief.
    (None, "risk_posture", 0.0, 0.9),
    # The floor under the reservation price, in MXN per hour of work.
    ("SHIFT_CALIBRATION", "base_reservation_mxn_per_hour", 6.0, 30.0),
    # Early fussiness and late desperation, as multipliers on that floor.
    ("SHIFT_CALIBRATION", "choosy_factor_early", 0.85, 1.35),
    ("SHIFT_CALIBRATION", "desperate_factor_late", 0.35, 0.95),
    # How many more chances the courier BELIEVES are coming this hour. Too
    # high and every offer looks worth refusing for the next one.
    ("RESERVATION_CALIBRATION", "prior_offers_per_hour", 3.0, 16.0),
    # What standing in a dead cell is believed to cost, in unpaid minutes.
    ("DESTINATION_CALIBRATION", "dead_minutes_at_zero_demand", 5.0, 22.0),
    # How long to stand still before riding somewhere better, and how far.
    ("REPOSITION_CALIBRATION", "min_idle_minutes_before_moving", 2.0, 16.0),
    ("REPOSITION_CALIBRATION", "max_reposition_km", 2.0, 10.0),
    ("REPOSITION_CALIBRATION", "min_gain_minutes", 1.0, 7.0),
    # What the courier believes a jam costs THEM versus the cars around them.
    # Added after measuring WHY the agent was losing: it priced every leg at
    # the full car congestion multiplier (traces showing 2.4-2.6x), so a long
    # well-paid trip looked ruinous per hour and it specialised in short cheap
    # work while the floor took the good jobs.
    ("TRAVEL_CALIBRATION", "congestion_belief_factor", 0.05, 1.0),
    # What the courier charges themselves per kilometre and per minute. Same
    # failure mode from a different direction: overcharge distance and the
    # long trips -- which is where this fare card puts the money -- stop
    # clearing the bar.
    ("ECONOMICS_CALIBRATION", "fuel_cost_per_km_mxn", 0.0, 1.6),
    ("ECONOMICS_CALIBRATION", "vehicle_wear_per_km_mxn", 0.0, 0.8),
    ("ECONOMICS_CALIBRATION", "opportunity_mxn_per_minute", 0.4, 3.0),
    # What the courier assumes a kitchen will make them wait when they have
    # never delivered from that branch. Shipped at 9.0 against a world whose
    # median prep is 17.2 minutes and whose spread is 7.7 to 31.4 -- and prep
    # correlates 0.003 with trip distance, so it is the one big cost that the
    # payout carries no information about. Underestimating it by half makes
    # every offer look better per hour than it is.
    ("HANDLING_CALIBRATION", "default_kitchen_minutes", 6.0, 26.0),
]

# Integer-valued nothing; every knob above is continuous.
PARAM_NAMES = [key for _dict, key, _lo, _hi in SEARCH_SPACE]

# What a kilometre costs the courier, from the agent's own economics. Used
# ONLY to score candidates -- the world charges fuel in time, never pesos.
# It is in the objective because "maximise earnings" without it rewards
# driving 170 km for the money.
COST_PER_KM_MXN = 1.5


@dataclass(frozen=True)
class ScenarioSpec:
    seed: int
    iso: str
    weekday: str
    hostile: bool

    @property
    def key(self) -> str:
        return f"{self.iso}:{self.seed}:{'chaos' if self.hostile else 'calm'}"


# The two days the showcase uses, so a tuned agent is tuned on the worlds it
# will actually be shown in.
CALM = ("2026-07-10", "Friday")
CHAOS = ("2026-06-18", "Thursday")

# Every real weekday the search runs on. Tuning on ONE date fits the agent to
# that date's weather and that date's order stream as surely as tuning on one
# seed fits it to one draw -- the first sweep here covered six seeds of a
# single Friday, which is six samples of the same afternoon. These four are
# the ones `build_days.py` already characterises, chosen for contrast across
# the shift window and nothing else.
DAYS: tuple[tuple[str, str], ...] = (
    ("2026-07-15", "Wednesday"),   # mild and dry, 33.2 C
    ("2026-07-10", "Friday"),      # the reference Friday, 34.3 C, trace rain
    ("2026-08-04", "Tuesday"),     # rain across 8 of 8 hours, 38.1 C
    ("2026-06-18", "Thursday"),    # extreme heat, above 40 C for three hours
)

# Seeds the search may look at, and seeds it may not. The split is fixed here
# rather than drawn, so "the hold-out" means the same thing across runs.
SEARCH_SEEDS = (42, 101, 202, 303, 404, 505)
HOLDOUT_SEEDS = (606, 707, 808, 909, 1010, 1111)

CLOSURE_PLAN: tuple[tuple[int, int], ...] = ((240, 180), (260, 170), (280, 160), (300, 150), (320, 140))
CORRIDOR_CELLS = 3


# ---------------------------------------------------------------------------
# Worker state: built once per process, reused for every candidate
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init() -> None:
    """Import the world once and load the travel oracle once, per process."""
    import scripts.run_shift as rs
    from src.world import events as events_mod
    from src.agent import calibration as cal

    oracle = rs.load_base_oracle(use_cache=True)
    graph = oracle.matrix.graph
    events_mod._try_load_graph = lambda path=events_mod.GRAPH_FIXTURE_PATH: graph

    _W["rs"] = rs
    _W["events"] = events_mod
    _W["cal"] = cal
    _W["oracle"] = oracle
    _W["graph"] = graph
    _W["skeleton"] = rs.load_travel_skeleton()
    _W["window"] = rs.SHIFT_WINDOWS["reference"]
    _W["scenarios"] = {}
    _W["floor"] = {}
    # A pristine copy of every calibration dict, so each candidate starts from
    # the shipped constants rather than from whatever the previous candidate
    # left behind in this process.
    _W["pristine"] = {
        name: copy.deepcopy(getattr(cal, name))
        for name in dir(cal)
        if name.isupper() and isinstance(getattr(cal, name), dict)
    }


def _build_scenario(spec: ScenarioSpec):
    """The world for one (seed, day, variant), built once and cached.

    The hostile variant's closures are placed from probe runs of the DEFAULT
    agent and the floor -- both config-independent -- and then frozen. Every
    candidate therefore rides into the same closed roads, which is the only
    way two candidates' hostile-day numbers mean anything next to each other.
    """
    cached = _W["scenarios"].get(spec.key)
    if cached is not None:
        return cached

    rs, events_mod = _W["rs"], _W["events"]
    window, oracle, graph, skeleton = _W["window"], _W["oracle"], _W["graph"], _W["skeleton"]
    built = rs.build_scenario(
        spec.seed, Date.fromisoformat(spec.iso), window.start_min, window.end_min, spec.weekday
    )

    if spec.hostile:
        _reset_calibration({})  # probe with the SHIPPED constants, always
        corridors = {}
        for tag, policy_name in (("smart", "smart"), ("floor", "fixed_threshold")):
            probe, _ = rs.run_one(
                rs.POLICY_FACTORIES[policy_name](), built, oracle, window, graph, skeleton
            )
            counts: dict[str, int] = {}
            for tick in probe.result.ticks:
                counts[tick.courier.cell] = counts.get(tick.courier.cell, 0) + 1
            corridors[tag] = events_mod.corridor_from_occupancy(counts, limit=CORRIDOR_CELLS)

        closures = []
        for index, (offset, duration) in enumerate(CLOSURE_PLAN, start=1):
            tag = "smart" if index % 2 else "floor"
            try:
                closures.append(events_mod.corridor_closure(
                    f"tune-closure-{index:02d}-{tag}", window.start_min + offset,
                    duration, corridors[tag], oracle.matrix,
                ))
            except ValueError:
                continue
        if closures:
            scenario = built.scenario.model_copy(update={"events_timeline": sorted(
                list(built.scenario.events_timeline) + closures, key=lambda e: e.start_min)})
            built = rs.BuiltScenario(
                scenario=scenario, platform=built.platform, surge_field=built.surge_field
            )

    _W["scenarios"][spec.key] = built
    return built


def _reset_calibration(overrides: dict[str, dict[str, float]]) -> None:
    """Restore the shipped constants, then apply this candidate's overrides.

    In place, because `smart.py` reads these dicts by key at call time --
    rebinding the module attribute would leave any already-imported reference
    pointing at the old dict.
    """
    cal = _W["cal"]
    for name, pristine in _W["pristine"].items():
        live = getattr(cal, name)
        live.clear()
        live.update(copy.deepcopy(pristine))
    for name, values in overrides.items():
        getattr(cal, name).update(values)


def _measure(result) -> dict:
    last = max(result.ticks, key=lambda t: t.minute).courier
    gross = sum(d.payout_mxn + d.tip_mxn for d in result.deliveries)
    hours = last.minutes_elapsed / 60.0
    km = last.km_traveled
    return {
        "gross_mxn": round(gross, 1),
        "km": round(km, 1),
        "hours": round(hours, 2),
        "deliveries": len(result.deliveries),
        "mxn_h": round(gross / hours, 2) if hours else 0.0,
        "mxn_km": round(gross / km, 3) if km else 0.0,
        "net_mxn_h": round((gross - COST_PER_KM_MXN * km) / hours, 2) if hours else 0.0,
        "per_hour": round(len(result.deliveries) / hours, 2) if hours else 0.0,
        "idle_share": round(last.minutes_idle / last.minutes_elapsed, 3) if last.minutes_elapsed else 0.0,
        "accept_rate": round(last.offers_accepted / last.offers_seen, 3) if last.offers_seen else 0.0,
    }


def _floor_metrics(spec: ScenarioSpec) -> dict:
    """The baseline for one scenario. Config-independent, so computed once."""
    cached = _W["floor"].get(spec.key)
    if cached is not None:
        return cached
    rs = _W["rs"]
    built = _build_scenario(spec)
    _reset_calibration({})
    out, _ = rs.run_one(
        rs.POLICY_FACTORIES["fixed_threshold"](), built,
        _W["oracle"], _W["window"], _W["graph"], _W["skeleton"],
    )
    metrics = _measure(out.result)
    _W["floor"][spec.key] = metrics
    return metrics


def _split_config(config: dict[str, float]) -> tuple[dict, dict]:
    """Separate constructor arguments from calibration overrides."""
    ctor: dict[str, float] = {}
    overrides: dict[str, dict[str, float]] = {}
    for dict_name, key, _lo, _hi in SEARCH_SPACE:
        if key not in config:
            continue
        if dict_name is None:
            ctor[key] = config[key]
        else:
            overrides.setdefault(dict_name, {})[key] = config[key]
    return ctor, overrides


def evaluate_batch(payload: tuple[list[dict], list[dict]]) -> list[dict]:
    """Every candidate against every scenario THIS worker owns.

    Partitioned by scenario rather than by candidate on purpose: a worker
    pays the 33-second scenario build once and then answers in 0.4 seconds
    per candidate.
    """
    configs, specs_raw = payload
    specs = [ScenarioSpec(**s) for s in specs_raw]
    rs = _W["rs"]

    rows: list[dict] = []
    pid = os.getpid()
    for spec_number, spec in enumerate(specs, start=1):
        t0 = time.time()
        built = _build_scenario(spec)
        floor = _floor_metrics(spec)
        print(f"    [w{pid}] {spec.key} built in {time.time() - t0:.0f} s, "
              f"floor {floor['net_mxn_h']:.1f} net MXN/h; "
              f"{len(configs)} candidates to run", flush=True)
        t0 = time.time()
        for index, config in enumerate(configs):
            if index and index % 100 == 0:
                rate = (time.time() - t0) / index
                left = (len(configs) - index) * rate
                print(f"    [w{pid}] {spec.key} {index}/{len(configs)} "
                      f"({rate:.2f} s/run, ~{left / 60:.1f} min left on this scenario; "
                      f"scenario {spec_number}/{len(specs)})", flush=True)
            ctor, overrides = _split_config(config)
            _reset_calibration(overrides)
            out, _ = rs.run_one(
                rs.POLICY_FACTORIES["smart"](**ctor), built,
                _W["oracle"], _W["window"], _W["graph"], _W["skeleton"],
            )
            smart = _measure(out.result)
            rows.append({
                "config_index": index,
                "scenario": spec.key,
                "hostile": spec.hostile,
                "smart": smart,
                "floor": floor,
            })
    _reset_calibration({})
    return rows


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def shipped_config() -> dict[str, float]:
    """The constants as they ship today, as a candidate, so every table has
    the thing being improved on in it."""
    from src.agent import calibration as cal
    from src.agent.smart import SmartPolicy

    # Read off the constructor's own defaults rather than restating them, so
    # this cannot drift from what actually ships.
    import inspect
    defaults = {
        name: param.default
        for name, param in inspect.signature(SmartPolicy.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    config: dict[str, float] = {}
    for dict_name, key, _lo, _hi in SEARCH_SPACE:
        if dict_name is None:
            config[key] = float(defaults[key])
        else:
            config[key] = float(getattr(cal, dict_name)[key])
    return config


def random_configs(n: int, rng) -> list[dict[str, float]]:
    out = []
    for _ in range(n):
        out.append({
            key: round(rng.uniform(lo, hi), 4)
            for _dict, key, lo, hi in SEARCH_SPACE
        })
    return out


# Knobs worth studying one at a time, and the values to walk them through.
# A random search finds A winner; it does not say WHY it won, and a config
# nobody can explain is a config nobody should ship. Each ladder holds every
# other knob at the base config and moves one, so the resulting table reads
# as a cause.
LADDERS: dict[str, tuple[float, ...]] = {
    # The direct acceptance lever: lower means accept work the computed bar
    # would refuse. This is the one to read first -- the agent's own problem
    # is that it is too fussy for a world of cheap, frequent trips.
    "bar_factor": (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10),
    # What exposure costs: night, rain, heat, riding into a jam.
    "risk_posture": (0.0, 0.15, 0.30, 0.45, 0.60, 0.80),
    # The floor under the reservation price.
    "base_reservation_mxn_per_hour": (8.0, 12.0, 16.0, 20.0, 24.0, 28.0),
    # How many more chances the courier believes are coming.
    "prior_offers_per_hour": (4.0, 6.0, 8.0, 10.0, 12.0, 15.0),
    # How long to stand still before riding somewhere better.
    "min_idle_minutes_before_moving": (3.0, 6.0, 9.0, 12.0, 15.0),
    # The new suspects. 1.0 is what shipped; the world's own motorcycle
    # constant is 0.15, and whether the search lands near it without being
    # told is the interesting part.
    "congestion_belief_factor": (0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00),
    "fuel_cost_per_km_mxn": (0.0, 0.3, 0.6, 0.9, 1.1, 1.4),
    "opportunity_mxn_per_minute": (0.5, 0.8, 1.0, 1.5, 2.0, 2.5),
    # 9.0 is what ships; 17.2 is the world's median. Whether the search walks
    # to the truth again, as it did with the congestion belief, is the test.
    "default_kitchen_minutes": (6.0, 9.0, 12.0, 15.0, 17.0, 20.0, 24.0),
}


def ladder_configs(base: dict[str, float]) -> tuple[list[dict[str, float]], list[dict]]:
    """One-at-a-time walks around `base`. Returns the configs and a label per
    config so the table can name what was moved."""
    configs: list[dict[str, float]] = []
    labels: list[dict] = []
    for key, values in LADDERS.items():
        for value in values:
            candidate = dict(base)
            candidate[key] = value
            configs.append(candidate)
            labels.append({"kind": "ladder", "knob": key, "value": value})
    return configs, labels


def neighbours(base: dict[str, float], n: int, rng, spread: float) -> list[dict[str, float]]:
    """Gaussian jitter around a winner, clipped to the search box."""
    out = []
    for _ in range(n):
        candidate = {}
        for _dict, key, lo, hi in SEARCH_SPACE:
            sigma = (hi - lo) * spread
            candidate[key] = round(min(hi, max(lo, rng.gauss(base[key], sigma))), 4)
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_rows(rows: list[dict], n_configs: int) -> list[dict]:
    """Per candidate: how it did against the floor, averaged over scenarios.

    The headline is NET MXN PER HOUR -- takings minus the agent's own per-km
    cost, over hours actually worked. Gross alone rewards a courier who drives
    170 km for the money, and this project's whole recalibration was about not
    letting a shortcut in a number fall the way you want it to.
    """
    by_config: dict[int, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row["config_index"], []).append(row)

    summaries = []
    for index in range(n_configs):
        runs = by_config.get(index, [])
        if not runs:
            continue
        net = [r["smart"]["net_mxn_h"] for r in runs]
        ratio = [
            r["smart"]["net_mxn_h"] / r["floor"]["net_mxn_h"] if r["floor"]["net_mxn_h"] else 0.0
            for r in runs
        ]
        wins = sum(1 for r in runs if r["smart"]["net_mxn_h"] > r["floor"]["net_mxn_h"])
        summaries.append({
            "config_index": index,
            "runs": len(runs),
            "net_mxn_h": round(sum(net) / len(net), 2),
            "net_ratio": round(sum(ratio) / len(ratio), 4),
            "worst_ratio": round(min(ratio), 4),
            "wins": wins,
            "gross_mxn_h": round(sum(r["smart"]["mxn_h"] for r in runs) / len(runs), 2),
            "mxn_km": round(sum(r["smart"]["mxn_km"] for r in runs) / len(runs), 3),
            "per_hour": round(sum(r["smart"]["per_hour"] for r in runs) / len(runs), 2),
            "idle_share": round(sum(r["smart"]["idle_share"] for r in runs) / len(runs), 3),
            "accept_rate": round(sum(r["smart"]["accept_rate"] for r in runs) / len(runs), 3),
            "floor_net_mxn_h": round(sum(r["floor"]["net_mxn_h"] for r in runs) / len(runs), 2),
        })
    summaries.sort(key=lambda s: -s["net_mxn_h"])
    return summaries


# ---------------------------------------------------------------------------
# Driving the pool
# ---------------------------------------------------------------------------

def specs_for(
    seeds: tuple[int, ...],
    calm_only: bool = False,
    days: tuple[tuple[str, str], ...] = DAYS,
) -> list[ScenarioSpec]:
    """One calm scenario per (seed, day), and optionally the hostile day.

    Seeds and days are crossed rather than zipped: a config that only works
    on the rainy Tuesday is not a config, and neither is one that only works
    on seed 42.
    """
    specs = []
    for seed in seeds:
        for iso, weekday in days:
            specs.append(ScenarioSpec(seed, iso, weekday, False))
        if not calm_only:
            specs.append(ScenarioSpec(seed, CHAOS[0], CHAOS[1], True))
    return specs


def run_pool(configs: list[dict], specs: list[ScenarioSpec], workers: int) -> list[dict]:
    """Scenarios are dealt round-robin across workers; every worker sees every
    candidate. Returns the flat row list."""
    buckets: list[list[dict]] = [[] for _ in range(workers)]
    for i, spec in enumerate(specs):
        buckets[i % workers].append(
            {"seed": spec.seed, "iso": spec.iso, "weekday": spec.weekday, "hostile": spec.hostile}
        )
    payloads = [(configs, bucket) for bucket in buckets if bucket]

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(payloads), initializer=_worker_init) as pool:
        for chunk in pool.map(evaluate_batch, payloads):
            rows.extend(chunk)
    return rows


def print_table(summaries: list[dict], configs: list[dict], top: int, title: str) -> None:
    print(f"\n{title}")
    print(f"{'#':>4} {'net/h':>7} {'vs floor':>9} {'worst':>7} {'wins':>6} "
          f"{'gross/h':>8} {'MXN/km':>7} {'trips/h':>8} {'idle':>6} {'accept':>7}")
    for s in summaries[:top]:
        print(f"{s['config_index']:>4} {s['net_mxn_h']:>7.1f} {s['net_ratio']:>8.1%} "
              f"{s['worst_ratio']:>6.1%} {s['wins']:>3}/{s['runs']:<2} "
              f"{s['gross_mxn_h']:>8.1f} {s['mxn_km']:>7.2f} {s['per_hour']:>8.2f} "
              f"{s['idle_share']:>5.1%} {s['accept_rate']:>6.1%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search for the smart agent's constants.")
    parser.add_argument("--stage", default="explore", choices=("explore", "study", "refine", "confirm"))
    parser.add_argument("--from-config", type=Path, default=None,
                        help="JSON file with a config dict (or a tuning payload) to build around")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--max-seeds", type=int, default=0,
                        help="use only the first N search seeds (smoke tests)")
    parser.add_argument("--with-chaos", action="store_true",
                        help="include hostile scenarios in a search stage (~180 s per run)")
    parser.add_argument("--chaos-seeds", type=int, default=4,
                        help="how many seeds get a hostile scenario in --stage confirm")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "Docs" / "tuning")
    args = parser.parse_args(argv)

    import random
    rng = random.Random(args.rng_seed)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    shipped = shipped_config()
    search_seeds = SEARCH_SEEDS[:args.max_seeds] if args.max_seeds else SEARCH_SEEDS
    holdout_seeds = HOLDOUT_SEEDS[:args.max_seeds] if args.max_seeds else HOLDOUT_SEEDS

    def load_base_config() -> dict[str, float]:
        if args.from_config is None:
            return shipped
        payload = json.loads(args.from_config.read_text(encoding="utf-8"))
        return payload.get("best_config", payload)

    if args.stage == "study":
        # The interpretable pass: the base config, every one-knob ladder
        # around it, and a ring of small joint perturbations so the table
        # also says whether the knobs interact.
        base = load_base_config()
        ladders, labels = ladder_configs(base)
        configs = [shipped, base] + ladders + neighbours(base, args.samples, rng, spread=0.08)
        config_labels = (
            [{"kind": "shipped"}, {"kind": "base"}]
            + labels
            + [{"kind": "jitter"}] * args.samples
        )
        specs = specs_for(search_seeds, calm_only=not args.with_chaos)
        label = f"STUDY — {len(configs)} candidates x {len(specs)} scenarios"
    elif args.stage == "explore":
        configs = [shipped] + random_configs(args.samples, rng)
        specs = specs_for(search_seeds, calm_only=not args.with_chaos)
        label = f"EXPLORE — {len(configs)} candidates x {len(specs)} scenarios"
    elif args.stage == "refine":
        previous = json.loads((args.out / "explore.json").read_text(encoding="utf-8"))
        seeds_from = [previous["best_config"]] + previous["runner_up_configs"]
        configs = [shipped] + seeds_from
        for base in seeds_from[:3]:
            configs.extend(neighbours(base, args.samples // 4, rng, spread=0.10))
        specs = specs_for(search_seeds, calm_only=not args.with_chaos)
        label = f"REFINE — {len(configs)} candidates x {len(specs)} scenarios"
    else:
        previous = json.loads((args.out / "refine.json").read_text(encoding="utf-8"))
        configs = [shipped, previous["best_config"]] + previous["runner_up_configs"][:4]
        # Calm on every seed, search and hold-out; hostile on a few, because
        # a hostile run costs about three hundred calm ones.
        specs = specs_for(search_seeds + holdout_seeds, calm_only=True)
        chaos_seeds = (search_seeds + holdout_seeds)[:args.chaos_seeds]
        specs += [ScenarioSpec(seed, CHAOS[0], CHAOS[1], True) for seed in chaos_seeds]
        label = f"CONFIRM — {len(configs)} candidates x {len(specs)} scenarios (incl. hold-out)"

    print(label, flush=True)
    print(f"  {args.workers} workers; scenario builds dominate, runs are ~0.4 s each", flush=True)

    rows = run_pool(configs, specs, args.workers)
    summaries = score_rows(rows, len(configs))
    print_table(summaries, configs, args.top, f"{args.stage.upper()} — best by net MXN/h")

    shipped_row = next(s for s in summaries if s["config_index"] == 0)
    print(f"\nshipped constants (index 0): net {shipped_row['net_mxn_h']:.1f} MXN/h, "
          f"{shipped_row['net_ratio']:.1%} of the floor, "
          f"{shipped_row['wins']}/{shipped_row['runs']} wins")

    if args.stage == "confirm":
        # Split the confirm table by whether the search was allowed to see it.
        search_keys = {s.key for s in specs_for(search_seeds, calm_only=True)}
        for name, wanted in (("SEARCH seeds", True), ("HOLD-OUT seeds", False)):
            subset = [r for r in rows if (r["scenario"] in search_keys) == wanted]
            print_table(score_rows(subset, len(configs)), configs, args.top,
                        f"CONFIRM — {name} only")

    best = summaries[0]
    payload = {
        "stage": args.stage,
        "config_labels": locals().get("config_labels"),
        "configs": configs,
        "rng_seed": args.rng_seed,
        "scenarios": [s.key for s in specs],
        "search_seeds": list(search_seeds),
        "holdout_seeds": list(holdout_seeds),
        "param_names": PARAM_NAMES,
        "shipped_config": shipped,
        "shipped_summary": shipped_row,
        "best_config": configs[best["config_index"]],
        "best_summary": best,
        "runner_up_configs": [configs[s["config_index"]] for s in summaries[1:9]],
        "summaries": summaries,
        "rows": rows,
    }
    path = args.out / f"{args.stage}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"Total wall time: {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
