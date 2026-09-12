"""Ground truth: surge as an emergent supply/demand mechanism, not noise.

This module is ground truth. `src/agent/` must never import it directly — a
courier only ever sees the coarse, lagged, quantised heatmap the platform
layer derives from this, never `surge_at_spawn` or the raw supply field.

The cheap way to fake surge is `payout * random(1.2, 2.0)`. This module does
not do that. Surge here is the read-out of a real dynamical system:

    ratio(c, t) = demand(c, t) / max(supply(c, t), eps)
    surge(c, t) = clip(1 + k * (ratio - ratio_equilibrium), surge_min, surge_max)

`demand(c, t)` is the order-arrival intensity computed by `demand.py`.
`supply(c, t)` is a competing-courier DENSITY FIELD, not a roster of
individually simulated couriers — modelling thousands of independent agents
would be both slower and less auditable than modelling the aggregate flow
they produce. The field migrates toward high-surge cells, but not
instantly: a decision made at minute `t` (in response to `surge(c, t)`)
only lands `reaction_lag_min` minutes later, once couriers have noticed the
signal and physically repositioned.

That lag is the entire point of this module. Every courier reacts to the
*same* published heatmap, and nobody teleports, so the response is
necessarily late. If the lag is long enough or the response aggressive
enough relative to it, a locally hot cell overshoots: enough supply arrives
by `t + lag` to flip the cell into an oversupplied state, which triggers an
exodus that lands `lag` minutes after that, which reopens the shortage, and
so on. Spike, collapse, spike — a limit cycle that falls straight out of
delayed negative feedback, not out of a scripted waveform. This is exactly
why the in-app heatmap "lies": it is always describing a supply state that
has already moved on by the time anyone acts on it.

Every constant below is a calibration knob, not measured data, and is kept
in one of the two dicts so the whole mechanism stays auditable from this
file alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.world import geo
from src.world.timeline import SupplyTick

# --------------------------------------------------------------------------
# Calibration (all values below are tuned knobs, not measured data)
# --------------------------------------------------------------------------

SURGE_CALIBRATION: dict[str, float] = {
    # Sensitivity of surge to the demand/supply imbalance. Re-tuned down
    # from 2.5 to 2.0 after fixing the supply-migration amplitude bug in
    # `SUPPLY_CALIBRATION` (`max_outflow_fraction`, `migration_rate`,
    # `gap_saturation`): with the dynamics fixed, this is what lands the
    # measured mean/percentile targets (see `surge.py`'s module docstring
    # for the mechanism) — `ratio_equilibrium` alone could not do this
    # (moving it only slid the whole city between "always saturated" and
    # "always flat", it never bought a graded middle, because the
    # amplitude problem was upstream in the supply field, not in this
    # read-out formula).
    "k": 2.0,
    # The demand/supply ratio treated as "balanced" (surge == 1.0 there).
    "ratio_equilibrium": 0.35,
    "surge_min": 1.0,
    "surge_max": 2.5,
    # Floor added to supply before dividing, so an empty cell doesn't blow
    # the ratio up to infinity. Deliberately NOT scaled together with
    # `total_couriers`/demand: this is a fixed numerical floor (roughly
    # "there's always a stray courier or two"), not a physical quantity
    # that grows with the fleet. At the old, ~10x smaller total_couriers,
    # this floor was 5-15% of a typical cell's supply — big enough to
    # swing the ratio wildly whenever a cell's supply dipped near zero,
    # which made the whole field behave like a handful of discrete agents
    # rather than a density field, and turned every migration cycle into a
    # bang-bang square wave between clipped extremes. Leaving it fixed
    # while the fleet scales up 10x makes it relatively negligible (well
    # under 1% of a typical cell's supply), which is what actually lets the
    # field behave as a continuum and lets `ratio_equilibrium` act as a
    # real tunable knob instead of a cliff edge.
    "eps": 0.05,
}

SUPPLY_CALIBRATION: dict[str, float] = {
    # City-wide competing-courier fleet size (a density total, not a literal
    # headcount — see module docstring: this is a field, not an agent roster).
    # Sized relative to `DEMAND_CALIBRATION["orders_per_weight_unit_per_min"]`
    # (demand.py) so a busy cell's demand/supply ratio actually crosses
    # `ratio_equilibrium` at the lunch/dinner peaks instead of sitting flat.
    # Scaled up 10x together with that demand rate from an earlier
    # calibration (45 couriers) specifically to fix a small-number
    # artefact — see the `eps` comment above for why the scale itself,
    # not just the ratio, matters here.
    "total_couriers": 450.0,
    # Minutes between a courier noticing a surge signal and actually being
    # present in the cell it points to (notice + physical repositioning).
    # THIS LAG IS THE MECHANISM: see module docstring.
    "reaction_lag_min": 11,
    # Fraction of a cell's current supply that responds to a surge gap in
    # one decision epoch, BEFORE the `max_outflow_fraction` hard cap below.
    # Measured against real order data this used to be 0.10, which (with an
    # 11-minute lag) overshot far enough to slam both the `surge_min`/
    # `surge_max` clip bounds every cycle — a bang-bang limit cycle, not a
    # graded market (see `max_outflow_fraction` for the actual amplitude
    # fix; this value stays high enough to reach the cap quickly once a gap
    # is real, so small/moderate gaps still get a proportional, non-zero
    # response instead of a dead zone).
    "migration_rate": 0.08,
    # Minimum surge gap (in surge units) between neighbours before couriers
    # bother moving at all — avoids constant micro-churn on noise.
    "gap_floor": 0.05,
    # Surge-unit gap at which the outflow fraction saturates (soft cap so a
    # single huge gap can't drain a cell to zero in one epoch). Widened
    # from 1.5 so saturation isn't reached by a couple of modest-gap
    # neighbours alone (radius-2 gives up to ~18 migration partners) —
    # together with the lower `max_outflow_fraction` this keeps the
    # response graded across a wider range of gap sizes instead of
    # snapping straight to the cap.
    "gap_saturation": 2.5,
    # Relative noise (std as a fraction of the flow) applied to each
    # migration flow via the `competitors` stream — real couriers don't all
    # react with identical precision to the same signal.
    "noise_std": 0.08,
    # Couriers don't re-decide off a single noisy minute-to-minute reading;
    # they react to a short rolling impression of the heatmap. EMA smoothing
    # factor applied to surge before it drives migration decisions (does
    # NOT affect the surge value recorded/returned — only what couriers
    # act on). Without this the delayed feedback loop below rings as a
    # stiff square wave instead of a graded spike-collapse-spike cycle.
    "decision_smoothing_alpha": 0.25,
    # A cell never migrates away more than this fraction of its current
    # supply in one decision epoch — real couriers don't all leave a zone
    # at once. THIS IS THE PRIMARY AMPLITUDE-DAMPING KNOB: measured against
    # real order data, the old 0.35 was so loose it never actually bound
    # (migration_rate alone reached the cap only around gap saturation,
    # and even then 0.35/minute compounded over an 11-minute lag empties
    # most of a cell before the response lands), which let each cell's
    # delayed feedback loop swing hard enough to slam both `surge_min` and
    # `surge_max` every cycle. A hard per-minute ceiling this low — well
    # under the linear `migration_rate` gain — is what actually turns the
    # loop's gain into a soft, self-limiting saturation instead of letting
    # amplitude grow until it hits the clip bounds, without touching the
    # lag itself (the emergent mechanism stays intact: verified that
    # zeroing `reaction_lag_min`/`reaction_lag_jitter_min` still collapses
    # the oscillation to ~0, so this is damping the response to the lag,
    # not replacing it with something scripted).
    "max_outflow_fraction": 0.041,
    # How many H3 grid rings out a cell's migration partners reach. A
    # radius-1 ring gave every cell only its 6 immediate neighbours, which
    # made a busy cell and its single dominant neighbour trade the *same*
    # block of couriers back and forth every cycle — a stiff two-node
    # square wave, not a field. Spreading migration over a wider
    # neighbourhood dilutes any one link's swing.
    "neighbor_ring_k": 2,
    # Per-cell jitter (minutes) on `reaction_lag_min`, drawn once per cell
    # from the `competitors` stream. Every cell reacting on the *exact*
    # same delay is what synchronises the whole grid into one lockstep
    # bang-bang oscillation; a spread of lags means neighbouring cells are
    # never all mid-overshoot at the same instant, which is what actually
    # breaks the square wave into a graded cycle. The oscillation is still
    # 100% emergent from the lag — this only desynchronises *which* minute
    # each cell's lag lands on, it never scripts a waveform.
    "reaction_lag_jitter_min": 4,
    # A cell can only migrate away supply ABOVE this fraction of its
    # nominal (baseline-allocation) supply — never below it. Without this,
    # a sustained one-sided gap compounds across consecutive one-minute
    # decision epochs (each capped at `max_outflow_fraction`, but
    # `(1 - max_outflow_fraction) ** n` still -> 0 for n large enough) and
    # drains a cell to ~0. Once supply is ~0, ratio = demand / eps
    # explodes to 10s-100s regardless of `ratio_equilibrium`/`k`, which
    # pins surge at the cap independent of calibration — this is what
    # actually produced the "cliff, not a curve" symptom: it wasn't a
    # smoothness problem in the oscillation, it was ratio occasionally
    # blowing past any sane threshold because a handful of cells were
    # briefly and repeatedly emptied out completely.
    "min_supply_reserve_fraction": 0.30,
}


# --------------------------------------------------------------------------
# Output container
# --------------------------------------------------------------------------


@dataclass
class SurgeField:
    """Ground-truth surge per cell per minute, plus the supply timeline that
    produced it. Built once per scenario by `build_supply_and_surge`."""

    cells: list[str]
    minutes: list[int]
    surge_by_cell: dict[str, np.ndarray]  # cell -> array aligned with `minutes`
    supply_ticks: list[SupplyTick]

    _minute_index: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._minute_index = {minute: i for i, minute in enumerate(self.minutes)}

    def at(self, cell: str, minute: int) -> float:
        """Ground-truth surge multiplier for one cell at one minute.

        Falls back to 1.0 (no surge) for a cell or minute outside the
        computed grid — this should only happen for a cell with no demand
        and no migration ever reaching it.
        """
        idx = self._minute_index.get(minute)
        if idx is None:
            return SURGE_CALIBRATION["surge_min"]
        arr = self.surge_by_cell.get(cell)
        if arr is None:
            return SURGE_CALIBRATION["surge_min"]
        return float(arr[idx])


# --------------------------------------------------------------------------
# Core mechanism
# --------------------------------------------------------------------------


def _neighbor_map(cells: Sequence[str], ring_k: int) -> dict[str, list[str]]:
    """H3 neighbours of each cell within `ring_k` grid rings, restricted to
    the cell set we are actually tracking (a cell whose real-world
    neighbour isn't part of the operating grid simply has fewer migration
    partners)."""
    cell_set = set(cells)
    return {c: [n for n in geo.cell_neighbors(c, k=ring_k) if n in cell_set] for c in cells}


def build_supply_and_surge(
    rng: np.random.Generator,
    cells: Sequence[str],
    minutes: Sequence[int],
    demand_by_cell: Mapping[str, np.ndarray],
    courier_supply_mult: Sequence[float] | None = None,
    baseline_weights: Mapping[str, float] | None = None,
    calibration: Mapping[str, float] | None = None,
) -> SurgeField:
    """Simulate the competing-courier supply field and read ground-truth
    surge off it, minute by minute, for the whole shift.

    Parameters
    ----------
    rng:
        Must be the `competitors` stream from `Scenario.rng_streams()`. Used
        only to add small realism noise to migration flows — never to
        decide anything about a single courier's identity or route.
    cells:
        The full cell grid to track, in a fixed deterministic order (the
        caller's responsibility — same order in, same order out).
    minutes:
        Consecutive simulated minutes for the shift, e.g.
        `list(range(shift_start_min, shift_end_min))`.
    demand_by_cell:
        cell -> array of per-minute order-arrival intensity, aligned with
        `minutes`. A cell absent from this mapping is treated as zero
        demand for the whole shift.
    courier_supply_mult:
        Optional per-minute city-wide multiplier on the total fleet size
        (e.g. rain keeping couriers off the road). Defaults to all-ones.
        Never sourced from `weather.py` here — the caller passes the
        number, this module never imports the weather producer.
    baseline_weights:
        Optional relative weight per cell used only to seed the *initial*
        supply distribution (e.g. population, so couriers start out roughly
        where people live). Defaults to a uniform split across `cells`.
    calibration:
        Optional overrides merged on top of `SURGE_CALIBRATION` and
        `SUPPLY_CALIBRATION`.
    """
    cal = {**SURGE_CALIBRATION, **SUPPLY_CALIBRATION}
    if calibration:
        cal.update(calibration)

    n_cells = len(cells)
    n_min = len(minutes)
    if n_cells == 0 or n_min == 0:
        return SurgeField(cells=list(cells), minutes=list(minutes), surge_by_cell={}, supply_ticks=[])

    cell_index = {c: i for i, c in enumerate(cells)}
    neighbors = _neighbor_map(cells, ring_k=int(cal.get("neighbor_ring_k", 1)))
    neighbor_idx = {c: [cell_index[n] for n in ns] for c, ns in neighbors.items()}

    mult = np.asarray(courier_supply_mult, dtype=float) if courier_supply_mult is not None else np.ones(n_min)
    if len(mult) != n_min:
        raise ValueError(f"courier_supply_mult length {len(mult)} != len(minutes) {n_min}")

    demand = np.zeros((n_cells, n_min))
    for c, arr in demand_by_cell.items():
        i = cell_index.get(c)
        if i is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if len(arr) != n_min:
            raise ValueError(f"demand_by_cell[{c!r}] length {len(arr)} != len(minutes) {n_min}")
        demand[i, :] = arr

    if baseline_weights:
        base = np.array([max(float(baseline_weights.get(c, 0.0)), 0.0) for c in cells])
        if base.sum() <= 0:
            base = np.ones(n_cells)
    else:
        base = np.ones(n_cells)
    base_frac = base / base.sum()

    supply = base_frac * cal["total_couriers"] * mult[0]

    surge_out = np.zeros((n_cells, n_min))
    supply_ticks: list[SupplyTick] = []
    pending: dict[int, np.ndarray] = {}

    base_lag = int(cal["reaction_lag_min"])
    jitter = int(cal.get("reaction_lag_jitter_min", 0))
    if jitter > 0:
        # Drawn once per cell, in the fixed `cells` order, from the
        # `competitors` stream — deterministic given the same seed, and
        # never re-drawn minute to minute (a cell's reaction speed is a
        # stable trait, not noise).
        lag_offsets = rng.integers(-jitter, jitter + 1, size=n_cells)
        lag_per_cell = np.maximum(base_lag + lag_offsets, 1)
    else:
        lag_per_cell = np.full(n_cells, max(base_lag, 1))
    eps = cal["eps"]
    k = cal["k"]
    ratio_eq = cal["ratio_equilibrium"]
    surge_min = cal["surge_min"]
    surge_max = cal["surge_max"]
    migration_rate = cal["migration_rate"]
    gap_floor = cal["gap_floor"]
    gap_sat = cal["gap_saturation"]
    noise_std = cal["noise_std"]
    smoothing_alpha = cal["decision_smoothing_alpha"]
    max_outflow_fraction = cal["max_outflow_fraction"]
    reserve_fraction = cal.get("min_supply_reserve_fraction", 0.0)

    surge_ema: np.ndarray | None = None

    for t_idx in range(n_min):
        # 1. City-wide fleet size can breathe (e.g. rain sends couriers
        #    home); rescale the whole field to the new target while keeping
        #    each cell's relative share.
        target_total = cal["total_couriers"] * mult[t_idx]
        current_total = supply.sum()
        if current_total > 0:
            supply = supply * (target_total / current_total)
        else:
            supply = base_frac * target_total
        min_supply = reserve_fraction * base_frac * target_total

        # 2. Migration decided `lag` minutes ago lands now.
        arriving = pending.pop(t_idx, None)
        if arriving is not None:
            supply = supply + arriving
            np.clip(supply, 0.0, None, out=supply)

        # 3. Record the ground-truth supply that is actually in effect
        #    during this minute, before this minute's own decisions land.
        supply_ticks.append(SupplyTick(minute=minutes[t_idx], couriers_per_cell=dict(zip(cells, supply.tolist()))))

        # 4. Read surge off the current demand/supply state.
        ratio = demand[:, t_idx] / np.maximum(supply, eps)
        surge_now = np.clip(1.0 + k * (ratio - ratio_eq), surge_min, surge_max)
        surge_out[:, t_idx] = surge_now

        # 5. Couriers react to a short rolling impression of the surge
        #    signal, not one noisy instantaneous minute — but even that
        #    decision only materialises `lag` minutes from now, never
        #    sooner. That delay is what lets the field overshoot instead of
        #    tracking demand.
        surge_ema = surge_now.copy() if surge_ema is None else smoothing_alpha * surge_now + (1 - smoothing_alpha) * surge_ema

        for c in cells:
            i = cell_index[c]
            # Each cell's own (jittered) lag governs when ITS courier
            # population, reacting now, actually finishes relocating.
            arrival_idx = t_idx + int(lag_per_cell[i])
            if arrival_idx >= n_min:
                continue
            ns = neighbor_idx[c]
            if not ns:
                continue
            own = surge_ema[i]
            gaps = np.maximum(surge_ema[ns] - own, 0.0)
            total_gap = gaps.sum()
            if total_gap <= gap_floor:
                continue
            saturation = min(total_gap, gap_sat) / gap_sat
            outflow = min(migration_rate * saturation, max_outflow_fraction) * supply[i]
            # Never migrate a cell below its reserve floor: `min_supply` was
            # computed above but previously never consulted here, so a
            # sustained one-sided gap could compound across consecutive
            # one-minute epochs and drain a cell toward zero even though
            # each individual step looked capped. Once supply hits ~0,
            # ratio = demand / eps explodes regardless of `k`/`ratio_eq`,
            # pinning surge at `surge_max` — this is the actual source of
            # the "cliff, not curve" sweep signature, not a calibration
            # problem. Clamping outflow to respect the reserve fixes it at
            # the root, and also stops the local emptying from re-appearing
            # as a same-instant compensating boost everywhere else via the
            # step-1 city-wide renormalisation (the closest thing to a
            # global coupling in this model — see its comment above).
            outflow = min(outflow, max(supply[i] - min_supply[i], 0.0))
            if outflow <= 0:
                continue
            shares = gaps / total_gap
            bucket = pending.setdefault(arrival_idx, np.zeros(n_cells))
            for n_i, share in zip(ns, shares):
                move = outflow * share
                if move <= 0:
                    continue
                # Noise on courier response: not everyone reacts identically
                # to the same published signal.
                move = move * max(0.0, 1.0 + rng.normal(0.0, noise_std))
                bucket[i] -= move
                bucket[n_i] += move

    surge_by_cell = {c: surge_out[cell_index[c], :] for c in cells}
    return SurgeField(cells=list(cells), minutes=list(minutes), surge_by_cell=surge_by_cell, supply_ticks=supply_ticks)
