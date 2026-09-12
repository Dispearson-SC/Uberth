"""Everything derived from a `ShiftResult`.

This module is pure: it never imports `engine`, `platform`, `enrichment` or
`agent`. It reads only `src.core.ports` types (`ShiftResult`, `TickRecord`,
`DeliveryRecord`, `DecisionTrace`, `OfferEvaluation`), which is exactly what
a headless shift run and a hand-built fixture both provide.

Kilometres and minutes are reported as separate fields everywhere. They
decouple under traffic and detours, and that decoupling is the point of the
whole exercise — collapsing them into one number would hide it.

Every metric here reads directly off `ports.py` fields — no injected
side-channel mappings and no approximations of data the contract already
carries:

- Average/median fare use `DeliveryRecord.payout_mxn + tip_mxn`, the
  REALISED payout, never the `OfferEvaluation.expected_net_mxn` estimate.
- Surge capture reads `OfferEvaluation.surge_flag` straight out of every
  `DecisionTrace.considered` entry.
- Unpaid kilometres are read from `ShiftResult.unpaid_km`, which the engine
  accumulates directly (repositioning, the unpaid leg out from home, and
  the unpaid leg back at the end) — this module does not re-derive it from
  per-tick deltas.

One metric nobody asked for but that DeliveryRecord now makes possible:
fare calibration — the signed gap between what the policy EXPECTED when it
accepted an order (`OfferEvaluation.expected_net_mxn` on the accepting
tick) and what it actually GOT (`DeliveryRecord.payout_mxn + tip_mxn`). A
policy with a well-calibrated model of its own economics should average
close to zero here; a large positive or negative mean, or a wide spread,
says the policy is flying blind about its own numbers even when its
accept/reject choices look reasonable in aggregate.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from src.core.ports import Action, DeliveryRecord, ShiftResult, TickRecord

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class ShiftMetrics:
    """Everything derived from one `ShiftResult`."""

    policy_name: str
    seed: int
    shift_start_min: int
    shift_end_min: int

    # Money and throughput
    gross_mxn: float
    mxn_per_hour: float
    deliveries: int
    deliveries_per_hour: float

    # Kept separate on purpose, always
    km_traveled: float
    minutes_elapsed: float

    idle_fraction: float
    acceptance_rate: float
    offers_seen: int
    offers_accepted: int

    # Realised, from DeliveryRecord.payout_mxn + tip_mxn. None if no
    # deliveries completed.
    average_fare_mxn: float | None
    median_fare_mxn: float | None

    # The cost the platform never shows the courier. Read straight from
    # ShiftResult.unpaid_km, accumulated by the engine.
    unpaid_km: float
    unpaid_km_fraction: float  # unpaid_km / km_traveled, 0.0 if no distance travelled

    end_of_shift_distance_from_home_km: float | None

    # From OfferEvaluation.surge_flag across every DecisionTrace.considered.
    # None only when the shift had no offers to measure.
    surge_capture_accepted_share: float | None
    surge_capture_offered_share: float | None
    surge_capture_edge: float | None  # accepted_share - offered_share

    # actual (payout + tip) minus expected (expected_net_mxn at accept
    # time), averaged over deliveries where both are known. None if no
    # delivery could be matched back to its accepting decision.
    fare_calibration_mean_error_mxn: float | None
    fare_calibration_stdev_error_mxn: float | None
    fare_calibration_n: int


def _sorted_ticks(ticks: Sequence[TickRecord]) -> list[TickRecord]:
    return sorted(ticks, key=lambda t: t.minute)


def _realised_fares(deliveries: Sequence[DeliveryRecord]) -> list[float]:
    return [d.payout_mxn + d.tip_mxn for d in deliveries]


def _surge_capture(ticks: Sequence[TickRecord]) -> tuple[float | None, float | None, float | None]:
    """Surge capture straight from `OfferEvaluation.surge_flag`.

    Offered share: over every distinct order_id ever `considered` in a
    DecisionTrace (deduplicated — the same order can be reconsidered across
    minutes). Accepted share: over every order actually accepted.
    """
    seen_flags: dict[str, bool] = {}
    accepted_flags: list[bool] = []
    for tick in _sorted_ticks(ticks):
        decision = tick.decision
        if decision is None:
            continue
        for evaluation in decision.trace.considered:
            seen_flags.setdefault(evaluation.order_id, evaluation.surge_flag)
            if decision.action == Action.ACCEPT and decision.order_id == evaluation.order_id:
                accepted_flags.append(evaluation.surge_flag)

    offered_share = (sum(seen_flags.values()) / len(seen_flags)) if seen_flags else None
    accepted_share = (sum(accepted_flags) / len(accepted_flags)) if accepted_flags else None
    edge = (accepted_share - offered_share) if (accepted_share is not None and offered_share is not None) else None
    return accepted_share, offered_share, edge


def _expected_net_mxn_at_accept(ticks: Sequence[TickRecord], order_id: str, accepted_at_min: int) -> float | None:
    """The `OfferEvaluation.expected_net_mxn` the policy believed for
    `order_id` at the exact tick it accepted it. `None` if that tick or
    evaluation cannot be found (e.g. a hand-built fixture that skips it)."""
    for tick in ticks:
        if tick.minute != accepted_at_min:
            continue
        decision = tick.decision
        if decision is None or decision.action != Action.ACCEPT or decision.order_id != order_id:
            continue
        for evaluation in decision.trace.considered:
            if evaluation.order_id == order_id:
                return evaluation.expected_net_mxn
    return None


def _fare_calibration(
    ticks: Sequence[TickRecord],
    deliveries: Sequence[DeliveryRecord],
) -> tuple[float | None, float | None, int]:
    errors: list[float] = []
    for delivery in deliveries:
        expected = _expected_net_mxn_at_accept(ticks, delivery.order_id, delivery.accepted_at_min)
        if expected is None:
            continue
        actual = delivery.payout_mxn + delivery.tip_mxn
        errors.append(actual - expected)
    if not errors:
        return None, None, 0
    mean_error = statistics.fmean(errors)
    stdev_error = statistics.stdev(errors) if len(errors) > 1 else 0.0
    return mean_error, stdev_error, len(errors)


def _end_of_shift_distance_from_home_km(ticks: Sequence[TickRecord]) -> float | None:
    if not ticks:
        return None
    last = _sorted_ticks(ticks)[-1]
    c = last.courier
    return haversine_km(c.lat, c.lon, c.home_lat, c.home_lon)


def compute_metrics(result: ShiftResult) -> ShiftMetrics:
    """Compute every metric for one `ShiftResult`, reading realised
    payout, surge flags and unpaid km directly off the ports contract."""
    fares = _realised_fares(result.deliveries)
    accepted_share, offered_share, edge = _surge_capture(result.ticks)
    calib_mean, calib_stdev, calib_n = _fare_calibration(result.ticks, result.deliveries)

    return ShiftMetrics(
        policy_name=result.policy_name,
        seed=result.seed,
        shift_start_min=result.shift_start_min,
        shift_end_min=result.shift_end_min,
        gross_mxn=result.earnings_mxn,
        mxn_per_hour=result.mxn_per_hour,
        deliveries=result.deliveries_completed,
        deliveries_per_hour=result.deliveries_per_hour,
        km_traveled=result.km_traveled,
        minutes_elapsed=result.minutes_elapsed,
        idle_fraction=(result.minutes_idle / result.minutes_elapsed) if result.minutes_elapsed else 0.0,
        acceptance_rate=result.acceptance_rate,
        offers_seen=result.offers_seen,
        offers_accepted=result.offers_accepted,
        average_fare_mxn=statistics.fmean(fares) if fares else None,
        median_fare_mxn=statistics.median(fares) if fares else None,
        unpaid_km=result.unpaid_km,
        unpaid_km_fraction=(result.unpaid_km / result.km_traveled) if result.km_traveled else 0.0,
        end_of_shift_distance_from_home_km=_end_of_shift_distance_from_home_km(result.ticks),
        surge_capture_accepted_share=accepted_share,
        surge_capture_offered_share=offered_share,
        surge_capture_edge=edge,
        fare_calibration_mean_error_mxn=calib_mean,
        fare_calibration_stdev_error_mxn=calib_stdev,
        fare_calibration_n=calib_n,
    )


__all__ = ["EARTH_RADIUS_KM", "ShiftMetrics", "compute_metrics", "haversine_km"]
