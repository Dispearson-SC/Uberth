"""The A/B harness: run N policies over the identical frozen scenario and
report a comparison table with mean and spread across seeds.

This module does not import `engine`, `world` or `agent`. It has no way to
run a shift by itself — those layers are being written in parallel and may
not exist yet — so it takes a `ShiftRunner` callable as a dependency:

    ShiftRunner = Callable[[Policy, int, int, int], ShiftResult]
                            policy, seed, shift_start_min, shift_end_min

Whoever wires the real engine together (or a test) supplies that callable.
It is the caller's job to guarantee the "identical frozen scenario"
property: same seed, same exogenous world, same RNG streams across
policies. This module only guarantees it calls `run_shift` with the exact
same `(seed, shift_start_min, shift_end_min)` tuple for every policy, so the
hermetic comparison is not accidentally broken here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from src.core.ports import Policy, ShiftResult
from src.eval.metrics import ShiftMetrics, compute_metrics

# (policy, seed, shift_start_min, shift_end_min) -> ShiftResult
ShiftRunner = Callable[[Policy, int, int, int], ShiftResult]


@dataclass(frozen=True)
class ShiftWindow:
    """One of the four named shift windows.

    `end_min` may exceed 1440 (the Night window does: 1080-1560). Minutes
    are a continuous counter from the scenario's reference start, not
    reset at midnight, so `end_min > start_min` always holds and no
    wraparound arithmetic is needed to run a shift. `wall_clock` below is
    only for display, and it is where the wrap must be handled deliberately
    (mod 1440), so it never silently produces "26:00".
    """

    label: str
    start_min: int
    end_min: int

    @property
    def duration_minutes(self) -> int:
        return self.end_min - self.start_min

    @property
    def crosses_midnight(self) -> bool:
        return self.end_min > 1440

    def wall_clock(self, absolute_minute: int) -> str:
        """Format an absolute minute as HH:MM wall-clock time, wrapping at
        1440 minutes (24h). E.g. `wall_clock(1560) == "02:00"`."""
        minute_of_day = absolute_minute % 1440
        return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"

    @property
    def label_with_hours(self) -> str:
        return f"{self.label} {self.wall_clock(self.start_min)}-{self.wall_clock(self.end_min)}"


SHIFT_WINDOWS: dict[str, ShiftWindow] = {
    "early": ShiftWindow("Early", 300, 840),
    "day": ShiftWindow("Day", 720, 1200),
    "reference": ShiftWindow("Reference", 840, 1320),
    "night": ShiftWindow("Night", 1080, 1560),
}

# The metrics that get a delta-vs-baseline column in the comparison table.
# Percent deltas only make sense on non-negative, usually-positive metrics;
# fractions and shares are compared as plain percentage-point differences
# instead (see _delta below).
_PERCENT_METRICS = (
    "gross_mxn",
    "mxn_per_hour",
    "deliveries",
    "deliveries_per_hour",
    "km_traveled",
    "unpaid_km",
)
_POINT_METRICS = (
    "idle_fraction",
    "acceptance_rate",
    "unpaid_km_fraction",
    "surge_capture_accepted_share",
    "surge_capture_offered_share",
    "surge_capture_edge",
)
# Metrics whose delta vs baseline is a plain difference in the metric's own
# unit — a percent delta or a percentage-point delta would misrepresent
# them (MXN amounts that can be negative, or a fare-calibration spread).
_ABSOLUTE_METRICS = (
    "average_fare_mxn",
    "median_fare_mxn",
    "fare_calibration_mean_error_mxn",
    "fare_calibration_stdev_error_mxn",
    "end_of_shift_distance_from_home_km",
)


@dataclass(frozen=True)
class AggregatedMetric:
    """Mean and spread of one metric across seeds."""

    mean: float
    stdev: float
    minimum: float
    maximum: float
    n: int


def _aggregate(values: Sequence[float]) -> AggregatedMetric:
    return AggregatedMetric(
        mean=statistics.fmean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
        n=len(values),
    )


_METRIC_FIELDS = _PERCENT_METRICS + _POINT_METRICS + _ABSOLUTE_METRICS


@dataclass(frozen=True)
class PolicyWindowSummary:
    """One policy's aggregated metrics for one shift window, across seeds."""

    policy_name: str
    window_label: str
    seeds: tuple[int, ...]
    aggregated: dict[str, AggregatedMetric]
    per_seed: tuple[ShiftMetrics, ...]


def _summarize(policy_name: str, window: ShiftWindow, per_seed: Sequence[ShiftMetrics]) -> PolicyWindowSummary:
    aggregated: dict[str, AggregatedMetric] = {}
    for field_name in _METRIC_FIELDS:
        values = [getattr(m, field_name) for m in per_seed]
        values = [v for v in values if v is not None]
        if values:
            aggregated[field_name] = _aggregate(values)
    return PolicyWindowSummary(
        policy_name=policy_name,
        window_label=window.label,
        seeds=tuple(m.seed for m in per_seed),
        aggregated=aggregated,
        per_seed=tuple(per_seed),
    )


def _delta_pct(baseline_mean: float, candidate_mean: float) -> float | None:
    if baseline_mean == 0:
        return None
    return (candidate_mean - baseline_mean) / abs(baseline_mean) * 100.0


@dataclass(frozen=True)
class ComparisonRow:
    """One (window, policy) row of the comparison table, with deltas
    against the baseline policy for the same window."""

    window_label: str
    policy_name: str
    summary: PolicyWindowSummary
    is_baseline: bool
    delta_pct: dict[str, float | None]  # metric name -> percent delta vs baseline (None if not applicable)
    delta_points: dict[str, float | None]  # metric name -> absolute delta vs baseline, for fractions/shares
    delta_abs: dict[str, float | None]  # metric name -> plain delta vs baseline, in the metric's own unit


@dataclass(frozen=True)
class SweepReport:
    baseline_policy_name: str
    windows: tuple[ShiftWindow, ...]
    rows: tuple[ComparisonRow, ...]

    def rows_for_window(self, window_label: str) -> list[ComparisonRow]:
        return [r for r in self.rows if r.window_label == window_label]


def run_sweep(
    policies: Sequence[Policy],
    run_shift: ShiftRunner,
    seeds: Sequence[int],
    windows: Sequence[ShiftWindow] | None = None,
    *,
    baseline_policy_name: str | None = None,
) -> SweepReport:
    """Run every policy over every window over every seed, on the identical
    frozen scenario per (window, seed) pair, and build the comparison
    report.

    `baseline_policy_name` defaults to `policies[0].name`. Every other
    policy's metrics get a delta-vs-baseline column, per window.
    """
    if not policies:
        raise ValueError("run_sweep requires at least one policy")
    if not seeds:
        raise ValueError("run_sweep requires at least one seed")

    windows = tuple(windows) if windows is not None else tuple(SHIFT_WINDOWS.values())
    baseline_name = baseline_policy_name or policies[0].name

    rows: list[ComparisonRow] = []
    for window in windows:
        summaries: dict[str, PolicyWindowSummary] = {}
        for policy in policies:
            per_seed_metrics: list[ShiftMetrics] = []
            for seed in seeds:
                result: ShiftResult = run_shift(policy, seed, window.start_min, window.end_min)
                per_seed_metrics.append(compute_metrics(result))
            summaries[policy.name] = _summarize(policy.name, window, per_seed_metrics)

        if baseline_name not in summaries:
            raise ValueError(f"baseline policy {baseline_name!r} was not among the policies run")
        baseline_summary = summaries[baseline_name]

        for policy in policies:
            summary = summaries[policy.name]
            is_baseline = policy.name == baseline_name
            delta_pct: dict[str, float | None] = {}
            delta_points: dict[str, float | None] = {}
            delta_abs: dict[str, float | None] = {}
            for metric_name in _PERCENT_METRICS:
                base = baseline_summary.aggregated.get(metric_name)
                cand = summary.aggregated.get(metric_name)
                delta_pct[metric_name] = (
                    None if is_baseline or base is None or cand is None else _delta_pct(base.mean, cand.mean)
                )
            for metric_name in _POINT_METRICS:
                base = baseline_summary.aggregated.get(metric_name)
                cand = summary.aggregated.get(metric_name)
                delta_points[metric_name] = (
                    None if is_baseline or base is None or cand is None else cand.mean - base.mean
                )
            for metric_name in _ABSOLUTE_METRICS:
                base = baseline_summary.aggregated.get(metric_name)
                cand = summary.aggregated.get(metric_name)
                delta_abs[metric_name] = (
                    None if is_baseline or base is None or cand is None else cand.mean - base.mean
                )
            rows.append(
                ComparisonRow(
                    window_label=window.label,
                    policy_name=policy.name,
                    summary=summary,
                    is_baseline=is_baseline,
                    delta_pct=delta_pct,
                    delta_points=delta_points,
                    delta_abs=delta_abs,
                )
            )

    return SweepReport(baseline_policy_name=baseline_name, windows=windows, rows=tuple(rows))


_COLUMN_LABELS: dict[str, str] = {
    "gross_mxn": "gross MXN",
    "mxn_per_hour": "MXN/h",
    "deliveries": "deliveries",
    "deliveries_per_hour": "deliveries/h",
    "km_traveled": "km",
    "unpaid_km": "unpaid km",
    "idle_fraction": "idle %",
    "acceptance_rate": "accept %",
    "unpaid_km_fraction": "unpaid km %",
    "surge_capture_accepted_share": "surge accepted %",
    "surge_capture_offered_share": "surge offered %",
    "surge_capture_edge": "surge edge pp",
    "average_fare_mxn": "avg fare MXN",
    "median_fare_mxn": "median fare MXN",
    "fare_calibration_mean_error_mxn": "fare calib mean err MXN",
    "fare_calibration_stdev_error_mxn": "fare calib stdev MXN",
    "end_of_shift_distance_from_home_km": "end dist. home km",
}


def render_comparison_table(report: SweepReport) -> str:
    """Render a plain-text comparison table: one block per window, one row
    per policy, mean +/- stdev per metric plus delta vs baseline."""
    lines: list[str] = []
    for window in report.windows:
        lines.append(f"=== {window.label_with_hours} ===")
        header = ["policy"] + [_COLUMN_LABELS[m] for m in _METRIC_FIELDS] + ["delta vs baseline"]
        lines.append(" | ".join(header))
        for row in report.rows_for_window(window.label):
            cells = [row.policy_name + (" (baseline)" if row.is_baseline else "")]
            for metric_name in _METRIC_FIELDS:
                agg = row.summary.aggregated.get(metric_name)
                cells.append(f"{agg.mean:.2f} +/- {agg.stdev:.2f}" if agg else "n/a")
            if row.is_baseline:
                cells.append("-")
            else:
                deltas = []
                for metric_name in _PERCENT_METRICS:
                    d = row.delta_pct.get(metric_name)
                    if d is not None:
                        deltas.append(f"{_COLUMN_LABELS[metric_name]} {d:+.1f}%")
                for metric_name in _POINT_METRICS:
                    d = row.delta_points.get(metric_name)
                    if d is not None:
                        deltas.append(f"{_COLUMN_LABELS[metric_name]} {d * 100:+.1f}pp")
                for metric_name in _ABSOLUTE_METRICS:
                    d = row.delta_abs.get(metric_name)
                    if d is not None:
                        deltas.append(f"{_COLUMN_LABELS[metric_name]} {d:+.2f}")
                cells.append("; ".join(deltas) if deltas else "n/a")
            lines.append(" | ".join(cells))
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "AggregatedMetric",
    "ComparisonRow",
    "PolicyWindowSummary",
    "SHIFT_WINDOWS",
    "ShiftRunner",
    "ShiftWindow",
    "SweepReport",
    "render_comparison_table",
    "run_sweep",
]
