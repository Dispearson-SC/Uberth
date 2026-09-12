"""Derive a compact, auditable 24-hour Monterrey traffic profile from the raw
TomTom Traffic Stats Area Analysis fixture.

INPUT (never loaded into memory as a whole; 2.63 GB on disk):
    fixtures/raw/tomtom_monterrey_areaanalysis.json
This is TomTom's completed Traffic Stats Area Analysis for our exact
operating-area polygon (Monterrey, San Pedro, San Nicolas, Guadalupe),
July 2026, weekday-only ("WD-00" .. "WD-23"), FRC 0-7. It is streamed with
`ijson` in a single pass over `network.segmentResults` (the segment array is
nested under the top-level `network` object, not at the document root — the
real key path was found empirically with a small `ijson.parse` probe before
writing this script; it does not match the flatter path guessed from the
field list alone). `json.load` must never be called on this file.

OUTPUT (small, committable, human-readable):
    fixtures/monterrey_hourly_profile.csv
24 rows, one per hour of day, with columns:
    hour, harmonic_speed_kmh, congestion_multiplier, sample_size,
    total_distance_km, segment_count

METHOD
------
1. Read the `timeSets` header (small, 24 entries) to map each numeric
   `timeSet` id used inside `segmentTimeResults` to its hour of day, via the
   `WD-HH` naming convention TomTom used for this job (e.g. id 16 -> "WD-14"
   -> hour 14). This mapping is *read from the file*, not hardcoded, so it
   stays correct even if TomTom ever reorders `@id` allocation.

2. Stream every segment object once (`network.segmentResults.item`). Each
   segment carries a `distance` (meters) and, per time set, a
   `harmonicAverageSpeed` (km/h) and `sampleSize` (probe count for that
   segment-hour).

3. Aggregate to one number per hour using the DISTANCE-WEIGHTED HARMONIC MEAN
   of segment speeds — i.e. total distance divided by total travel time:

        speed(h) = sum(distance_i) / sum(distance_i / speed_i)

   This is the only aggregation that is dimensionally correct for a road
   network: travel time is what actually accumulates additively along a
   route, not speed. A segment's contribution to network travel time is
   `distance_i / speed_i` hours, so this ratio *is* the harmonic mean,
   weighted by distance, computed directly from segment-level data (using
   TomTom's own `harmonicAverageSpeed` per segment rather than
   `averageSpeed`, which is already an arithmetic mean and would double the
   averaging error if combined arithmetically again here). A plain
   arithmetic mean of per-segment speeds would let a handful of short, fast
   segments (e.g. a 50 m residential cut-through at 40 km/h) outweigh a long,
   slow arterial (a 5 km avenue crawling at 12 km/h) exactly backwards from
   how a courier actually experiences the network.

4. LOW-SAMPLE FILTER: a segment-hour is dropped from the aggregate if its
   `sampleSize` is below MIN_SAMPLE_SIZE (see the calibration dict below).
   Floating-car speed samples in the single digits are dominated by the
   idiosyncrasies of 1-4 individual vehicles (a driver who stopped for
   coffee, a single red-light cycle) rather than network conditions, and
   TomTom's own `normalizedSampleSize` field confirms these are a small
   fraction of total probe volume. The exact drop count is printed by this
   script for every run, so the threshold's effect is never silently baked
   into the output.

5. FREE-FLOW REFERENCE HOUR: this fixture has no `freeFlowSpeed` field (Area
   Analysis does not ship one), so free flow is defined empirically as the
   network's own fastest hour post-aggregation — expected, and confirmed by
   this script's printed output, to be the quiet pre-dawn window
   (03:00-04:00). The congestion multiplier is then:

        congestion_multiplier(h) = speed(reference_hour) / speed(h)

   which is exactly 1.0 at the reference hour by construction and grows
   above 1.0 as an hour gets slower than that reference. This script prints
   which hour actually won the reference-hour search, so the choice is
   auditable rather than assumed to be 03:00 or 04:00.

Run from the project root so `fixtures/` resolves relative to cwd:
    .venv\\Scripts\\python.exe scripts\\derive_monterrey_traffic_profile.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import ijson

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CALIBRATION — the only tuning knob in this script. Segment-hours with
# fewer than this many floating-car speed samples are excluded from the
# hourly aggregate as statistically unreliable (see method note 4 above).
MIN_SAMPLE_SIZE = 5

RAW_FIXTURE_PATH = PROJECT_ROOT / "fixtures" / "raw" / "tomtom_monterrey_areaanalysis.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "fixtures" / "monterrey_hourly_profile.csv"

CSV_COLUMNS = [
    "hour",
    "harmonic_speed_kmh",
    "congestion_multiplier",
    "sample_size",
    "total_distance_km",
    "segment_count",
]


def _to_float(value: object) -> float | None:
    """TomTom emits numeric fields as strings in this fixture (e.g.
    `"harmonicAverageSpeed": "23.6"`). Returns None for missing/blank/
    unparsable values rather than raising, so one malformed field drops only
    that single segment-hour, not the whole run."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def build_timeset_hour_map(raw_path: Path) -> dict[int, int]:
    """Read the small `timeSets` header and map each numeric `timeSet` id
    (referenced inside every segment's `segmentTimeResults`) to its hour of
    day 0-23, via the `WD-HH` naming convention. Read from the file rather
    than hardcoded so a reordering of TomTom's internal `@id` allocation
    cannot silently mis-map hours."""
    mapping: dict[int, int] = {}
    with raw_path.open("rb") as f:
        for time_set in ijson.items(f, "timeSets.item"):
            name = time_set["name"]  # e.g. "WD-14"
            prefix, _, hour_str = name.partition("-")
            if prefix != "WD":
                raise ValueError(
                    f"Unexpected time set name {name!r}: this fixture was expected to be "
                    "weekday-only (WD-00..WD-23); a non-weekday time set means the fixture "
                    "changed shape and this script's assumptions must be revisited."
                )
            mapping[time_set["@id"]] = int(hour_str)
    if sorted(mapping.values()) != list(range(24)):
        raise ValueError(f"Expected exactly hours 0..23 from timeSets, got {sorted(mapping.values())}")
    return mapping


def aggregate_hourly_profile(raw_path: Path, timeset_to_hour: dict[int, int]) -> tuple[list[dict], dict]:
    """Single streaming pass over every segment in `network.segmentResults`,
    accumulating per-hour (distance, travel-time, sample-size, segment-count)
    sums. Returns the 24 per-hour rows plus a diagnostics dict (segment
    count, dropped counts, elapsed time) for the printed report."""
    sum_distance_km = {h: 0.0 for h in range(24)}
    sum_travel_time_hours = {h: 0.0 for h in range(24)}
    sum_sample_size = {h: 0 for h in range(24)}
    segment_count = {h: 0 for h in range(24)}

    total_segments = 0
    total_segment_hours_seen = 0
    dropped_low_sample = 0
    dropped_missing_speed_or_distance = 0

    start = time.time()
    with raw_path.open("rb") as f:
        for segment in ijson.items(f, "network.segmentResults.item"):
            total_segments += 1
            distance_km_raw = _to_float(segment.get("distance"))
            if distance_km_raw is None:
                # No usable distance at all: every time-result for this
                # segment is unusable, since the weighting scheme needs it.
                dropped_missing_speed_or_distance += len(segment.get("segmentTimeResults", []))
                continue
            distance_km = distance_km_raw / 1000.0  # `distance` is meters

            for time_result in segment.get("segmentTimeResults", []):
                total_segment_hours_seen += 1
                hour = timeset_to_hour.get(time_result.get("timeSet"))
                if hour is None:
                    continue  # defensive: unrecognized time set id

                sample_size = time_result.get("sampleSize", 0) or 0
                if sample_size < MIN_SAMPLE_SIZE:
                    dropped_low_sample += 1
                    continue

                speed_kmh = _to_float(time_result.get("harmonicAverageSpeed"))
                if speed_kmh is None:
                    dropped_missing_speed_or_distance += 1
                    continue

                sum_distance_km[hour] += distance_km
                sum_travel_time_hours[hour] += distance_km / speed_kmh
                sum_sample_size[hour] += sample_size
                segment_count[hour] += 1

    elapsed = time.time() - start

    rows = []
    for hour in range(24):
        travel_time_h = sum_travel_time_hours[hour]
        distance_km = sum_distance_km[hour]
        harmonic_speed = distance_km / travel_time_h if travel_time_h > 0 else float("nan")
        rows.append(
            {
                "hour": hour,
                "harmonic_speed_kmh": harmonic_speed,
                "sample_size": sum_sample_size[hour],
                "total_distance_km": distance_km,
                "segment_count": segment_count[hour],
            }
        )

    diagnostics = {
        "total_segments": total_segments,
        "total_segment_hours_seen": total_segment_hours_seen,
        "dropped_low_sample": dropped_low_sample,
        "dropped_missing_speed_or_distance": dropped_missing_speed_or_distance,
        "elapsed_seconds": elapsed,
    }
    return rows, diagnostics


def attach_congestion_multiplier(rows: list[dict]) -> int:
    """Free-flow reference hour = the network's own fastest aggregated hour
    (empirical, no `freeFlowSpeed` field exists in this fixture — see module
    docstring method note 5). Mutates `rows` in place, adding
    `congestion_multiplier`. Returns the winning reference hour."""
    reference_row = max(rows, key=lambda row: row["harmonic_speed_kmh"])
    reference_speed = reference_row["harmonic_speed_kmh"]
    for row in rows:
        row["congestion_multiplier"] = reference_speed / row["harmonic_speed_kmh"]
    return reference_row["hour"]


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "hour": row["hour"],
                    "harmonic_speed_kmh": round(row["harmonic_speed_kmh"], 3),
                    "congestion_multiplier": round(row["congestion_multiplier"], 4),
                    "sample_size": row["sample_size"],
                    "total_distance_km": round(row["total_distance_km"], 2),
                    "segment_count": row["segment_count"],
                }
            )


def print_report(rows: list[dict], diagnostics: dict, reference_hour: int) -> None:
    print("=" * 78)
    print("Monterrey hourly traffic profile — derived from TomTom Area Analysis")
    print("=" * 78)
    print(f"Source fixture      : {RAW_FIXTURE_PATH}")
    print(f"Segments streamed   : {diagnostics['total_segments']:,}")
    print(f"Segment-hours seen  : {diagnostics['total_segment_hours_seen']:,}")
    print(
        f"Dropped (sample size < {MIN_SAMPLE_SIZE})"
        f"     : {diagnostics['dropped_low_sample']:,} segment-hours "
        f"({100 * diagnostics['dropped_low_sample'] / max(diagnostics['total_segment_hours_seen'], 1):.2f}%)"
    )
    print(
        f"Dropped (missing distance/speed): {diagnostics['dropped_missing_speed_or_distance']:,} segment-hours"
    )
    print(f"Streaming time      : {diagnostics['elapsed_seconds']:.1f}s")
    print(f"Free-flow reference : hour {reference_hour:02d}:00 (fastest aggregated hour, empirically chosen)")
    print()

    max_speed = max(row["harmonic_speed_kmh"] for row in rows)
    print(f"{'hour':>4}  {'speed_kmh':>9}  {'mult':>5}  {'samples':>9}  {'dist_km':>9}  {'segs':>7}  chart")
    for row in rows:
        bar_len = int(round(30 * row["harmonic_speed_kmh"] / max_speed)) if max_speed > 0 else 0
        bar = "#" * bar_len
        marker = " <- reference (free flow)" if row["hour"] == reference_hour else ""
        print(
            f"{row['hour']:>4}  {row['harmonic_speed_kmh']:>9.2f}  {row['congestion_multiplier']:>5.2f}  "
            f"{row['sample_size']:>9,}  {row['total_distance_km']:>9.1f}  {row['segment_count']:>7,}  "
            f"{bar}{marker}"
        )
    print()
    print(f"Wrote {OUTPUT_CSV_PATH}")


def main() -> None:
    timeset_to_hour = build_timeset_hour_map(RAW_FIXTURE_PATH)
    rows, diagnostics = aggregate_hourly_profile(RAW_FIXTURE_PATH, timeset_to_hour)
    reference_hour = attach_congestion_multiplier(rows)
    write_csv(rows, OUTPUT_CSV_PATH)
    print_report(rows, diagnostics, reference_hour)


if __name__ == "__main__":
    main()
