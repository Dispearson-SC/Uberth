"""Fetch a real Monterrey hourly congestion profile from the TomTom Traffic Stats API.

Provenance note, and the reason this script exists: TomTom's *free* Traffic
Index downloads cover only 11 cities and Monterrey is not among them, so the
traffic module originally borrowed the daily shape from TomTom's real Mexico
City series. This script replaces that borrowed shape with measured Monterrey
data, obtained through a Traffic Stats trial key.

The API is asynchronous: submit an Area Analysis job, poll until it reports
DONE, then download the result. Output lands in `fixtures/raw/` so the rest of
the simulator stays offline and deterministic — no live API calls ever happen
during a simulated shift or a demo.

Usage:
    python scripts/fetch_tomtom_monterrey.py submit
    python scripts/fetch_tomtom_monterrey.py poll <job_id>
    python scripts/fetch_tomtom_monterrey.py run     # submit, then poll to completion
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from shapely.geometry import MultiPoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "fixtures" / "raw"
CELLS_PARQUET = PROJECT_ROOT / "fixtures" / "cells.parquet"
JOB_STATE_PATH = RAW_DIR / "tomtom_job.json"

API_BASE = "https://api.tomtom.com/traffic/trafficstats"
SUBMIT_URL = f"{API_BASE}/areaanalysis/1"
STATUS_URL = f"{API_BASE}/status/1"

# The trial provisions July 2026 data. A congestion profile is meant to be a
# *typical* curve, so we average a whole month rather than take a single day:
# one day would drag in that day's specific incidents, which the events module
# is responsible for injecting separately.
DATE_FROM = "2026-07-01"
DATE_TO = "2026-07-31"
TIME_ZONE = "America/Monterrey"

# Functional Road Classes 0-7: motorways down to local roads. FRC 8 (the most
# minor residential links) is excluded because probe coverage there is sparse
# and noisy.
FRCS = [0, 1, 2, 3, 4, 5, 6, 7]

WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI"]
WEEKEND = ["SAT", "SUN"]

POLYGON_BUFFER_DEG = 0.01  # ~1.1 km, keeps cells on the edge fully inside


def operating_polygon() -> dict:
    """Convex hull of the H3 cell centroids, as GeoJSON with [lon, lat] order."""
    cells = pd.read_parquet(CELLS_PARQUET)
    hull = MultiPoint(list(zip(cells["lon"], cells["lat"]))).convex_hull
    hull = hull.buffer(POLYGON_BUFFER_DEG)
    coords = [[round(x, 6), round(y, 6)] for x, y in hull.exterior.coords]
    return {"type": "Polygon", "coordinates": [coords]}


def hourly_time_sets() -> list[dict]:
    """A 24-point hourly curve for weekdays.

    The trial allows at most 24 time sets, which forces a choice: a full
    24-hour curve for one day type, or a coarser curve for two. We take the
    full weekday curve, because every demo shift is a Friday and because the
    night hours matter (the checkpoint / alcoholimetro window runs 22:00-04:00
    and a coarse profile would smear it).

    The weekend shape therefore stays derived from the weekend/weekday ratio
    of TomTom's Mexico City series applied on top of this real Monterrey
    weekday curve. Weekday simulation is 100% measured Monterrey data;
    weekend simulation is not, and must be labelled that way.
    """
    time_sets = []
    for label, days in (("WD", WEEKDAYS),):
        for hour in range(24):
            time_sets.append(
                {
                    "name": f"{label}-{hour:02d}",
                    # Ranges must be inclusively separate: the API rejects
                    # 00:00-01:00 next to 01:00-02:00 because they share the
                    # 01:00 boundary. Hence HH:00-HH:59.
                    "timeGroups": [{"days": days, "times": [f"{hour:02d}:00-{hour:02d}:59"]}],
                }
            )
    return time_sets


def build_job() -> dict:
    return {
        "jobName": "monterrey-delivery-operating-area-hourly",
        "distanceUnit": "KILOMETERS",
        "network": {
            "name": "Monterrey delivery operating area",
            "geometry": operating_polygon(),
            "timeZoneId": TIME_ZONE,
            "frcs": FRCS,
            "probeSource": "ALL",
        },
        "dateRange": {"name": "July 2026", "from": DATE_FROM, "to": DATE_TO},
        "timeSets": hourly_time_sets(),
    }


def api_key() -> str:
    key = os.environ.get("TOMTOM_API_KEY")
    if not key:
        env_path = PROJECT_ROOT / ".env"
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TOMTOM_API_KEY"):
                key = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not key:
        raise SystemExit("TOMTOM_API_KEY not found in environment or .env")
    return key


def submit() -> str:
    job = build_job()
    n_times = len(job["timeSets"])
    n_coords = len(job["network"]["geometry"]["coordinates"][0])
    print(f"[submit] polygon: {n_coords} vertices | time sets: {n_times} | {DATE_FROM}..{DATE_TO}")

    response = requests.post(SUBMIT_URL, params={"key": api_key()}, json=job, timeout=120)
    print(f"[submit] http {response.status_code}")
    body = response.text
    print(f"[submit] {body[:900]}")
    if response.status_code != 200:
        # Never call raise_for_status here: requests embeds the full request
        # URL, API key included, in the exception message and therefore in
        # any log that captures it.
        raise SystemExit(f"submit failed with http {response.status_code} (see body above)")

    payload = response.json()
    job_id = str(payload.get("jobId") or payload.get("id") or "")
    if not job_id:
        raise SystemExit(f"could not find a job id in the response: {payload}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    JOB_STATE_PATH.write_text(json.dumps({"job_id": job_id, "submitted": payload}, indent=2), encoding="utf-8")
    print(f"[submit] job id: {job_id}")
    return job_id


def poll(job_id: str, max_wait_s: int = 3600, interval_s: int = 30) -> dict | None:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{STATUS_URL}/{job_id}", params={"key": api_key()}, timeout=60)
        except requests.RequestException as exc:
            # Transient DNS/connection failures must not end a poll that may
            # have to run for the better part of an hour.
            print(f"[poll] connection error, retrying: {type(exc).__name__}")
            time.sleep(interval_s)
            continue
        if response.status_code != 200:
            print(f"[poll] http {response.status_code}: {response.text[:300]}")
            time.sleep(interval_s)
            continue

        payload = response.json()
        state = payload.get("jobState") or payload.get("responseStatus")
        print(f"[poll] state={state}")

        if state in {"DONE", "COMPLETED"}:
            return download(payload)
        if state in {"ERROR", "FAILED", "REJECTED"}:
            print(f"[poll] job failed: {json.dumps(payload)[:800]}")
            return None
        time.sleep(interval_s)

    print("[poll] timed out waiting for the job")
    return None


def download(status_payload: dict) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls = status_payload.get("urls") or []
    saved = []
    for url in urls:
        if not isinstance(url, str):
            continue
        suffix = ".json" if "json" in url.lower() else ".dat"
        target = RAW_DIR / f"tomtom_monterrey_hourly{suffix}"
        blob = requests.get(url, params={"key": api_key()}, timeout=300)
        if blob.status_code == 200 and blob.content:
            target.write_bytes(blob.content)
            saved.append((target.name, len(blob.content)))
            print(f"[download] wrote {target.name} ({len(blob.content):,} bytes)")

    if not saved:
        fallback = RAW_DIR / "tomtom_monterrey_status.json"
        fallback.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
        print(f"[download] no downloadable urls; saved raw status to {fallback.name}")
    return status_payload


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "submit":
        submit()
    elif command == "poll":
        poll(sys.argv[2])
    elif command == "run":
        poll(submit())
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
