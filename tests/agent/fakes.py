"""A hand-built `RawSourcePort`, and why it is better than the old fixtures.

The policy used to be tested by handing it a finished `Observation`. That
fixture could state what the agent knew; it could not state what the agent
COULD NOT FIND OUT, because every field had to hold something. An unvisited
kitchen and a kitchen remembered at nine minutes were both just entries in a
dict.

A fake port can express ignorance exactly, which is the interesting half:

  - a kitchen never visited returns `None`, not a prior;
  - a cell the traffic app does not cover has no congestion reading at all;
  - a zone with no fitted ETA bias returns `None`, which is the whole of a
    first shift in a new city;
  - and `queries` records every question the agent actually asked, so a test
    can assert on the agent's curiosity rather than only on its answer.

Nothing here touches `src/world/`, `src/engine/`, a file or a network. The
port boundary is what makes that possible, and these fakes are the proof
that it holds.
"""

from __future__ import annotations

import math

from src.core.ports import Estimate, PerceivedEvent

EARTH_RADIUS_KM = 6371.0088

# What a straight line becomes on real streets, and how fast a scooter
# covers it door to door. Deliberately FLAT here: the real adapter reads a
# per-corridor route factor and speed off its OSM skeleton, and a test
# comparing two offers wants one number it can reason about, not a road
# network.
DEFAULT_DETOUR_FACTOR = 1.35
DEFAULT_SCOOTER_KMH = 22.0
DEFAULT_MIN_LEG_MINUTES = 1.0
DEFAULT_SAME_PLACE_KM = 0.05

# Confidence the real adapter reports for a purely structural travel answer
# — skeleton plus prior, nothing measured. The agent reads its live
# congestion multiplier at full weight at this value (see
# `src.agent.calibration.TRAVEL_CALIBRATION`), so a fake that reports it is
# a fake with nothing learned.
STRUCTURAL_MINUTES_CONFIDENCE = 0.55
KM_CONFIDENCE = 0.85


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    h = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, h))))


class FakeRawSource:
    """Every answer stated explicitly; everything unstated is unknowable.

    `poi_density` is what the courier's map app reports per cell: how much
    food commerce sits there, on a 0-1 scale. It is NOT a believed demand —
    the agent multiplies it by its own hour-of-day rhythm to get that, which
    is exactly the composition being tested. `tests.agent.factories` has a
    helper that inverts the rhythm for a test that wants to state a believed
    demand directly.
    """

    def __init__(
        self,
        *,
        temp_c: float = 24.0,
        apparent_c: float | None = None,
        precip_mm_per_hour: float = 0.0,
        congestion: dict[str, Estimate] | None = None,
        poi_density: dict[str, Estimate] | None = None,
        disruptions: tuple[PerceivedEvent, ...] = (),
        cell_coords: dict[str, tuple[float, float]] | None = None,
        kitchen: dict[str, Estimate] | None = None,
        eta_bias: dict[str, Estimate] | None = None,
        travel_correction: Estimate | None = None,
        detour_factor: float = DEFAULT_DETOUR_FACTOR,
        scooter_kmh: float = DEFAULT_SCOOTER_KMH,
    ) -> None:
        self._temp_c = temp_c
        self._apparent_c = apparent_c if apparent_c is not None else temp_c + 1.0
        self._precip_mm_per_hour = precip_mm_per_hour
        self._congestion = dict(congestion or {})
        self._poi_density = dict(poi_density or {})
        self._disruptions = disruptions
        self._cell_coords = dict(cell_coords or {})
        self._kitchen = dict(kitchen or {})
        self._eta_bias = dict(eta_bias or {})
        self._travel_correction = travel_correction
        self._detour_factor = detour_factor
        self._scooter_kmh = scooter_kmh

        # Every question asked of this port, in order. A test can assert the
        # agent went looking for something — or never did.
        self.queries: list[tuple] = []
        # Everything the engine told this port. Write-only unless a test
        # reads it back.
        self.recorded_kitchens: list[tuple[str, float, int]] = []
        self.recorded_trips: list[tuple[str, str, int, float, float, float]] = []
        self.refits = 0

    # -- weather ---------------------------------------------------------

    def weather_at(self, lat: float, lon: float, minute: int) -> dict[str, Estimate]:
        self.queries.append(("weather_at", round(lat, 5), round(lon, 5), minute))
        return {
            "temp_c": Estimate(self._temp_c, 0.95, 1.0),
            "apparent_c": Estimate(self._apparent_c, 0.90, 1.0),
            "precip_mm_per_hour": Estimate(self._precip_mm_per_hour, 0.65, 1.0),
        }

    # -- road network ----------------------------------------------------

    def travel_estimate(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, minute: int
    ) -> tuple[Estimate, Estimate]:
        self.queries.append(("travel_estimate", round(from_lat, 5), round(from_lon, 5),
                             round(to_lat, 5), round(to_lon, 5), minute))
        straight = haversine_km(from_lat, from_lon, to_lat, to_lon)
        if straight < DEFAULT_SAME_PLACE_KM:
            return Estimate(0.0, 1.0, 0.0), Estimate(0.0, 1.0, 0.0)
        km = straight * self._detour_factor
        minutes = km / self._scooter_kmh * 60.0
        if self._travel_correction is None:
            confidence = STRUCTURAL_MINUTES_CONFIDENCE
        else:
            minutes *= self._travel_correction.value
            confidence = self._travel_correction.confidence
        return (
            Estimate(km, KM_CONFIDENCE, 0.0),
            Estimate(max(DEFAULT_MIN_LEG_MINUTES, minutes), confidence, 0.0),
        )

    def congestion_near(
        self, lat: float, lon: float, radius_km: float, minute: int
    ) -> dict[str, Estimate]:
        self.queries.append(("congestion_near", round(lat, 5), round(lon, 5), radius_km, minute))
        return dict(self._congestion)

    # -- POIs ------------------------------------------------------------

    def poi_density_near(self, lat: float, lon: float, radius_km: float) -> dict[str, Estimate]:
        self.queries.append(("poi_density_near", round(lat, 5), round(lon, 5), radius_km))
        return dict(self._poi_density)

    # -- disruptions -----------------------------------------------------

    def perceived_disruptions(
        self, lat: float, lon: float, minute: int
    ) -> tuple[PerceivedEvent, ...]:
        self.queries.append(("perceived_disruptions", round(lat, 5), round(lon, 5), minute))
        return self._disruptions

    # -- geography -------------------------------------------------------

    def cell_coords(self, cells: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        self.queries.append(("cell_coords", tuple(cells)))
        return {cell: self._cell_coords[cell] for cell in cells if cell in self._cell_coords}

    # -- history ---------------------------------------------------------

    def recall_kitchen(self, denue_id: str) -> Estimate | None:
        """`None` for a branch never visited. That is the honest answer and
        the one the old fixture could not give."""
        self.queries.append(("recall_kitchen", denue_id))
        return self._kitchen.get(denue_id)

    def record_kitchen(self, denue_id: str, observed_minutes: float, minute: int) -> None:
        self.recorded_kitchens.append((denue_id, observed_minutes, minute))

    def recall_eta_bias(self, cell: str) -> Estimate | None:
        self.queries.append(("recall_eta_bias", cell))
        return self._eta_bias.get(cell)

    def record_trip(
        self,
        from_cell: str,
        to_cell: str,
        minute: int,
        promised_minutes: float,
        actual_minutes: float,
        actual_km: float,
    ) -> None:
        self.recorded_trips.append(
            (from_cell, to_cell, minute, promised_minutes, actual_minutes, actual_km)
        )

    def refit(self) -> dict[str, int]:
        self.refits += 1
        return {"trips": len(self.recorded_trips), "kitchens": len(self.recorded_kitchens)}

    # -- test helpers ----------------------------------------------------

    def asked(self, method: str) -> list[tuple]:
        return [q for q in self.queries if q[0] == method]
