"""Everything the agent has learned, and the only thing that survives a shift.

Two tables are fitted here, and both matter for the same reason: they are
the half of the agent that is not a prior. On arrival in a new city this
object is EMPTY, every query returns `None`, and the agent has to fall back
on structural priors it knows it barely believes. Three shifts later it has
measured the city for itself.

    travel correction   realised minutes against the OSM skeleton's
                        free-flow minutes, by zone and hour. This is the
                        agent's own measurement of how long things actually
                        take on its own vehicle, over its own routes — see
                        `osm_travel.py` for why that is the correct
                        measurement rather than a cheap substitute for a
                        traffic feed.

    app-ETA bias        what the app promised against what the trip took,
                        by drop-off zone. Entirely self-learned, needs no
                        external source at all, and it is the cleanest
                        arbitrage available to a courier: after a hundred
                        trips they know the bias per zone better than the
                        platform will admit.

THE SPLIT THAT MAKES A SEVEN-SECOND BUDGET WORK. Recording is free and
happens live; FITTING happens in `refit()`, which is called BETWEEN shifts
and never inside a decision. So a query during a shift is a dict lookup
against whatever the last fit produced, and on shift one that is nothing.
Kitchen memory is the deliberate exception — it updates live, because
"this branch is always slow" is knowledge a courier has by the second
pickup, not by the next morning.

Both tables are keyed ONLY on things the agent can observe: a cell id it was
told about, and the hour on its own clock. Neither is keyed on anything from
`src/world/`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.ports import Estimate
from src.enrichment.calibration import HISTORY_FIT_CALIBRATION
from src.enrichment.kitchen_memory import KitchenMemory


@dataclass(frozen=True)
class TripRecord:
    """One completed trip, as the courier experienced it.

    `structural_minutes` is what the OSM skeleton predicted for this pair
    before any correction. Storing it alongside the realised time is what
    makes the correction fittable later: the ratio of the two IS the thing
    being learned, and recomputing it at fit time would need the skeleton
    that produced it, which may since have been rebuilt.
    """

    from_cell: str
    to_cell: str
    minute: int
    promised_minutes: float
    actual_minutes: float
    actual_km: float
    structural_minutes: float


def _hour_bucket(minute: int) -> int:
    """The hour-of-day bucket a minute falls in. Wraps at midnight, because
    a shift does not: minutes are a continuous counter and 1500 is 01:00."""
    hours = int(HISTORY_FIT_CALIBRATION["hour_bucket_hours"])
    return int((minute % 1440) // 60) // max(1, hours)


@dataclass
class _Fitted:
    """One fitted cell of a relationship table: a ratio and its sample size.

    The sample size is not decoration. It is what stops a zone visited twice
    speaking with the authority of one visited two hundred times, and it is
    therefore what makes a cold start conservative rather than reckless.
    """

    ratio: float
    samples: int


class CourierHistory:
    """The agent's accumulated experience, across shifts.

    Handed from one shift's `RawSourceAdapter` to the next so the agent's
    memory outlives the scenario object. A fresh instance is a courier who
    has never worked this city.
    """

    def __init__(self) -> None:
        self.kitchen = KitchenMemory()
        self._trips: list[TripRecord] = []
        self._travel_correction: dict[tuple[str, int], _Fitted] = {}
        self._travel_pooled: _Fitted | None = None
        self._eta_bias: dict[str, _Fitted] = {}
        self._eta_pooled: _Fitted | None = None
        self._fits = 0

    # -- recording (live, free) ------------------------------------------

    def record_kitchen(self, denue_id: str, observed_minutes: float, minute: int) -> None:
        if not denue_id:
            return
        self.kitchen.record_visit(denue_id, observed_minutes, minute)

    def recall_kitchen(self, denue_id: str, minute: int) -> Estimate | None:
        if not denue_id:
            return None
        return self.kitchen.snapshot(minute).get(denue_id)

    def record_trip(self, trip: TripRecord) -> None:
        self._trips.append(trip)

    @property
    def trips(self) -> tuple[TripRecord, ...]:
        return tuple(self._trips)

    @property
    def fits(self) -> int:
        """How many times this history has been re-fitted. Zero means every
        fitted query returns `None` — which is exactly the cold start."""
        return self._fits

    # -- reading the fitted tables (a dict lookup, inside 7 s) -----------

    def recall_travel_correction(self, to_cell: str, minute: int) -> Estimate | None:
        """How much longer a leg into this zone at this hour really takes
        than the free-flow skeleton says. `None` until `refit()` has run on
        evidence — the agent then rides on the structural prior alone and
        knows it."""
        return self._lookup(
            self._travel_correction.get((to_cell, _hour_bucket(minute))), self._travel_pooled
        )

    def recall_eta_bias(self, cell: str) -> Estimate | None:
        """How much the app's ETA has lied in this zone: realised over
        promised. Above 1.0 means the app is optimistic, which it is."""
        return self._lookup(self._eta_bias.get(cell), self._eta_pooled)

    def _lookup(self, local: _Fitted | None, pooled: _Fitted | None) -> Estimate | None:
        cal = HISTORY_FIT_CALIBRATION
        if local is not None:
            samples = local.samples
            ratio = local.ratio
        elif pooled is not None:
            # No evidence about this zone, but the city as a whole has
            # spoken. Believed at the pooled sample's own weight, which is
            # the honest amount: it is a fact about the city, not this zone.
            samples = pooled.samples
            ratio = pooled.ratio
        else:
            return None
        confidence = min(
            cal["max_confidence"],
            cal["first_trip_confidence"]
            + (cal["max_confidence"] - cal["first_trip_confidence"])
            * (samples / (samples + cal["confidence_trips"])),
        )
        return Estimate(value=ratio, confidence=confidence, age_minutes=0.0)

    # -- the offline fit -------------------------------------------------

    def refit(self) -> dict[str, int]:
        """Re-fit both tables from accumulated trips. Called BETWEEN shifts.

        Every (zone, hour) ratio is shrunk toward the city-wide pooled
        ratio with weight `n / (n + shrinkage_trips)`, so a bucket with one
        trip behind it barely moves off the city's average and a bucket with
        forty is mostly its own. That is the whole reason a first shift in
        an unknown city is conservative instead of confidently wrong.

        Returns sample counts per table so a caller can see what the agent
        actually had evidence for, rather than trusting that it had any.
        """
        cal = HISTORY_FIT_CALIBRATION

        travel_samples: dict[tuple[str, int], list[float]] = {}
        eta_samples: dict[str, list[float]] = {}
        for trip in self._trips:
            if trip.structural_minutes > 0.0 and trip.actual_minutes > 0.0:
                ratio = trip.actual_minutes / trip.structural_minutes
                if cal["min_ratio"] <= ratio <= cal["max_ratio"]:
                    travel_samples.setdefault(
                        (trip.to_cell, _hour_bucket(trip.minute)), []
                    ).append(ratio)
            if trip.promised_minutes > 0.0 and trip.actual_minutes > 0.0:
                ratio = trip.actual_minutes / trip.promised_minutes
                if cal["min_ratio"] <= ratio <= cal["max_ratio"]:
                    eta_samples.setdefault(trip.to_cell, []).append(ratio)

        self._travel_correction, self._travel_pooled = self._fit_table(travel_samples)
        self._eta_bias, self._eta_pooled = self._fit_table(eta_samples)
        self._fits += 1

        return {
            "trips": len(self._trips),
            "travel_correction_cells": len(self._travel_correction),
            "travel_correction_samples": sum(len(v) for v in travel_samples.values()),
            "eta_bias_cells": len(self._eta_bias),
            "eta_bias_samples": sum(len(v) for v in eta_samples.values()),
            "kitchens": len(self.kitchen.snapshot(0)),
        }

    @staticmethod
    def _fit_table(samples):
        cal = HISTORY_FIT_CALIBRATION
        flat = [value for values in samples.values() for value in values]
        if not flat:
            return {}, None
        pooled_ratio = sum(flat) / len(flat)
        pooled = _Fitted(ratio=pooled_ratio, samples=len(flat))
        fitted = {}
        for key, values in samples.items():
            n = len(values)
            own = sum(values) / n
            weight = n / (n + cal["shrinkage_trips"])
            fitted[key] = _Fitted(
                ratio=pooled_ratio + (own - pooled_ratio) * weight, samples=n
            )
        return fitted, pooled
