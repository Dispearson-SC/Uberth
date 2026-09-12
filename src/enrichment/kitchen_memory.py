"""The courier's memory of restaurant speed — earned, never given.

`kitchen_minutes_by_denue_id` on the returned `Observation` starts EMPTY:
there is no cold-start entry for a restaurant the courier has never picked
up from. Confidence tightens with repeat visits via `record_visit`, which
the engine calls once per pickup. This is a real, stateful memory (unlike
every other estimate in this package, which is recomputed fresh each
minute) because "which kitchens are slow" is exactly the kind of knowledge a
real courier accumulates over a shift and would lose on restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.ports import Estimate
from src.enrichment.calibration import KITCHEN_MEMORY_CALIBRATION


@dataclass
class _KitchenRecord:
    observed_minutes: list[float] = field(default_factory=list)
    last_seen_minute: int = 0

    @property
    def mean_minutes(self) -> float:
        return sum(self.observed_minutes) / len(self.observed_minutes)

    @property
    def visit_count(self) -> int:
        return len(self.observed_minutes)


class KitchenMemory:
    """Per-shift, per-courier memory of observed kitchen prep times.

    Deliberately holds NO entry for a restaurant that has not been visited
    — a cold-start prior would have to be a broad, low-confidence guess to
    stay honest (see the module docstring), and this implementation avoids
    the question entirely by not guessing at all: an unvisited restaurant is
    simply absent from `kitchen_minutes_by_denue_id`, exactly like real
    memory of a place you have never been.
    """

    def __init__(self) -> None:
        self._records: dict[str, _KitchenRecord] = {}

    def record_visit(self, denue_id: str, observed_prep_minutes: float, minute: int) -> None:
        """Called by the engine on pickup, with the ACTUAL prep time the
        courier just experienced at this restaurant. Repeat visits to the
        same restaurant tighten confidence (see `KITCHEN_MEMORY_CALIBRATION`)
        rather than each being reported independently."""
        record = self._records.setdefault(denue_id, _KitchenRecord())
        record.observed_minutes.append(observed_prep_minutes)
        record.last_seen_minute = minute

    def snapshot(self, minute: int) -> dict[str, Estimate]:
        """Current belief per visited restaurant, as of `minute`. Restaurants
        never visited are absent — not present with a placeholder value."""
        cal = KITCHEN_MEMORY_CALIBRATION
        out: dict[str, Estimate] = {}
        for denue_id, record in self._records.items():
            n = record.visit_count
            confidence = min(
                cal["max_confidence"],
                1.0 - (1.0 - cal["first_visit_confidence"]) * (cal["decay_per_visit"] ** (n - 1)),
            )
            age_minutes = float(max(minute - record.last_seen_minute, 0))
            out[denue_id] = Estimate(value=record.mean_minutes, confidence=confidence, age_minutes=age_minutes)
        return out
