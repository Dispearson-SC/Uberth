"""Ground truth: the Scenario container (exogenous world, frozen at seed time).

This module is ground truth. Everything a courier cannot influence —
weather, traffic, incidents, the demand field, and the order stream — is
precomputed once from `seed` and frozen into a `Scenario` before any policy
runs. Two policies evaluated on the same Scenario face, by construction, an
identical world: the world never depends on agent actions, so replay is
trivial and A/B comparison is hermetic. `src/agent/` must never import this
module directly.
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from src.world.timeline import Event, OrderOffer, SupplyTick, TrafficTick, WeatherTick

# Default shift: Friday, 14:00-22:00 (minute 840 to minute 1320), 8 hours.
DEFAULT_DAY_OF_WEEK = "Friday"
DEFAULT_SHIFT_START_MIN = 840
DEFAULT_SHIFT_END_MIN = 1320

# Named RNG streams, one per exogenous concern. A single global RNG is the
# classic mistake here: tweaking one event type (e.g. incident rate) would
# reshuffle an unrelated stream (e.g. the order stream) and destroy
# comparability between runs/policies.
RNG_STREAM_NAMES: tuple[str, ...] = (
    "orders",
    "weather",
    "traffic",
    "incidents",
    "kitchen",
    "competitors",
    "observation_noise",
)


def rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """Spawn one independent `numpy.random.Generator` per concern from a
    single seed via `SeedSequence.spawn`."""
    seed_sequence = np.random.SeedSequence(seed)
    children = seed_sequence.spawn(len(RNG_STREAM_NAMES))
    return {name: np.random.default_rng(child) for name, child in zip(RNG_STREAM_NAMES, children)}


class Scenario(BaseModel):
    """Frozen exogenous world for one simulated shift.

    The timelines are populated by the exogenous producers (weather.py,
    traffic.py, events.py, demand.py), each written independently against
    the typed schemas in `timeline.py`. Declaring them up front keeps
    `Scenario` a stable, versionable, hand-editable container: a scenario
    file can be authored by hand to place a specific event at a specific
    minute for a demo, instead of hoping the RNG cooperates.
    """

    seed: int
    date: Date
    day_of_week: str = DEFAULT_DAY_OF_WEEK
    shift_start_min: int = DEFAULT_SHIFT_START_MIN
    shift_end_min: int = DEFAULT_SHIFT_END_MIN

    # Typed exogenous timelines. Their schemas live in `timeline.py`, which is
    # the coordination contract between the independently-written producers
    # (weather.py, traffic.py, events.py, demand.py) and the engine. Leaving
    # these as untyped dicts would let each producer invent its own shape.
    weather_timeline: list[WeatherTick] = Field(default_factory=list)
    traffic_timeline: list[TrafficTick] = Field(default_factory=list)
    events_timeline: list[Event] = Field(default_factory=list)
    supply_timeline: list[SupplyTick] = Field(default_factory=list)
    order_stream: list[OrderOffer] = Field(default_factory=list)

    def rng_streams(self) -> dict[str, np.random.Generator]:
        return rng_streams(self.seed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        path = Path(path)
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
