"""The honesty-critical module: perceived events, and nothing else.

`perceived_events` on the returned `Observation` is built EXCLUSIVELY from
`src.world.events.perceivable_events(...)`. This module does not import
`active_events`, does not accept the raw ground-truth event list as anything
other than the input `perceivable_events` itself filters, and never inspects
`Event.is_active`/`Event.start_min` directly to decide visibility. If a
crash starts at minute 143 with `detect_offset_min=+6`
(`detectable_from_min == 149`), the observation at minute 145 must not
contain it and the observation at minute 149 must — that guarantee lives
entirely inside `perceivable_events`, not here; this module only translates
whatever it returns into the agent-facing shape.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.core.ports import Estimate, PerceivedEvent
from src.enrichment.calibration import EVENT_DELAY_CALIBRATION
from src.world import geo
from src.world.events import perceivable_events
from src.world.timeline import Event


def _anchor_latlon(event: Event) -> tuple[float, float] | None:
    """Best-effort (lat, lon) for display purposes only — never used to
    decide visibility, that is entirely `perceivable_events`'s job."""
    if event.point_lat is not None and event.point_lon is not None:
        return event.point_lat, event.point_lon
    if event.cells:
        return geo.cell_centroid(event.cells[0])
    return None


def _expected_delay_minutes(event: Event, rng: np.random.Generator) -> float:
    """Translate an event's effect into "how many minutes will this cost
    me", the shape a courier actually reasons in — never a raw speed
    multiplier. See `EVENT_DELAY_CALIBRATION` for the (hand-tuned, not
    measured) conversion."""
    cal = EVENT_DELAY_CALIBRATION
    base_leg_minutes = cal["typical_affected_leg_km"] / cal["assumed_free_flow_kmh"] * 60.0

    if event.close_edges:
        delay = cal["closure_detour_minutes"]
    elif event.speed_mult < 1.0:
        delay = base_leg_minutes * (1.0 / event.speed_mult - 1.0)
    else:
        delay = 0.0

    delay += event.fixed_delay_min

    if delay > 0.0:
        noise = float(rng.normal(0.0, cal["value_noise_std_fraction"] * delay))
        delay = max(delay + noise, 0.0)

    return delay


def build_perceived_events(
    events: list[Event],
    minute: int,
    courier_lat: float,
    courier_lon: float,
    graph: nx.MultiDiGraph | None,
    rng: np.random.Generator,
) -> tuple[PerceivedEvent, ...]:
    """The ONLY entry point into ground-truth events from this whole
    package. `events` is handed straight to `perceivable_events`, which does
    all detectability/radius/lifetime filtering; only what it returns is
    ever turned into a `PerceivedEvent`.
    """
    visible = perceivable_events(events, minute, courier_lat, courier_lon, graph=graph)
    visible = sorted(visible, key=lambda e: e.event_id)  # deterministic iteration/rng-consumption order

    perceived: list[PerceivedEvent] = []
    for event in visible:
        anchor = _anchor_latlon(event)
        delay_value = _expected_delay_minutes(event, rng)
        age_minutes = max(float(minute - event.detectable_from_min), 0.0)

        perceived.append(
            PerceivedEvent(
                event_id=event.event_id,
                kind=str(event.type),
                lat=anchor[0] if anchor else None,
                lon=anchor[1] if anchor else None,
                affects_cells=tuple(event.cells) if event.cells else (),
                expected_delay_minutes=Estimate(
                    value=delay_value,
                    confidence=event.confidence,
                    age_minutes=age_minutes,
                ),
                confidence=event.confidence,
            )
        )
    return tuple(perceived)
