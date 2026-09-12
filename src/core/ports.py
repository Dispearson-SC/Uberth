"""Hexagonal ports: the contracts every layer outside `world/` is written against.

This file is the coordination contract, the same role `world/timeline.py` plays
for the exogenous producers. Independent modules are written in parallel
against these protocols; if a type is not declared here, two authors will
invent two incompatible versions of it.

DEPENDENCY DIRECTION — the rule the whole architecture rests on:

    world/          ground truth. Knows nobody. Never imported by agent/.
      |
    core/ports.py   this file. Pure protocols and domain types.
      |
    platform/       driven adapter: the six fields the courier's app shows.
    enrichment/     driven adapter: the courier's own external tools, noisy.
    engine/         driving adapter: the clock that runs a shift.
      |
    agent/          the decision. Imports ONLY core/ports.py.

`src/agent/` importing from `src/world/` is the single failure that would make
this project dishonest: the policy would be able to see the future, and a
judge unpicks that with one question. The boundary is physical, not a
convention — agent code has no import path to ground truth.

Nothing here depends on pydantic, pandas, numpy, OSMnx or the filesystem.
Domain types are plain dataclasses so the core stays framework-free and the
adapters carry the infrastructure.

Hard requirement, load-bearing and repeated everywhere it applies: kilometres
and minutes are separate first-class fields. Never collapse them into one
"distance" or "cost" number. They decouple exactly when traffic or a detour
hits, and that decoupling is what changes which offer is worth taking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

# --------------------------------------------------------------------------
# What the platform shows — deliberately impoverished
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OfferCard:
    """One offer as it appears on the courier's screen.

    This is the ENTIRE information set the platform gives away per offer.
    Adding a field here is a product claim that the real app shows it, so
    do not add one without being able to defend it.
    """

    order_id: str
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    payout_mxn: float
    eta_minutes: float  # the app's own estimate, not ground truth
    distance_km: float  # the app's own estimate, not ground truth
    surge_flag: bool  # coarse and binary, never the real multiplier
    restaurant_name: str
    expires_in_seconds: int  # seconds to decide, the pressure in the brief
    # Stable identifier for the branch, so a policy can join an offer against
    # what it has learned about that kitchen. This leaks nothing: a courier
    # plainly sees which branch they are being sent to, and remembering that
    # this particular one is always slow is exactly the knowledge a good
    # courier accumulates. Without it, `Observation.kitchen_minutes_by_denue_id`
    # is unjoinable from an OfferCard and learned kitchen speed is unusable.
    restaurant_denue_id: str = ""


@dataclass(frozen=True)
class HeatCell:
    """One cell of the in-app demand heatmap: lagged, coarse, quantised."""

    cell: str
    lat: float
    lon: float
    level: int  # quantised bucket, NOT the underlying surge value


@dataclass(frozen=True)
class PlatformView:
    """Everything the app shows at one minute."""

    minute: int
    offers: tuple[OfferCard, ...]
    heatmap: tuple[HeatCell, ...]
    acceptance_rate: float  # the platform shows this back to the courier
    deliveries_completed: int
    earnings_shown_mxn: float


# --------------------------------------------------------------------------
# What the courier's own tools estimate — the edge, and it is never certain
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A value the courier believes, with how sure they are.

    Every field an enrichment source returns is an Estimate, never a bare
    float. A policy that ignores `confidence` is choosing to; it cannot be
    handed certainty it does not have.
    """

    value: float
    confidence: float  # 0..1
    age_minutes: float  # how stale — a 20-minute-old traffic reading is not fresh


@dataclass(frozen=True)
class PerceivedEvent:
    """A disruption the courier could plausibly know about right now.

    Built ONLY from what detectability allows. If a crash starts at minute
    143 and is detectable from 149, this must not exist at minute 145. That
    rule is what keeps the demo honest rather than clairvoyant.
    """

    event_id: str
    kind: str
    lat: float | None
    lon: float | None
    affects_cells: tuple[str, ...]
    expected_delay_minutes: Estimate
    confidence: float


@dataclass(frozen=True)
class Observation:
    """The courier's belief state at one minute.

    This is what the agent reasons over. It is assembled from the courier's
    own external tools — a weather app, a traffic service, their memory of
    which kitchens are slow — never from world ground truth.
    """

    minute: int
    at_lat: float
    at_lon: float
    at_cell: str

    temp_c: Estimate
    apparent_c: Estimate
    precip_mm: Estimate

    # Per-cell travel-time multipliers the courier can estimate for nearby
    # zones. Sparse on purpose: you do not know traffic across the whole city.
    traffic_by_cell: dict[str, Estimate]

    perceived_events: tuple[PerceivedEvent, ...]

    # Remembered kitchen speed per restaurant, learned during the shift.
    # Empty at minute zero — the courier has to earn this knowledge.
    kitchen_minutes_by_denue_id: dict[str, Estimate]

    # Where the courier believes demand is, from the app heatmap plus their
    # own sense of the day's rhythm.
    demand_by_cell: dict[str, Estimate]

    minutes_left_in_shift: int
    km_to_home: float
    fuel_minutes_remaining: float


# --------------------------------------------------------------------------
# The decision, and why — a value, not a log line
# --------------------------------------------------------------------------


class Action(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    REPOSITION = "reposition"
    REFUEL = "refuel"
    REST = "rest"
    HOLD = "hold"


@dataclass(frozen=True)
class ScoreFactor:
    """One term that moved a score, in units a human can argue with."""

    label: str
    delta_mxn: float | None = None
    delta_minutes: float | None = None
    note: str = ""


@dataclass(frozen=True)
class OfferEvaluation:
    order_id: str
    expected_net_mxn: float
    expected_minutes: float
    expected_km: float
    expected_mxn_per_hour: float
    factors: tuple[ScoreFactor, ...]
    rejected_because: str | None = None
    # Carried through from the OfferCard so the evaluation layer can measure
    # surge capture: the share of ACCEPTED orders that carried surge against
    # the share among all offers seen. A policy that selects well captures
    # more surge than it was offered; one that does not is not selecting.
    surge_flag: bool = False


@dataclass(frozen=True)
class DecisionTrace:
    """Why the policy did what it did, at the moment it did it.

    The reasoning is RETURNED, not logged. Logging it after the fact does not
    work: the state that produced it is gone by the time anyone asks. This is
    what answers the judge's question, and Judgement and Clarity are two of
    the four scored criteria.

    Contains ONLY what the agent could know at this minute. If it names an
    event the courier could not yet perceive, the trace is a lie and the whole
    demo is a lie with it.
    """

    minute: int
    considered: tuple[OfferEvaluation, ...]
    chosen_order_id: str | None
    threshold_mxn_per_hour: float
    binding_constraint: str  # "time_budget" | "acceptance_rate" | "fuel" | "home" | "none"
    summary: str  # one plain sentence, ready to read out loud


@dataclass(frozen=True)
class Decision:
    action: Action
    order_id: str | None
    target_cell: str | None
    trace: DecisionTrace


# --------------------------------------------------------------------------
# Courier state as the agent is allowed to see it
# --------------------------------------------------------------------------


class CourierActivity(StrEnum):
    IDLE = "idle"
    TO_RESTAURANT = "to_restaurant"
    WAITING_KITCHEN = "waiting_kitchen"
    TO_CUSTOMER = "to_customer"
    REPOSITIONING = "repositioning"
    REFUELLING = "refuelling"
    RESTING = "resting"


@dataclass
class CourierSnapshot:
    """The courier's own situation. Everything here is self-knowledge — a real
    courier knows their own position, earnings and fuel."""

    minute: int
    lat: float
    lon: float
    cell: str
    activity: CourierActivity

    earnings_mxn: float
    deliveries_completed: int
    km_traveled: float  # separate from minutes, always
    minutes_elapsed: float
    minutes_idle: float

    carrying_order_ids: tuple[str, ...]

    # Constraints that make the problem real. Without acceptance rate the
    # optimal policy degenerates to "reject everything until something
    # perfect appears", which would sink a courier in real life.
    offers_seen: int
    offers_accepted: int
    fuel_minutes_remaining: float
    home_lat: float
    home_lon: float
    minutes_left_in_shift: int

    @property
    def acceptance_rate(self) -> float:
        return self.offers_accepted / self.offers_seen if self.offers_seen else 1.0


# --------------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------------


@runtime_checkable
class TravelOracle(Protocol):
    """Travel between two points. Returns km and minutes SEPARATELY.

    Two implementations exist on purpose and must never be confused: the
    engine's, backed by the real network and true conditions, and the
    agent's own forward model, backed by its beliefs. Same interface, and
    that symmetry is what lets a policy plan ahead without ever seeing
    ground truth.
    """

    def travel(self, from_cell: str, to_cell: str, minute: int) -> tuple[float, float]:
        """Return (km, minutes)."""
        ...


@runtime_checkable
class PlatformPort(Protocol):
    """The app. Gives the courier the six fields and nothing more."""

    def view_at(self, minute: int, courier: CourierSnapshot) -> PlatformView: ...


@runtime_checkable
class EnrichmentPort(Protocol):
    """The courier's own external tools, with realistic error and lag."""

    def observe(self, minute: int, courier: CourierSnapshot) -> Observation: ...


@runtime_checkable
class Policy(Protocol):
    """The decision. Implementations import ONLY this module."""

    name: str

    def decide(self, view: PlatformView, observation: Observation, courier: CourierSnapshot) -> Decision: ...


@dataclass(frozen=True)
class DeliveryRecord:
    """A completed delivery, with what it ACTUALLY paid.

    Distinct from the expectation in a DecisionTrace on purpose: the gap
    between what the policy expected and what it got is the honest measure
    of how good its model is, and it is only visible if both are recorded.
    """

    order_id: str
    accepted_at_min: int
    delivered_at_min: int
    payout_mxn: float  # realised, surge locked at acceptance
    tip_mxn: float
    surge_locked: float
    km: float
    minutes: float
    kitchen_wait_minutes: float


@dataclass
class TickRecord:
    """One simulated minute, recorded. The evaluation layer and any replay
    reads these; nothing needs to re-run a shift to inspect it."""

    minute: int
    courier: CourierSnapshot
    decision: Decision | None
    offers_shown: int

    # Completed this minute, if any. Realised payout lives here, never in
    # the trace, which only ever holds what was expected beforehand.
    delivered: DeliveryRecord | None = None

    # Route the courier is on, as a reference into ShiftResult.routes plus a
    # progress fraction. The polyline is emitted ONCE and referenced by id:
    # repeating the coordinate list every minute bloats a replay file
    # enormously for no gain.
    route_id: str | None = None
    route_progress: float = 0.0

    # Events the courier COULD perceive this minute, by id only. This is what
    # lets a replay show "the agent learned about the crash at 149" honestly,
    # without the recorder ever touching ground truth. An event absent here
    # was not knowable, and must render as not knowable.
    perceived_event_ids: tuple[str, ...] = ()


@runtime_checkable
class RecorderPort(Protocol):
    def record(self, tick: TickRecord) -> None: ...


@dataclass
class ShiftResult:
    """What a whole shift produced. The comparison unit against a baseline."""

    policy_name: str
    seed: int
    shift_start_min: int
    shift_end_min: int

    earnings_mxn: float
    deliveries_completed: int
    km_traveled: float
    minutes_elapsed: float
    minutes_idle: float
    offers_seen: int
    offers_accepted: int
    ticks: list[TickRecord] = field(default_factory=list)

    # Realised deliveries, so earnings can be audited per order rather than
    # only in aggregate.
    deliveries: list[DeliveryRecord] = field(default_factory=list)

    # Real street polylines keyed by route id, emitted once and referenced
    # from ticks. Drawn straight onto the demo map.
    routes: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    # Kilometres that earned nothing: repositioning, the unpaid leg out from
    # home at the start, and the unpaid leg back at the end. The platform
    # never shows a courier this number, which is exactly why it belongs on
    # screen.
    unpaid_km: float = 0.0

    @property
    def mxn_per_hour(self) -> float:
        hours = self.minutes_elapsed / 60.0
        return self.earnings_mxn / hours if hours else 0.0

    @property
    def deliveries_per_hour(self) -> float:
        hours = self.minutes_elapsed / 60.0
        return self.deliveries_completed / hours if hours else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.offers_accepted / self.offers_seen if self.offers_seen else 1.0
