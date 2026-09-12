"""Ground truth: endogenous state schema (courier, orders, trip legs).

This module is ground truth. It defines the shared vocabulary the
simulation engine (a later slice) mutates tick by tick, and that any
evaluation tooling reads back. `src/agent/` must never import this module
directly for decision-making — a policy only ever sees a restricted,
possibly-noisy view surfaced through `src/platform/`/`src/enrichment/`
(later slices).

Hard requirement: km and minutes are always separate first-class fields.
Never collapse them into a single "distance" or "cost" number, here or
anywhere else in the simulator.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TripPurpose(StrEnum):
    TO_RESTAURANT = "to_restaurant"
    TO_CUSTOMER = "to_customer"
    REPOSITION = "reposition"
    IDLE = "idle"


class TripLeg(BaseModel):
    """One movement segment. km and minutes are independent measurements of
    the same leg, not derived from one another (real routes have variable
    speed profiles)."""

    from_cell: str
    to_cell: str
    km: float = Field(ge=0)
    minutes: float = Field(ge=0)
    purpose: TripPurpose
    start_min: float | None = None
    end_min: float | None = None


class RestaurantRef(BaseModel):
    """Pointer to a DENUE-backed restaurant (an order's origin)."""

    denue_id: str
    cell: str
    lat: float
    lon: float


class DestinationRef(BaseModel):
    """An order's delivery destination."""

    cell: str
    lat: float
    lon: float


class OrderState(BaseModel):
    order_id: str
    restaurant: RestaurantRef
    destination: DestinationRef

    # Payout components, kept separate so surge/tip logic is auditable.
    base_payout_mxn: float = Field(ge=0)
    surge_multiplier: float = Field(default=1.0, ge=0)
    surge_locked: bool = False
    tip_mxn: float = Field(default=0.0, ge=0)

    accepted_at_min: float | None = None
    picked_up_at_min: float | None = None
    delivered_at_min: float | None = None

    # Actuals: filled in once the order is delivered. km and minutes stay
    # separate outputs, same as everywhere else in the simulator.
    km_actual: float | None = None
    minutes_actual: float | None = None

    @property
    def total_payout_mxn(self) -> float:
        return self.base_payout_mxn * self.surge_multiplier + self.tip_mxn


class CourierState(BaseModel):
    courier_id: str

    # Position
    cell: str
    lat: float
    lon: float

    # Current movement, if any
    current_leg: TripLeg | None = None
    leg_progress_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    # Cumulative counters. km_traveled and minutes_elapsed are independent
    # first-class outputs (never collapsed into one "distance" figure).
    km_traveled: float = 0.0
    minutes_elapsed: float = 0.0
    minutes_idle: float = 0.0
    earnings_mxn: float = 0.0

    active_orders: list[OrderState] = Field(default_factory=list)
    completed_orders: list[OrderState] = Field(default_factory=list)
