"""Ground truth: typed schema for every exogenous timeline in a Scenario.

This module is ground truth. It is the coordination contract between the
exogenous producers (`weather.py`, `traffic.py`, `events.py`, `demand.py`)
and the simulation engine that consumes them. Each producer is written
independently; this file is what keeps them interoperable.

Everything declared here is *exogenous*: precomputed once from the scenario
seed and frozen before any policy runs. A courier cannot influence any of
it, which is exactly what makes replay trivial and A/B comparison hermetic.

Hard requirement, repeated because it is load-bearing: km and minutes are
always separate first-class fields. Never collapse them into one number.

`src/agent/` must never import this module. A policy sees only the reduced
`src/platform/` view plus the noisy `src/enrichment/` estimates.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------


class WeatherTick(BaseModel):
    """City-wide weather at one simulated minute.

    Sourced from real Open-Meteo archive data for Monterrey (see
    `fixtures/raw/`), interpolated from hourly observations to per-minute.
    This is measured data, not a calibration knob.
    """

    minute: int
    temp_c: float
    apparent_c: float
    precip_mm: float = Field(ge=0)
    wind_kmh: float = Field(ge=0)
    humidity_pct: float = Field(ge=0, le=100)
    weather_code: int

    @property
    def is_raining(self) -> bool:
        return self.precip_mm > 0.0

    @property
    def is_extreme_heat(self) -> bool:
        """Monterrey's operative threshold: apparent temperature at or above
        40 C is the condition the challenge brief calls out explicitly."""
        return self.apparent_c >= 40.0


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------


class TrafficTick(BaseModel):
    """Congestion field at one simulated minute.

    Multipliers scale free-flow travel time: 1.0 is free flow, 2.0 means a
    leg takes twice as long. `city_multiplier` is the baseline for the whole
    operating area; `cell_multipliers` overrides specific H3 cells that
    deviate from it (downtown and San Pedro do not behave like the
    periphery). A cell absent from the override map uses `city_multiplier`.

    Provenance, stated precisely because it is not uniform across the three
    dimensions of this field:

    - SPATIAL (which cells are slow) is derived from Monterrey's own OSM
      drive graph and DENUE commercial density. Real, local data.
    - TEMPORAL (which hour is slow) is the normalised daily shape of
      TomTom's real measured Mexico City hourly series. Monterrey is NOT in
      TomTom's free tier, and no free per-hour or per-edge traffic feed for
      Monterrey exists. Peak TIMING is what transfers between two Mexican
      metros sharing work and meal schedules; magnitude does not.
    - MAGNITUDE is a single explicit calibration constant, pinned by the
      simulator's plausibility assertions (earnings and deliveries per hour
      for a real Monterrey courier), not by measured traffic.

    Never label anything derived from this as "Monterrey traffic data". If a
    judge asks where the numbers came from, the honest answer must be the
    one written here.
    """

    minute: int
    city_multiplier: float = Field(gt=0)
    cell_multipliers: dict[str, float] = Field(default_factory=dict)

    def multiplier_for(self, cell: str) -> float:
        return self.cell_multipliers.get(cell, self.city_multiplier)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


class EventType(StrEnum):
    RAIN_ONSET = "rain_onset"
    EXTREME_HEAT = "extreme_heat"
    CRASH = "crash"
    STREET_CLOSURE = "street_closure"
    CHECKPOINT = "checkpoint"  # reten / alcoholimetro
    SURGE_WINDOW = "surge_window"
    MASS_EVENT = "mass_event"  # stadium, concert
    KITCHEN_BACKLOG = "kitchen_backlog"


class Event(BaseModel):
    """A disruptive occurrence on the exogenous timeline.

    Geometry: exactly one of `edges`, `cells`, or `point_radius_km` (paired
    with `point_lat`/`point_lon`) locates the event. Edge-scoped events are
    the ones that can force a travel-matrix recomputation.

    Effects are multiplicative over the baseline unless stated otherwise.
    All effect and detectability values are calibration knobs, not measured
    data.
    """

    event_id: str
    type: EventType
    start_min: int
    duration_min: int = Field(gt=0)

    # --- Where (exactly one locator) ---
    edges: list[tuple[int, int]] | None = None
    cells: list[str] | None = None
    point_lat: float | None = None
    point_lon: float | None = None
    point_radius_km: float | None = None

    # --- What it does ---
    speed_mult: float = Field(default=1.0, gt=0)
    close_edges: bool = False
    demand_mult: float = Field(default=1.0, ge=0)
    payout_mult: float = Field(default=1.0, ge=0)
    courier_supply_mult: float = Field(default=1.0, ge=0)
    fixed_delay_min: float = Field(default=0.0, ge=0)
    risk_delta: float = 0.0

    # --- How the courier finds out ---
    # Negative = announced in advance (forecast, scheduled roadworks).
    # Positive = the courier learns about it late, as in real life.
    detect_offset_min: int = 0
    # The courier only perceives the event within this radius. A very small
    # radius models "you find out when you are already on top of it", which
    # is exactly how a checkpoint works.
    detect_radius_km: float = Field(default=2.0, ge=0)
    # How reliable the signal is once received, in [0, 1].
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def end_min(self) -> int:
        return self.start_min + self.duration_min

    def is_active(self, minute: int) -> bool:
        return self.start_min <= minute < self.end_min

    @property
    def detectable_from_min(self) -> int:
        """First minute at which the courier could possibly perceive this."""
        return self.start_min + self.detect_offset_min


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------


class OrderOffer(BaseModel):
    """One order generated by the city, before any courier sees it.

    This is the ground-truth record. What actually reaches a courier's
    screen is a reduced projection built by `src/platform/` — and which
    orders reach a given courier depends on where that courier is standing,
    which is what makes repositioning a real strategy.

    Payout is stored decomposed so the fare model stays auditable: the exact
    Uber Eats / DiDi Food coefficients are not public, so `base_mxn`,
    `per_km_mxn` and `per_min_mxn` are calibration knobs tuned until whole
    shifts land in the earnings range real Monterrey couriers report. Never
    present them as platform-published figures.
    """

    order_id: str
    spawn_min: int

    # Origin: a real DENUE-backed restaurant.
    restaurant_denue_id: str
    origin_cell: str
    origin_lat: float
    origin_lon: float

    # Destination: sampled from real INEGI AGEB population.
    dest_cell: str
    dest_lat: float
    dest_lon: float

    # Straight-line reference values. The realised km/minutes depend on the
    # route actually taken and on live conditions, and are recorded on
    # OrderState. Kept separate, as always.
    ref_km: float = Field(ge=0)
    ref_minutes: float = Field(ge=0)

    # Fare components (pre-surge).
    base_mxn: float = Field(ge=0)
    per_km_mxn: float = Field(ge=0)
    per_min_mxn: float = Field(ge=0)

    # Ground-truth surge at spawn time. The app shows a coarse, lagged,
    # quantised version of this — the gap between the two is the product.
    surge_at_spawn: float = Field(default=1.0, gt=0)

    # Kitchen prep time drawn from the restaurant's distribution. The
    # courier does not know this until they arrive.
    prep_minutes: float = Field(default=0.0, ge=0)

    # Tip realised on delivery. Correlated with order value, rain and
    # punctuality; unknown to the courier at decision time.
    tip_mxn: float = Field(default=0.0, ge=0)

    @property
    def gross_payout_mxn(self) -> float:
        """Fare before surge and before tip."""
        return self.base_mxn + self.per_km_mxn * self.ref_km + self.per_min_mxn * self.ref_minutes


# --------------------------------------------------------------------------
# Supply field (drives surge; see events.py / demand.py)
# --------------------------------------------------------------------------


class SupplyTick(BaseModel):
    """Competing-courier density per H3 cell at one simulated minute.

    This is the piece that makes surge a mechanism instead of a random
    number: surge is demand over supply, and supply migrates toward high
    surge with a lag. That lag is why the in-app heatmap "lies" — couriers
    arrive after the imbalance they were chasing has already closed. The
    oscillation is emergent, not scripted.
    """

    minute: int
    couriers_per_cell: dict[str, float] = Field(default_factory=dict)

    def supply_in(self, cell: str) -> float:
        return self.couriers_per_cell.get(cell, 0.0)
