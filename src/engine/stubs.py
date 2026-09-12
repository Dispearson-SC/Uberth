"""Trivial in-package STUB adapters: `StubPlatform`, `StubRawSource`,
`StubEnrichment`.

These exist for exactly one reason: `src/platform/` and `src/enrichment/`
are separate slices written in parallel by other agents, and the engine
must compile and run standalone before either lands. Both stubs implement
their port minimally and honestly — no fabricated ground truth, no noise
model dressed up as realism — and are clearly named as stubs so nobody
mistakes them for the real adapters.

STUB SIMPLIFICATIONS, stated once here rather than scattered in comments:
  - `StubPlatform` only ever offers an order during the exact minute it
    spawns, and only to an IDLE courier within a fixed radius. It never
    re-offers, never explicitly expires an offer mid-tick (the one-tick
    window IS the expiry), and applies the acceptance-rate throttle
    (`calibration.acceptance_rate_offer_multiplier`) as a straight
    offer-count cap — never a fabricated price. A quality-based throttle
    ("worse offers", not just fewer) is the real platform layer's job.
  - `StubRawSource` answers every query at confidence 1.0, age 0 — it is
    ground truth wrapped in `Estimate`, not a noisy belief. It knows no
    congestion, no POI density, no disruptions, and it learns nothing: its
    history is write-only and `refit` fits nothing. The real
    `src.enrichment.RawSourceAdapter` owns lag, noise, staleness and the
    learned tables; this stub only exists so the engine has something to
    pass through.
  - `StubEnrichment` implements the DEPRECATED push-based `EnrichmentPort`
    and is kept only so nothing that still wires it breaks. The engine no
    longer calls it: see `engine.py`'s docstring for why pushing a finished
    `Observation` at a policy cannot be exported to another city.
"""

from __future__ import annotations

from src.core.ports import (
    CourierActivity,
    CourierSnapshot,
    Estimate,
    Observation,
    OfferCard,
    PerceivedEvent,
    PlatformView,
)
from src.engine.calibration import (
    STUB_PLATFORM_CALIBRATION,
    STUB_RAW_SOURCE_CALIBRATION,
    acceptance_rate_offer_multiplier,
)
from src.engine.travel import NetworkTravelOracle
from src.world import geo
from src.world.timeline import OrderOffer, WeatherTick


class StubPlatform:
    """Minimal `PlatformPort` implementation. See module docstring."""

    def __init__(self, orders_by_minute: dict[int, list[OrderOffer]]):
        self._orders_by_minute = orders_by_minute

    def view_at(self, minute: int, courier: CourierSnapshot) -> PlatformView:
        cal = STUB_PLATFORM_CALIBRATION
        offers: tuple[OfferCard, ...] = ()
        if courier.activity == CourierActivity.IDLE:
            candidates = self._orders_by_minute.get(minute, [])
            nearby = [
                order
                for order in candidates
                if geo.great_circle_km(courier.lat, courier.lon, order.origin_lat, order.origin_lon)
                <= cal["offer_radius_km"]
            ]
            offer_mult = acceptance_rate_offer_multiplier(courier.acceptance_rate)
            max_offers = max(0, round(cal["max_offers_per_tick"] * offer_mult))
            offers = tuple(
                OfferCard(
                    order_id=order.order_id,
                    pickup_lat=order.origin_lat,
                    pickup_lon=order.origin_lon,
                    dropoff_lat=order.dest_lat,
                    dropoff_lon=order.dest_lon,
                    payout_mxn=round(order.gross_payout_mxn * order.surge_at_spawn, 2),
                    eta_minutes=order.ref_minutes,
                    distance_km=order.ref_km,
                    surge_flag=order.surge_at_spawn > 1.05,
                    restaurant_name=f"Restaurant {order.restaurant_denue_id}",
                    expires_in_seconds=int(cal["expires_in_seconds"]),
                )
                for order in nearby[:max_offers]
            )
        return PlatformView(
            minute=minute,
            offers=offers,
            heatmap=(),  # stub: no heatmap; the real platform layer owns this
            acceptance_rate=courier.acceptance_rate,
            deliveries_completed=courier.deliveries_completed,
            earnings_shown_mxn=courier.earnings_mxn,
        )


class StubEnrichment:
    """Minimal `EnrichmentPort` implementation. See module docstring."""

    def __init__(
        self,
        travel: NetworkTravelOracle,
        weather_by_minute: dict[int, WeatherTick],
        home_cell: str,
    ):
        self._travel = travel
        self._weather_by_minute = weather_by_minute
        self._home_cell = home_cell

    def observe(self, minute: int, courier: CourierSnapshot) -> Observation:
        weather = self._weather_by_minute.get(minute)
        temp_c = weather.temp_c if weather else 25.0
        apparent_c = weather.apparent_c if weather else 25.0
        precip_mm = weather.precip_mm if weather else 0.0
        km_to_home, _minutes_to_home = self._travel.travel(courier.cell, self._home_cell, minute)
        return Observation(
            minute=minute,
            at_lat=courier.lat,
            at_lon=courier.lon,
            at_cell=courier.cell,
            temp_c=Estimate(value=temp_c, confidence=1.0, age_minutes=0.0),
            apparent_c=Estimate(value=apparent_c, confidence=1.0, age_minutes=0.0),
            precip_mm=Estimate(value=precip_mm, confidence=1.0, age_minutes=0.0),
            traffic_by_cell={},  # stub: no per-cell traffic sense; enrichment layer owns this
            perceived_events=(),  # stub: no event perception; enrichment layer owns this
            kitchen_minutes_by_denue_id={},  # stub: no learned kitchen memory yet
            demand_by_cell={},  # stub: no demand sense beyond the platform heatmap
            minutes_left_in_shift=courier.minutes_left_in_shift,
            km_to_home=km_to_home,
            fuel_minutes_remaining=courier.fuel_minutes_remaining,
        )


class StubRawSource:
    """Minimal `RawSourcePort` implementation. See module docstring.

    Deliberately as ignorant as it is honest: it answers the questions the
    port defines, at full confidence, from the weather timeline and a
    straight line. Nothing here is a noise model dressed up as realism, and
    nothing here learns — which means a shift run against this stub shows
    the engine working, never the agent calibrating.
    """

    def __init__(
        self,
        travel: NetworkTravelOracle,
        weather_by_minute: dict[int, WeatherTick],
        home_cell: str,
    ):
        self._travel = travel
        self._weather_by_minute = weather_by_minute
        self._home_cell = home_cell
        self._kitchen: dict[str, list[float]] = {}
        self._trips: list[tuple[str, str, int, float, float, float]] = []

    # -- weather ---------------------------------------------------------

    def weather_at(self, lat: float, lon: float, minute: int) -> dict[str, Estimate]:
        weather = self._weather_by_minute.get(minute)
        return {
            "temp_c": Estimate(weather.temp_c if weather else 25.0, 1.0, 0.0),
            "apparent_c": Estimate(weather.apparent_c if weather else 25.0, 1.0, 0.0),
            "precip_mm": Estimate(weather.precip_mm if weather else 0.0, 1.0, 0.0),
        }

    # -- road network ----------------------------------------------------

    def travel_estimate(
        self, from_lat: float, from_lon: float, to_lat: float, to_lon: float, minute: int
    ) -> tuple[Estimate, Estimate]:
        """Straight line at a fixed speed. No skeleton, no learned
        correction — the real adapter owns both."""
        km = geo.great_circle_km(from_lat, from_lon, to_lat, to_lon) * STUB_RAW_SOURCE_CALIBRATION[
            "detour_factor"
        ]
        minutes = km / STUB_RAW_SOURCE_CALIBRATION["speed_kmh"] * 60.0
        return Estimate(km, 1.0, 0.0), Estimate(minutes, 1.0, 0.0)

    def congestion_near(
        self, lat: float, lon: float, radius_km: float, minute: int
    ) -> dict[str, Estimate]:
        return {}  # stub: no traffic sense; the real adapter owns this

    # -- POIs ------------------------------------------------------------

    def poi_density_near(self, lat: float, lon: float, radius_km: float) -> dict[str, Estimate]:
        return {}  # stub: no POI pull; the real adapter owns this

    # -- disruptions -----------------------------------------------------

    def perceived_disruptions(
        self, lat: float, lon: float, minute: int
    ) -> tuple[PerceivedEvent, ...]:
        return ()  # stub: no event perception; the real adapter owns this

    # -- geography -------------------------------------------------------

    def cell_coords(self, cells: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        return {cell: geo.cell_centroid(cell) for cell in cells if cell}

    # -- history: written, never fitted ----------------------------------

    def recall_kitchen(self, denue_id: str) -> Estimate | None:
        observed = self._kitchen.get(denue_id)
        if not observed:
            return None
        return Estimate(sum(observed) / len(observed), 1.0, 0.0)

    def record_kitchen(self, denue_id: str, observed_minutes: float, minute: int) -> None:
        self._kitchen.setdefault(denue_id, []).append(observed_minutes)

    def recall_eta_bias(self, cell: str) -> Estimate | None:
        return None  # stub: nothing is ever fitted

    def record_trip(
        self,
        from_cell: str,
        to_cell: str,
        minute: int,
        promised_minutes: float,
        actual_minutes: float,
        actual_km: float,
    ) -> None:
        self._trips.append(
            (from_cell, to_cell, minute, promised_minutes, actual_minutes, actual_km)
        )

    def refit(self) -> dict[str, int]:
        return {"trips": len(self._trips), "kitchens": len(self._kitchen)}
