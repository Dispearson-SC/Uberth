"""Trivial in-package STUB adapters: `StubPlatform` and `StubEnrichment`.

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
  - `StubEnrichment` returns every estimate at confidence 1.0, age 0 — it
    is ground truth wrapped in `Estimate`, not a noisy belief. It reports no
    perceived events and no learned kitchen memory. The real enrichment
    layer owns adding lag, noise and staleness; this stub only exists so
    the engine has something to call.
"""

from __future__ import annotations

from src.core.ports import (
    CourierActivity,
    CourierSnapshot,
    Estimate,
    Observation,
    OfferCard,
    PlatformView,
)
from src.engine.calibration import STUB_PLATFORM_CALIBRATION, acceptance_rate_offer_multiplier
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
