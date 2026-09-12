"""The app: `PlatformAdapter` implements `core.ports.PlatformPort`.

THE GAP IS THE PRODUCT. `src/world/` computes an exact, continuous,
instantaneous truth about the city — a precise surge multiplier per cell per
minute, a real supply/demand mechanism, exact prep times and tips per order.
No real courier app shows any of that. It shows six fields per offer and a
coarse heatmap, updated on its own schedule, filtered to whatever the app
decided was worth telling you. The entire thesis of this simulator is that an
agent which computes what the app *won't* show — the seventh number — has an
edge. This module is where that gap is drawn, deliberately and on purpose,
in five ways:

1. OFFER PROJECTION (`_build_offer`). `OrderOffer` (ground truth) carries
   `surge_at_spawn` as an exact float, `prep_minutes` and `tip_mxn` as known
   quantities. None of that reaches `OfferCard`:
     - `surge_flag` collapses the continuous multiplier to a boolean at a
       fixed, documented threshold (`SURGE_FLAG_CALIBRATION`).
     - `eta_minutes` / `distance_km` are NOT ground truth. They are the
       app's own (optimistic, noisy) estimate of the `ref_km`/`ref_minutes`
       reference leg, biased low and jittered per order
       (`ETA_BIAS_CALIBRATION`). A real app under-quotes because it is
       selling the trip, not measuring it.
     - `prep_minutes` and `tip_mxn` never appear anywhere in this module.
       Leaking either would be clairvoyance: a courier learns kitchen speed
       only by standing in the kitchen, and the tip only after delivery.
     - `restaurant_name` (the real DENUE `nom_estab`) and `restaurant_
       denue_id` (passed through verbatim from `OrderOffer`) both reach the
       card — this is what legitimately lets a policy learn "this branch is
       always slow": the name is what a human reads on screen, the id is
       the stable join key against `Observation.kitchen_minutes_by_denue_id`
       (a courier plainly sees which branch they are sent to, so neither
       leaks anything a real courier wouldn't already know).
     - `expires_in_seconds` is the accept/reject pressure the brief
       describes, sized off the app's own (already-biased) distance
       estimate (`OFFER_CALIBRATION`).

2. OFFER REACH (`view_at` + `_reach_km`). Which orders even reach a courier's screen
   depends on where that courier is standing. An order is only offered to
   couriers within a reach radius of its restaurant, and that radius shrinks
   as local competing courier supply rises (`REACH_CALIBRATION`) — a busy
   area with plenty of couriers already nearby does not need the platform to
   broadcast far to get the order picked up; a quiet area does. This is what
   makes "learn the rhythm of the day and go stand where it will pay off"
   (one of the three approaches the brief names) an actual lever: two
   policies standing in different places see genuinely different offer
   streams, deterministically, from the same scenario.

3. HEATMAP (`_build_heatmap`). Ground-truth surge is exact, instantaneous
   and fine-grained (H3 resolution 7, i.e. ~1.2 km cells, current minute).
   The in-app heatmap this module builds is none of those three things:
     - LAGGED by `HEATMAP_CALIBRATION["lag_minutes"]` minutes — the app is
       always describing a supply/demand state that has already moved on,
       exactly mirroring `src/world/surge.py`'s own point that its migration
       mechanism makes the true heatmap "lie" to anyone who reacts to it.
     - COARSE — aggregated up from resolution-7 cells to resolution-5
       parents (`HEATMAP_CALIBRATION["coarse_resolution"]`, roughly a 6x
       linear / 36x area reduction), by averaging the ground-truth surge of
       whichever child cells actually had order activity in the lag window.
     - QUANTISED into `level` buckets (`HEATMAP_CALIBRATION["level_thresholds"]`)
       — an int 1-4, never the underlying multiplier.
   The gap between this heatmap and the true, current, fine-grained surge
   field IS the product: an agent that can estimate the true field better
   than this lagged, coarse, quantised proxy has an edge over one that
   trusts the app's own map.

4. ACCEPTANCE-RATE RETALIATION (`_penalty_terms`). A real platform
   defends itself against exactly the optimisation this whole project is
   building: reject everything until a perfect offer shows up. Once a
   courier's own shown `acceptance_rate` drops below a threshold, both the
   reach radius and the offer volume shrink, and only the cheaper tail of
   otherwise-eligible orders keeps reaching them — fewer AND worse offers,
   both explicit, both in `ACCEPTANCE_RETALIATION_CALIBRATION`. Without this
   the optimal policy degenerates to `reject-until-perfect`, which would
   sink a real courier.

Determinism: the only randomness this module uses (`ETA_BIAS_CALIBRATION`'s
per-order noise) is drawn once, at construction time, from a `numpy.random.
Generator` obtained from `src.world.scenario.rng_streams(seed)` — never a
global RNG, never `np.random.seed`, never `random`. Every order's noise draw
is stored once, keyed by `order_id`, in the fixed order the order stream
already arrives in (itself deterministic given the scenario seed), so
calling `view_at` repeatedly for the same minute, or replaying the whole
scenario, is always byte-identical. Reach and heatmap degrade information
through fixed formulas and lookups, with no randomness of their own.

Nothing here imports pydantic/pandas beyond loading the DENUE restaurant
names fixture at construction time; nothing here imports `src.agent` or
`src.engine`. This module implements `core.ports.PlatformPort` and must stay
usable by any engine that only knows that protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h3
import numpy as np
import pandas as pd

from src.core.ports import CourierSnapshot, HeatCell, OfferCard, PlatformView
from src.world import geo
from src.world.scenario import rng_streams
from src.world.timeline import OrderOffer, SupplyTick

RESTAURANTS_FIXTURE_PATH: Path = geo.FIXTURES_DIR / "restaurants.parquet"

# --------------------------------------------------------------------------
# Calibration knobs. Every value below is a deliberate product/UX choice for
# how impoverished the app's view should be, NOT a measured quantity. Kept
# in explicit, labelled dicts so the whole degradation stays auditable from
# this file alone, the same convention `src/world/` uses for its own
# calibration constants.
# --------------------------------------------------------------------------

SURGE_FLAG_CALIBRATION: dict[str, float] = {
    # Binary cut point on the continuous ground-truth `surge_at_spawn`.
    # 1.2 is not an arbitrary round number: it is the exact threshold
    # `src/world/surge.py`'s own calibration notes already use to describe
    # the realism band ("share above 1.2 between 6% and 18%"), so the flag
    # asks the same question the world layer's own documentation already
    # asks, rather than inventing a second, incompatible notion of "surge".
    "surge_flag_threshold": 1.2,
}

ETA_BIAS_CALIBRATION: dict[str, float] = {
    # Real delivery apps quote an optimistic, near-best-case leg, not a
    # traffic-adjusted one — they are selling the trip. Modelled as a
    # constant fractional UNDER-estimate plus small per-order noise, applied
    # to the world's own `ref_km` / `ref_minutes` reference leg (itself
    # already a straight-line-ish reference, per `timeline.py`'s docstring
    # — the closest available ground truth to bias away from).
    "eta_optimism_fraction": 0.12,  # the app shows ~12% less time than the reference
    "eta_noise_std_fraction": 0.06,  # +/- noise on top, as a fraction of the reference
    "distance_optimism_fraction": 0.05,  # apps under-quote distance too, but less
    "distance_noise_std_fraction": 0.04,
    "min_eta_minutes": 1.0,
    "min_distance_km": 0.1,
}

OFFER_CALIBRATION: dict[str, float] = {
    # Seconds to accept or reject once a card is shown — the pressure the
    # brief describes. Sized off the app's own (already-biased) distance
    # estimate: a farther trip gets a little more thinking time, exactly
    # like the "estimated pickup distance" apps already show alongside the
    # countdown.
    "expires_in_seconds_base": 45.0,
    "expires_in_seconds_per_km": 3.0,
    "expires_in_seconds_min": 20.0,
    "expires_in_seconds_max": 90.0,
}

REACH_CALIBRATION: dict[str, float] = {
    # An order is offered to a courier only if the courier is within
    # `reach_km` of ITS RESTAURANT — competition is modelled at the
    # restaurant (how many couriers are already close to that pickup), not
    # as a radius that inflates around the courier. `baseline_reach_km` is
    # the radius at `reference_supply` local competing couriers (roughly the
    # city-wide median at an order's origin cell/minute, see the platform
    # verification script's supply-decile diagnostic).
    #
    # An undersupplied restaurant genuinely has to reach farther to find a
    # courier, so the radius DOES expand below `reference_supply` and
    # contract above it — that effect is real and kept. But local supply is
    # strongly anti-correlated with surge (an earlier revision of this
    # calibration found corr(supply, surge) ~ -0.29 in this fixture), so an
    # UNBOUNDED expansion systematically over-samples exactly the
    # high-surge orders the reach radius reaches farthest to include: a
    # first pass measured 78% of offers flagged `surge_flag=True` at a
    # position whose own local-neighbourhood ORIGIN surge share (the ceiling
    # any pure position effect could reach) was only ~27%. `max_expansion_
    # ratio` is the hard cap on that effect: reach_km is bounded to
    # [baseline_reach_km / max_expansion_ratio, baseline_reach_km *
    # max_expansion_ratio], so an undersupplied zone's radius can stretch,
    # but never past a bounded multiple of the typical radius.
    # `expansion_softening` (an exponent < 1 on the supply ratio) further
    # dampens how fast the radius reacts before the hard cap even applies.
    # `supply_floor` guards the ratio against a near-zero local supply
    # reading blowing the exponent up before softening/clamping catch it.
    #
    # All four values are tuned together (see the verification script's
    # three-position surge-ratio table) to land the ratio of "reaching-offer
    # surge share" to "local-neighbourhood origin surge share" modestly
    # above 1 (a real but bounded positional edge), never near 3 (a
    # selection artefact), while keeping total reach volume near the
    # brief's ~7.5 offers/hour target.
    # 0.70, up from 0.35, and paired with a `max_offers_per_minute` of 1
    # (below). Both are MEASUREMENT corrections against the brief's ~7.5
    # offers-per-courier-hour anchor, not a loosening; the reach MECHANISM
    # is untouched (still a radius around the ORDER'S RESTAURANT, still
    # expanding into undersupplied areas and contracting in crowded ones,
    # still bounded by `max_expansion_ratio` for the anti-correlation reason
    # documented above). Only the radius at `reference_supply` moved, and
    # the reason it had to is worth recording:
    #
    # 0.35 was tuned before the engine polled this adapter every minute. Run
    # against a courier polled every minute along a real working trajectory
    # (`scripts/run_shift.py --probe`, which exists to measure exactly this)
    # it delivered 4.5 offers/hour. Worse, that average hid a brutal
    # asymmetry: a courier RIDING passes restaurant after restaurant, while
    # a courier standing where their last customer happened to live is
    # 450 m from nothing at all. Measured, a parked courier saw as little as
    # 1.1 offers/hour on some seeds — so ANY policy that ever chose to wait
    # was starved into a spiral, and "never be idle" won by default no
    # matter how bad the work it took. That is not a courier app; a real one
    # reaches kilometres, not street corners.
    #
    # Raising reach alone would flood a well-placed courier, so the per-
    # minute cap drops to 1 at the same time — which is also simply more
    # faithful to the brief ("la app muestra un FLUJO de pedidos, y el
    # repartidor tiene SEGUNDOS para aceptar o rechazar": a flow with
    # seconds on the clock, not a menu to browse). The cap holds a
    # well-placed courier near the anchor while the wider radius lets a
    # parked one still see work. Measured over five seeds at 0.70/1: 7.9
    # offers per courier-hour against the 7.5 anchor. Sensitivity is real
    # and worth knowing: 0.70 and 0.80 both land near the anchor, 0.60 falls
    # to 5.9 and 0.90 climbs to 8.9.
    "baseline_reach_km": 0.70,
    "reference_supply": 5.0,
    "supply_floor": 1.0,
    "expansion_softening": 0.35,
    "max_expansion_ratio": 1.35,
    # At most this many offers reach a courier in a single simulated minute,
    # before the acceptance-rate penalty (below) can shrink it further.
    # ONE: the app shows a flow, one card at a time, with seconds to decide
    # — not a shortlist to rank. See `baseline_reach_km` above for why this
    # and the radius were recalibrated together.
    "max_offers_per_minute": 1,
}

ACCEPTANCE_RETALIATION_CALIBRATION: dict[str, float] = {
    # The platform's self-defence against "reject everything until a
    # perfect offer arrives". Below `penalty_start_rate` the reach radius,
    # the per-minute offer cap, and the payout ceiling of what still reaches
    # the courier all shrink linearly down to their floor at
    # `full_penalty_rate`. Ignored until the courier has seen at least
    # `min_offers_seen_for_penalty` offers, so nobody is judged on a sample
    # of one at minute zero (where `CourierSnapshot.acceptance_rate`
    # defaults to 1.0 anyway).
    # `penalty_start_rate` sits BELOW the brief's own ~1-in-3 target
    # acceptance rate (0.333), deliberately: the brief's whole ~7.5
    # offers/hour -> ~2.5 deliveries/hour calibration assumes a courier who
    # accepts about a third of what they see is operating NORMALLY, not
    # being punished. An earlier draft set this to 0.35 — just above 1/3 —
    # which meant a courier holding exactly the intended acceptance rate
    # was already under permanent mild retaliation, silently suppressing
    # the reference-position offer rate below its tuned target. 0.28 gives
    # headroom below 1/3 so normal operation (and its ordinary sampling
    # noise) draws no penalty, and only a courier meaningfully more
    # reject-happy than the brief's own baseline gets throttled.
    "min_offers_seen_for_penalty": 5.0,
    "penalty_start_rate": 0.28,
    "full_penalty_rate": 0.05,
    # Reach radius and offer cap are both scaled by this multiplier at full
    # penalty (never all the way to zero — a penalised courier still works,
    # just badly).
    "min_reach_multiplier": 0.30,
    # At full penalty, only orders at or below the `worse_offer_quantile`
    # percentile of the SCENARIO-WIDE payout distribution (precomputed once,
    # see `PlatformAdapter._order_payouts`) still get shown — the "worse"
    # half of "fewer and worse". Anchored city-wide rather than per-minute
    # because a shrunk reach radius typically leaves 0-1 candidates in any
    # one minute, too few for a per-minute quantile to mean anything.
    "worse_offer_quantile": 0.35,
}

HEATMAP_CALIBRATION: dict[str, object] = {
    # The app's heatmap describes the recent past, not now.
    "lag_minutes": 6,
    # H3 resolution the heatmap is aggregated up to. Ground truth lives at
    # resolution 7 (~1.22 km edge); resolution 5 (~8.5 km edge) is roughly a
    # 6x linear / ~36x area reduction — a genuinely coarse city-district
    # view, not a cosmetic rounding.
    "coarse_resolution": 5,
    # Three thresholds -> four buckets (1 calm ... 4 hot). Chosen around the
    # same 1.2 reference used for `surge_flag`, so "hot" on the map and
    # "surge!" on a card are describing compatible ideas at different
    # granularities.
    "level_thresholds": (1.05, 1.2, 1.6),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp_down(x: float, x_hi: float, x_lo: float, y_hi: float, y_lo: float) -> float:
    """Linearly interpolate y as x falls from `x_hi` (-> `y_hi`) to `x_lo`
    (-> `y_lo`), clamped beyond either end. Used for every "penalty ramps in
    as a rate falls" calibration curve in this module."""
    if x >= x_hi:
        return y_hi
    if x <= x_lo:
        return y_lo
    frac = (x_hi - x) / (x_hi - x_lo)
    return y_hi + frac * (y_lo - y_hi)


def _load_restaurant_names(path: Path = RESTAURANTS_FIXTURE_PATH) -> dict[str, str]:
    """Map `restaurant_denue_id` -> the real DENUE `nom_estab` trade name.

    A small fraction of DENUE records (roughly 1 in 1600 in this fixture)
    carry an empty `nom_estab` — real census gaps, not a bug. For those, the
    app falls back to the establishment's real activity classification
    (`nombre_act`) plus a short id suffix, e.g. "Restaurante de autoservicio
    (#0657)" — still a real, stable, distinguishing label a courier could
    plausibly see on a receipt or storefront, never a fabricated proper
    name.
    """
    df = pd.read_parquet(path, columns=["denue_id", "nom_estab", "nombre_act"])
    names: dict[str, str] = {}
    for row in df.itertuples(index=False):
        denue_id = str(row.denue_id)
        nom_estab = str(row.nom_estab).strip()
        if nom_estab:
            names[denue_id] = nom_estab
        else:
            activity = str(row.nombre_act).strip() or "Restaurant"
            names[denue_id] = f"{activity} (#{denue_id[-4:]})"
    return names


class PlatformAdapter:
    """Implements `core.ports.PlatformPort`: the app, and nothing more.

    Constructed once per scenario from ground-truth objects the engine
    already has to hand — the full `order_stream` and the `supply_timeline`
    that `src/world/surge.py` produced alongside it — plus the scenario
    seed, used only to draw this layer's own deterministic ETA/distance
    noise from the shared `rng_streams` machinery (see module docstring).
    Every `view_at` call afterwards is a pure, deterministic read.
    """

    def __init__(
        self,
        order_stream: Sequence[OrderOffer],
        supply_timeline: Sequence[SupplyTick],
        scenario_seed: int,
        restaurants_path: Path = RESTAURANTS_FIXTURE_PATH,
        surge_flag_calibration: Mapping[str, float] | None = None,
        eta_bias_calibration: Mapping[str, float] | None = None,
        offer_calibration: Mapping[str, float] | None = None,
        reach_calibration: Mapping[str, float] | None = None,
        acceptance_retaliation_calibration: Mapping[str, float] | None = None,
        heatmap_calibration: Mapping[str, object] | None = None,
    ) -> None:
        self._surge_flag_cal = {**SURGE_FLAG_CALIBRATION, **(surge_flag_calibration or {})}
        self._eta_cal = {**ETA_BIAS_CALIBRATION, **(eta_bias_calibration or {})}
        self._offer_cal = {**OFFER_CALIBRATION, **(offer_calibration or {})}
        self._reach_cal = {**REACH_CALIBRATION, **(reach_calibration or {})}
        self._retaliation_cal = {
            **ACCEPTANCE_RETALIATION_CALIBRATION,
            **(acceptance_retaliation_calibration or {}),
        }
        self._heatmap_cal = {**HEATMAP_CALIBRATION, **(heatmap_calibration or {})}

        self._restaurant_names = _load_restaurant_names(restaurants_path)

        # Orders indexed by spawn minute, in the order the (deterministic)
        # world stream produced them — an order is only ever offered during
        # the one minute it spawns, mirroring a real push notification.
        self._orders_by_minute: dict[int, list[OrderOffer]] = {}
        for order in order_stream:
            self._orders_by_minute.setdefault(order.spawn_min, []).append(order)

        # Local competing supply per (cell, minute), read straight from the
        # ground-truth supply field the same scenario's surge mechanism
        # produced — this is what makes reach a real function of "how
        # crowded is it here right now", not a demand-side proxy.
        self._supply_by_minute: dict[int, dict[str, float]] = {
            tick.minute: tick.couriers_per_cell for tick in supply_timeline
        }

        # Ground truth for the heatmap: surge_at_spawn readings grouped by
        # (origin_cell, minute), then indexed by minute for fast windowed
        # lookup. Every order sharing a (cell, minute) was stamped from the
        # exact same ground-truth surge reading (see `demand.py`), so the
        # first value seen per key is already exact — no averaging needed,
        # and no reconstruction of the surge field itself. Cells/minutes
        # with zero orders simply have no reading, which the heatmap
        # builder treats as calm (level 1): no demand signal is itself
        # information a real app would show as quiet.
        seen_cell_minute: set[tuple[str, int]] = set()
        self._readings_by_minute: dict[int, list[tuple[str, float]]] = {}
        for order in order_stream:
            key = (order.origin_cell, order.spawn_min)
            if key in seen_cell_minute:
                continue
            seen_cell_minute.add(key)
            self._readings_by_minute.setdefault(order.spawn_min, []).append(
                (order.origin_cell, order.surge_at_spawn)
            )

        # Deterministic per-order ETA/distance noise, drawn ONCE at
        # construction from the scenario's own named RNG machinery — never
        # a global RNG. `src/world/scenario.py`'s `RNG_STREAM_NAMES` has no
        # dedicated "platform" stream (this package did not exist when that
        # list was frozen), so this layer reuses "observation_noise": the
        # same category of concern (a courier-facing noisy estimate, not a
        # world mechanism), and — because `rng_streams(seed)` spawns each
        # named generator fresh from the seed on every call — this layer's
        # own copy of that generator is a fully independent stream in
        # practice: consuming from it here never perturbs whatever
        # `src/enrichment/` does with its own separate copy of the same
        # name.
        noise_rng = rng_streams(scenario_seed)["observation_noise"]
        # Drawn in a fixed order (ascending spawn minute, then order_id,
        # both already deterministic) so the mapping from seed to noise
        # values never depends on incidental list ordering upstream.
        ordered_ids = sorted((o.order_id for o in order_stream))
        eta_noise = noise_rng.normal(0.0, 1.0, size=len(ordered_ids))
        distance_noise = noise_rng.normal(0.0, 1.0, size=len(ordered_ids))
        self._eta_noise_by_order: dict[str, float] = dict(zip(ordered_ids, eta_noise.tolist()))
        self._distance_noise_by_order: dict[str, float] = dict(zip(ordered_ids, distance_noise.tolist()))

        # City-wide payout distribution, precomputed once, for the
        # acceptance-rate retaliation's "worse" filter. A per-MINUTE
        # quantile (computed only over that minute's handful of reachable
        # candidates) is nearly always a no-op at this layer's reach scale
        # (a shrunk reach radius typically leaves 0-1 candidates in a given
        # minute, and a quantile over a sample of size 1 always keeps it) —
        # so the filter is anchored against the whole scenario's payout
        # distribution instead, giving a stable, meaningful cutoff
        # regardless of how sparse any one minute's candidate pool is.
        self._order_payouts: np.ndarray = np.array([_gross_offer_payout(o) for o in order_stream])

    # ----------------------------------------------------------------
    # PlatformPort
    # ----------------------------------------------------------------

    def view_at(self, minute: int, courier: CourierSnapshot) -> PlatformView:
        candidates = self._orders_by_minute.get(minute, [])
        reach_ceiling_km, offer_cap_mult, quantile_keep = self._penalty_terms(courier)

        reachable: list[tuple[float, OrderOffer]] = []
        for order in candidates:
            local_supply = self._supply_by_minute.get(order.spawn_min, {}).get(order.origin_cell, 0.0)
            # The penalty is applied as an absolute CEILING, not a
            # multiplicative scale of `_reach_km`'s own (surge-correlated)
            # radius: scaling every order's radius by the same factor would
            # preserve the gap between an expanded (low-supply, high-surge)
            # radius and a contracted (high-supply, low-surge) one, so a
            # uniform shrink would still let through disproportionately more
            # of the already-larger high-surge radii — reintroducing the
            # exact over-sampling bug the reach fix above exists to remove,
            # but now triggered by retaliation instead of ordinary reach. A
            # flat ceiling instead clips every order toward the SAME small
            # radius once the penalty is severe enough, which is what
            # actually makes the surviving set skew cheap once combined
            # with the quantile filter below.
            reach_km = min(self._reach_km(local_supply), reach_ceiling_km)
            distance_km = geo.great_circle_km(courier.lat, courier.lon, order.origin_lat, order.origin_lon)
            if distance_km <= reach_km:
                reachable.append((distance_km, order))

        if reachable and quantile_keep < 1.0:
            # Anchored against the scenario-wide payout distribution
            # (precomputed in __init__), not the handful of candidates
            # reachable this specific minute — see `_order_payouts` for why.
            threshold = float(np.quantile(self._order_payouts, quantile_keep))
            reachable = [
                (dist, order) for dist, order in reachable if _gross_offer_payout(order) <= threshold
            ]

        reachable.sort(key=lambda item: (item[0], item[1].order_id))
        offer_cap = max(1, round(self._reach_cal["max_offers_per_minute"] * offer_cap_mult))
        selected = reachable[:offer_cap]

        offers = tuple(self._build_offer(order) for _, order in selected)
        heatmap = self._build_heatmap(minute)

        return PlatformView(
            minute=minute,
            offers=offers,
            heatmap=heatmap,
            acceptance_rate=courier.acceptance_rate,
            deliveries_completed=courier.deliveries_completed,
            # The app rounds earnings to whole pesos for display — a UI
            # simplification, not a different number from the courier's own
            # ground truth `earnings_mxn`.
            earnings_shown_mxn=round(courier.earnings_mxn, 2),
        )

    # ----------------------------------------------------------------
    # Offer projection
    # ----------------------------------------------------------------

    def _build_offer(self, order: OrderOffer) -> OfferCard:
        eta_minutes = self._biased_eta_minutes(order)
        distance_km = self._biased_distance_km(order)
        payout_mxn = round(_gross_offer_payout(order), 2)
        surge_flag = order.surge_at_spawn > self._surge_flag_cal["surge_flag_threshold"]
        restaurant_name = self._restaurant_names.get(order.restaurant_denue_id, "Restaurant")
        expires_in_seconds = self._expires_in_seconds(distance_km)

        return OfferCard(
            order_id=order.order_id,
            pickup_lat=order.origin_lat,
            pickup_lon=order.origin_lon,
            dropoff_lat=order.dest_lat,
            dropoff_lon=order.dest_lon,
            payout_mxn=payout_mxn,
            eta_minutes=eta_minutes,
            distance_km=distance_km,
            surge_flag=surge_flag,
            restaurant_name=restaurant_name,
            expires_in_seconds=expires_in_seconds,
            restaurant_denue_id=order.restaurant_denue_id,
        )

    def _biased_eta_minutes(self, order: OrderOffer) -> float:
        cal = self._eta_cal
        noise = self._eta_noise_by_order.get(order.order_id, 0.0)
        biased = order.ref_minutes * (1.0 - cal["eta_optimism_fraction"] + cal["eta_noise_std_fraction"] * noise)
        return round(max(biased, cal["min_eta_minutes"]), 1)

    def _biased_distance_km(self, order: OrderOffer) -> float:
        cal = self._eta_cal
        noise = self._distance_noise_by_order.get(order.order_id, 0.0)
        biased = order.ref_km * (1.0 - cal["distance_optimism_fraction"] + cal["distance_noise_std_fraction"] * noise)
        return round(max(biased, cal["min_distance_km"]), 2)

    def _expires_in_seconds(self, distance_km: float) -> int:
        cal = self._offer_cal
        seconds = cal["expires_in_seconds_base"] + cal["expires_in_seconds_per_km"] * distance_km
        return int(round(_clamp(seconds, cal["expires_in_seconds_min"], cal["expires_in_seconds_max"])))

    # ----------------------------------------------------------------
    # Offer reach
    # ----------------------------------------------------------------

    def _reach_km(self, local_supply: float) -> float:
        """Reach radius around AN ORDER'S RESTAURANT, bounded to a fixed
        multiple of `baseline_reach_km` either way — see `REACH_CALIBRATION`
        for why the bound exists (an unbounded version of this formula
        systematically over-sampled high-surge orders, because local supply
        and surge are anti-correlated)."""
        cal = self._reach_cal
        baseline = cal["baseline_reach_km"]
        reference = cal["reference_supply"]
        floored_supply = max(local_supply, cal["supply_floor"])
        ratio = (reference / floored_supply) ** cal["expansion_softening"]
        cap = cal["max_expansion_ratio"]
        factor = _clamp(ratio, 1.0 / cap, cap)
        return baseline * factor

    def _penalty_terms(self, courier: CourierSnapshot) -> tuple[float, float, float]:
        """Return (reach_ceiling_km, offer_cap_multiplier, payout_quantile_keep)
        from the acceptance-rate retaliation curve. `reach_ceiling_km` is an
        absolute cap applied on top of (never multiplying) `_reach_km`'s own
        result — see `view_at` for why an absolute ceiling, not a scale
        factor, is required here. At no penalty the ceiling is set above
        the largest `_reach_km` could ever return, so `min()` never binds
        and behaviour is unchanged; at full penalty it is small enough to
        flatten every order — regardless of its own local-supply-driven
        radius — down to the same tight radius, which is what stops the
        penalty from re-introducing a surge-correlated selection bias."""
        cal = self._retaliation_cal
        reach_cal = self._reach_cal
        no_penalty_ceiling_km = reach_cal["baseline_reach_km"] * reach_cal["max_expansion_ratio"]
        if courier.offers_seen < cal["min_offers_seen_for_penalty"]:
            return no_penalty_ceiling_km, 1.0, 1.0
        rate = courier.acceptance_rate
        start, floor = cal["penalty_start_rate"], cal["full_penalty_rate"]
        ceiling_mult = _lerp_down(rate, start, floor, reach_cal["max_expansion_ratio"], cal["min_reach_multiplier"])
        reach_ceiling_km = reach_cal["baseline_reach_km"] * ceiling_mult
        offer_cap_mult = _lerp_down(rate, start, floor, 1.0, cal["min_reach_multiplier"])
        quantile_keep = _lerp_down(rate, start, floor, 1.0, cal["worse_offer_quantile"])
        return reach_ceiling_km, offer_cap_mult, quantile_keep

    # ----------------------------------------------------------------
    # Heatmap
    # ----------------------------------------------------------------

    def _build_heatmap(self, minute: int) -> tuple[HeatCell, ...]:
        lag = int(self._heatmap_cal["lag_minutes"])
        coarse_res = int(self._heatmap_cal["coarse_resolution"])
        thresholds = self._heatmap_cal["level_thresholds"]
        lagged_minute = minute - lag

        # Aggregate every ground-truth (cell, minute) reading that falls
        # inside the lag window [lagged_minute - lag, lagged_minute] into
        # its coarse parent cell — the app's heatmap describes a smeared
        # recent past, not one exact instant.
        readings_by_coarse: dict[str, list[float]] = {}
        for m in range(lagged_minute - lag, lagged_minute + 1):
            for cell, surge in self._readings_by_minute.get(m, []):
                coarse_cell = h3.cell_to_parent(cell, coarse_res)
                readings_by_coarse.setdefault(coarse_cell, []).append(surge)

        cells = []
        for coarse_cell, readings in readings_by_coarse.items():
            mean_surge = sum(readings) / len(readings)
            level = 1
            for threshold in thresholds:
                if mean_surge > threshold:
                    level += 1
            lat, lon = geo.cell_centroid(coarse_cell)
            cells.append(HeatCell(cell=coarse_cell, lat=lat, lon=lon, level=level))
        return tuple(cells)


def _gross_offer_payout(order: OrderOffer) -> float:
    """The payout an offer card shows: the real fare including the surge
    multiplier already applied (a real app DOES show the surge-boosted
    price — that is what surge means to a courier), but never the tip,
    which is unknown to anyone until after delivery."""
    return order.gross_payout_mxn * order.surge_at_spawn
