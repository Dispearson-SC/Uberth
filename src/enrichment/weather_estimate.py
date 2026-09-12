"""The courier's weather app: close to real, never exact.

Reads the real per-minute `WeatherTick` the scenario was built with (see
`src.world.weather`) and returns noisy `Estimate`s — a weather app IS
reliable, so temperature/apparent-temperature error is small and confidence
is high; precipitation is genuinely harder to nowcast, so it is noisier and
less confident. Every draw comes from the caller-supplied `observation_noise`
RNG stream, never a global RNG.
"""

from __future__ import annotations

import numpy as np

from src.core.ports import Estimate
from src.enrichment.calibration import WEATHER_NOISE_CALIBRATION
from src.world.timeline import WeatherTick


def estimate_weather(tick: WeatherTick, rng: np.random.Generator) -> tuple[Estimate, Estimate, Estimate]:
    """Return (temp_c, apparent_c, precip_mm_per_hour) estimates for one minute.

    All three are built the same way: true value + gaussian noise from
    `rng`, with a calibration-defined std and confidence per field. Returns
    (temp_c, apparent_c, precip_mm_per_hour) as a fixed-order tuple.
    """
    cal = WEATHER_NOISE_CALIBRATION

    temp_noise = float(rng.normal(0.0, cal["temp_c_std"]))
    apparent_noise = float(rng.normal(0.0, cal["apparent_c_std"]))

    # `WeatherTick.precip_mm` is the PER-MINUTE share of the hour's
    # accumulation -- see `world.weather.build_weather_timeline`, which splits
    # Open-Meteo's hourly total across 60 minutes, and `_hourly_rate_mm`,
    # which exists there precisely to reverse it.
    #
    # The agent is handed mm PER HOUR, converted here, and the field is named
    # for the unit. This is not tidiness: the agent's rain thresholds are
    # authored in mm/hour (`rain_risk_full_mm` 4.0, `rain_slowdown_per_mm`
    # 0.08) and were being compared against the per-minute value, a factor of
    # 60. Measured consequence on a real rainy shift: the wet-road risk
    # premium came out at 0.01 MXN and the rain slowdown at 0.26% instead of
    # ~16%. The agent did not charge for rain and did not slow down in it --
    # it rode a wet road as though it were dry, and nothing failed loudly
    # because both numbers are plausible-looking small floats.
    rate_mm_per_hour = tick.precip_mm * 60.0
    precip_std = max(
        rate_mm_per_hour * cal["precip_mm_std_fraction"], cal["precip_mm_std_floor_per_hour"]
    )
    precip_noise = float(rng.normal(0.0, precip_std))

    age = cal["age_minutes"]

    temp_c = Estimate(value=tick.temp_c + temp_noise, confidence=cal["temp_c_confidence"], age_minutes=age)
    apparent_c = Estimate(
        value=tick.apparent_c + apparent_noise, confidence=cal["apparent_c_confidence"], age_minutes=age
    )
    precip_mm = Estimate(
        value=max(rate_mm_per_hour + precip_noise, 0.0),
        confidence=cal["precip_mm_confidence"],
        age_minutes=age,
    )
    return temp_c, apparent_c, precip_mm
