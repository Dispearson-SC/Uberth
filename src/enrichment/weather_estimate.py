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
    """Return (temp_c, apparent_c, precip_mm) estimates for one minute.

    All three are built the same way: true value + gaussian noise from
    `rng`, with a calibration-defined std and confidence per field. Returns
    (temp_c, apparent_c, precip_mm) as a fixed-order tuple.
    """
    cal = WEATHER_NOISE_CALIBRATION

    temp_noise = float(rng.normal(0.0, cal["temp_c_std"]))
    apparent_noise = float(rng.normal(0.0, cal["apparent_c_std"]))

    precip_std = max(tick.precip_mm * cal["precip_mm_std_fraction"], cal["precip_mm_std_floor"])
    precip_noise = float(rng.normal(0.0, precip_std))

    age = cal["age_minutes"]

    temp_c = Estimate(value=tick.temp_c + temp_noise, confidence=cal["temp_c_confidence"], age_minutes=age)
    apparent_c = Estimate(
        value=tick.apparent_c + apparent_noise, confidence=cal["apparent_c_confidence"], age_minutes=age
    )
    precip_mm = Estimate(
        value=max(tick.precip_mm + precip_noise, 0.0),
        confidence=cal["precip_mm_confidence"],
        age_minutes=age,
    )
    return temp_c, apparent_c, precip_mm
