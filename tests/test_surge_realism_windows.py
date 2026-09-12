"""Characterization tests: surge realism across different shift windows.

CHARACTERIZATION, NOT UNIT TESTS — see tests/conftest.py's module docstring.

These are the numbers a courier would actually experience on each shift.
Every window here builds its own full order stream (~2-3.5s each), so this
module is marked `slow` in bulk — run `pytest -m "not slow"` for a fast dev
loop and the full suite (this file included) before trusting a calibration
change.

Seed and date fixed at 42 / 2026-07-10 (a Friday), matching every other
module in this suite; only the shift window varies.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.world.demand import build_order_stream
from tests.conftest import DATE, SEED

pytestmark = pytest.mark.slow


def _share_above_1_2(start_min: int, end_min: int) -> float:
    orders = build_order_stream(SEED, DATE, start_min, end_min, day_of_week="Friday")
    surges = np.array([o.surge_at_spawn for o in orders])
    return float((surges > 1.2).mean())


def test_early_shift_realism_band_share():
    # 05:00-14:00
    assert _share_above_1_2(300, 840) == pytest.approx(0.171, rel=0.02)


def test_night_shift_realism_band_share():
    # 18:00-02:00, crosses midnight
    assert _share_above_1_2(1080, 1560) == pytest.approx(0.104, rel=0.02)


def test_day_shift_known_limitation_share():
    """12:00-20:00. KNOWN LIMITATION, not a target: this window's share
    above 1.2 surge (~25.8%) sits outside the 6-18% realism band the other
    three windows land in (see surge.py's `COURIER_SUPPLY_TIME_OF_DAY_PROFILE`
    comment block for why — an 8-hour window landing squarely on the lunch
    peak already saturates `surge_max` at zero time-of-day modulation, which
    is a pre-existing property of the calibration, not something a courier-
    supply reshape alone can fix without breaking every other window's own
    band).

    This assertion exists to CATCH DRIFT in this known-bad number, not to
    claim it is acceptable. A test that silently widened its own tolerance
    to make this pass would be worse than no test at all.
    """
    assert _share_above_1_2(720, 1200) == pytest.approx(0.258, rel=0.02)
