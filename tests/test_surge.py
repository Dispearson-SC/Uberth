"""Characterization tests: the surge mechanism (src/world/surge.py).

CHARACTERIZATION, NOT UNIT TESTS — see tests/conftest.py's module docstring.
Numbers verified by hand for seed=42, date=2026-07-10 (Friday), shift
840..1320 (14:00-22:00).

This mechanism sits near a bifurcation: k=1.9 kills the oscillation and
k=2.1 breaches the surge ceiling under sweeps performed during calibration.
These tests exist so nobody discovers that the hard way during a live demo.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.world.surge import SURGE_CALIBRATION
from tests.conftest import build_surge_field


def test_mean_surge_at_spawn(reference_orders):
    surges = np.array([o.surge_at_spawn for o in reference_orders])
    assert surges.mean() == pytest.approx(1.138, rel=0.01)


def test_share_of_orders_above_1_2_surge(reference_orders):
    surges = np.array([o.surge_at_spawn for o in reference_orders])
    share = (surges > 1.2).mean()
    # 0.170 after the destination model changed: surge is demand over
    # supply per cell, so moving where customers are moves the demand
    # field. Still inside the 6-18% realism band that is the actual gate;
    # this pin only records where in the band we land.
    assert share == pytest.approx(0.170, rel=0.02)


def test_max_surge_hits_the_configured_ceiling(reference_orders):
    surges = np.array([o.surge_at_spawn for o in reference_orders])
    # Exact: surge is hard-clipped at surge_max (see surge.py's
    # `np.clip(..., surge_min, surge_max)`), so a full shift's worth of
    # orders reaching the cap should equal that constant exactly, not just
    # approximately.
    assert surges.max() == pytest.approx(SURGE_CALIBRATION["surge_max"], abs=1e-9)
    assert SURGE_CALIBRATION["surge_max"] == pytest.approx(2.50, abs=1e-9)


def test_temporal_std_per_cell_is_well_below_realism_bound(reference_surge_field):
    """Mean, over cells, of each cell's own std over time. Bounded well
    under 0.674 (an unrelated ceiling from an earlier, wilder calibration
    sweep) — the ~0.20 actually observed is what the current
    migration_rate/max_outflow_fraction/gap_saturation combination
    produces."""
    temporal_stds = [arr.std() for arr in reference_surge_field.surge_by_cell.values()]
    mean_temporal_std = float(np.mean(temporal_stds))
    assert mean_temporal_std < 0.674
    assert mean_temporal_std == pytest.approx(0.197, rel=0.1)


def test_cross_sectional_std_matches_calibration(reference_surge_field):
    """Mean, over minutes, of the std ACROSS cells at that minute.

    Deliberately NOT the std of each cell's shift-long mean: averaging a
    cell's surge over the whole shift smooths out exactly the oscillation
    this module exists to produce, so that alternative metric would silently
    report a much smaller (and wrong) number for a mechanism that is
    working correctly minute-to-minute.
    """
    cells = reference_surge_field.cells
    matrix = np.array([reference_surge_field.surge_by_cell[c] for c in cells])  # (cells, minutes)

    cross_sectional_std_per_minute = matrix.std(axis=0)
    assert cross_sectional_std_per_minute.mean() == pytest.approx(0.24, rel=0.05)

    # The metric this test is NOT allowed to silently become: std of each
    # cell's own shift-long mean averages away the oscillation and lands far
    # lower. Asserted here (loosely) only to document why it's the wrong
    # metric, not as a target to defend.
    per_cell_means = matrix.mean(axis=1)
    assert per_cell_means.std() < cross_sectional_std_per_minute.mean()


def test_surge_oscillation_is_lag_emergent(reference_surge_field):
    """The central claim of surge.py's module docstring: the spike-collapse
    cycle is an emergent property of the reaction lag, not a scripted
    waveform. Proven by forcing `reaction_lag_min` /
    `reaction_lag_jitter_min` to 0 (couriers react instantly) and observing
    the oscillation collapse toward a near-flat field.

    Passed as a `calibration` override to `build_supply_and_surge` (see
    `tests/conftest.py::build_surge_field`) rather than by mutating the
    module-level `SUPPLY_CALIBRATION` dict directly: the override merges on
    top of a COPY of the calibration inside `build_supply_and_surge`, so
    nothing here ever touches global state and there is nothing to restore
    or leak between tests.
    """
    lagged = reference_surge_field
    lagged_std = float(np.mean([arr.std() for arr in lagged.surge_by_cell.values()]))

    zero_lag = build_surge_field(calibration={"reaction_lag_min": 0, "reaction_lag_jitter_min": 0})
    zero_lag_std = float(np.mean([arr.std() for arr in zero_lag.surge_by_cell.values()]))

    assert zero_lag_std < 0.05  # collapses toward ~0.02; observed ~0.0044
    assert zero_lag_std < lagged_std / 4  # order-of-magnitude collapse, not just "somewhat lower"
