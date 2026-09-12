"""Engine layer: the clock that runs a courier through a shift.

`src/engine/` is a DRIVING adapter (see `src.core.ports`'s module docstring
for the dependency-direction diagram): it depends only on `src.core.ports`
and `src.world` (ground truth, which it is allowed to touch), and it must
never create any import path that would let `src.agent` reach `src.world`.

Public surface:
  - `run_shift`, `self_check` (`engine.py`) — the clock itself.
  - `NetworkTravelOracle` (`travel.py`) — the ground-truth `TravelOracle`.
  - `StubPlatform`, `StubRawSource`, `StubEnrichment` (`stubs.py`) —
    trivial, clearly-named stand-ins for the real
    `src/platform/`/`src/enrichment/` adapters, so this package is runnable
    and testable standalone. `StubEnrichment` implements the deprecated
    push-based port and the engine no longer calls it.
  - `calibration` — every tunable constant, in one auditable place.
"""

from src.engine.engine import run_shift, self_check
from src.engine.stubs import StubEnrichment, StubPlatform, StubRawSource
from src.engine.travel import NetworkTravelOracle

__all__ = [
    "run_shift",
    "self_check",
    "NetworkTravelOracle",
    "StubPlatform",
    "StubRawSource",
    "StubEnrichment",
]
