"""Platform layer: the deliberately impoverished view of the world a real
delivery app shows its courier.

See `src.platform.adapter` for the full account of every degradation this
layer applies and why. In one line: the whole product this simulator exists
to demonstrate is information asymmetry, and this package is the "asymmetry"
half of that sentence made concrete in code — everything ground truth
(`src/world/`) knows that this layer does NOT pass through is exactly the
edge a smarter policy is trying to compute for itself.

`src/agent/` must never import this package's internals directly for
decision-making beyond the frozen `PlatformView` it receives each minute —
the whole point is that the policy only ever sees what `PlatformAdapter`
chooses to show it.
"""

from __future__ import annotations

from src.platform.adapter import PlatformAdapter

__all__ = ["PlatformAdapter"]
