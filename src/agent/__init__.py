"""The decision layer.

This package imports `src.core.ports` and the standard library. Nothing else.
There is no import path from here to `src/world/`, `src/engine/`,
`src/platform/` or `src/enrichment/`, and `tests/agent/test_import_boundary.py`
fails the build if one ever appears.

That is not tidiness. If the policy could reach ground truth it could see the
future, its numbers would be unfalsifiable, and a judge would unpick the demo
with one question.
"""

from __future__ import annotations

from src.agent.baseline import AcceptAllPolicy, NearestFirstPolicy
from src.agent.smart import SmartPolicy

__all__ = ["AcceptAllPolicy", "NearestFirstPolicy", "SmartPolicy"]
