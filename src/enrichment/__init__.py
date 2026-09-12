"""The courier's own external tools: the `EnrichmentPort` driven adapter.

Public surface: `EnrichmentAdapter`, which implements
`src.core.ports.EnrichmentPort`. Everything else in this package is an
internal estimate builder wired together by that class.
"""

from __future__ import annotations

from src.enrichment.adapter import EnrichmentAdapter
from src.enrichment.kitchen_memory import KitchenMemory

__all__ = ["EnrichmentAdapter", "KitchenMemory"]
