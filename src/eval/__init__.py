"""Evaluation layer: metrics, the A/B harness, and the replay recorder.

This package proves that a smart `Policy` beats a baseline `Policy`, and it
produces the artifact the live demo plays back. It depends only on
`src.core.ports` (the hexagonal contract). Concrete adapters (`engine`,
`platform`, `enrichment`, `agent`) are imported lazily and guarded, never at
module load time, so this package imports cleanly on its own while those
layers are still being written.
"""

from __future__ import annotations
