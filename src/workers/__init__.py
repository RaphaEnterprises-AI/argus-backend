"""Argus Workers - Background processing workers for event-driven architecture."""

from .cognee_consumer import CogneeConfig, CogneeConsumer

__all__ = ["CogneeConsumer", "CogneeConfig"]
