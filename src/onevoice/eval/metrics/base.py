"""Shared metric result type."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricResult:
    """One metric's value plus whatever context makes it interpretable."""

    name: str
    value: float
    detail: dict[str, object] = field(default_factory=dict)
