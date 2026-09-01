from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class H3BCConfig:
    threshold: float
    start_percent: float = 0.10
    end_percent: float = 0.95
    max_consecutive_hits: int = 2
    probe_blocks: int = 1
    dynamic_threshold: bool = True
    edge_ratio: float = 0.60
    error_budget_units: float = 1.25
    audio_guard_ratio: float = 0.80
    temporal_guard: bool = True

    def validate(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if not (0.0 <= self.start_percent < self.end_percent <= 1.0):
            raise ValueError("cache window must satisfy 0 <= start < end <= 1")
        if self.max_consecutive_hits < 1:
            raise ValueError("max_consecutive_hits must be >= 1")
        if self.probe_blocks < 1:
            raise ValueError("probe_blocks must be >= 1")
        if not (0.05 <= self.edge_ratio <= 1.0):
            raise ValueError("edge_ratio must be in [0.05, 1]")
        if self.error_budget_units <= 0.0:
            raise ValueError("error_budget_units must be > 0")
        if not (0.05 <= self.audio_guard_ratio <= 2.0):
            raise ValueError("audio_guard_ratio must be in [0.05, 2]")


def window_phase(sigma: float, start_sigma: float, end_sigma: float) -> float:
    """Return 0..1 progress inside the protected cache window."""
    denom = start_sigma - end_sigma
    if abs(denom) < 1e-12:
        return 0.5
    return max(0.0, min(1.0, (start_sigma - sigma) / denom))


def threshold_multiplier(phase: float, edge_ratio: float, dynamic: bool) -> float:
    """Smoothly tighten cache at both window edges, relax near the middle."""
    if not dynamic:
        return 1.0
    phase = max(0.0, min(1.0, phase))
    middle = math.sin(math.pi * phase) ** 2
    return edge_ratio + (1.0 - edge_ratio) * middle


def normalized_error(video_diff: float | None, audio_diff: float | None, temporal_diff: float | None,
                     video_threshold: float, audio_threshold: float, temporal_threshold: float,
                     temporal_guard: bool) -> float | None:
    terms = []
    if video_diff is not None:
        terms.append(video_diff / max(video_threshold, 1e-12))
    if audio_diff is not None:
        terms.append(audio_diff / max(audio_threshold, 1e-12))
    if temporal_guard and temporal_diff is not None:
        terms.append(temporal_diff / max(temporal_threshold, 1e-12))
    if not terms:
        return None
    value = max(terms)
    return value if math.isfinite(value) else None
