from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


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
    warmup_steps: int = 0
    force_refresh_every: int = 0
    force_refresh_steps: tuple[int, ...] = ()
    adaptive_refresh: bool = True
    adaptive_refresh_ratio: float = 0.75

    @property
    def max_cached_run(self) -> int:
        """Production-facing alias for the legacy alpha1 field name."""
        return self.max_consecutive_hits

    def validate(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        if not (0.0 <= self.start_percent < self.end_percent <= 1.0):
            raise ValueError("cache window must satisfy 0 <= start < end <= 1")
        if self.max_consecutive_hits < 1:
            raise ValueError("max_consecutive_hits/max_cached_run must be >= 1")
        if self.probe_blocks < 1:
            raise ValueError("probe_blocks must be >= 1")
        if not (0.05 <= self.edge_ratio <= 1.0):
            raise ValueError("edge_ratio must be in [0.05, 1]")
        if self.error_budget_units <= 0.0:
            raise ValueError("error_budget_units must be > 0")
        if not (0.05 <= self.audio_guard_ratio <= 2.0):
            raise ValueError("audio_guard_ratio must be in [0.05, 2]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.force_refresh_every < 0:
            raise ValueError("force_refresh_every must be >= 0")
        if any(step < 1 for step in self.force_refresh_steps):
            raise ValueError("force_refresh_steps are 1-based and must be >= 1")
        if not (0.0 <= self.adaptive_refresh_ratio <= 1.0):
            raise ValueError("adaptive_refresh_ratio must be in [0, 1]")


def parse_refresh_steps(value: str | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        values = [int(p) for p in parts]
    else:
        values = [int(v) for v in value]
    return tuple(sorted(set(values)))


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


def normalized_error(
    video_diff: float | None,
    audio_diff: float | None,
    temporal_diff: float | None,
    video_threshold: float,
    audio_threshold: float,
    temporal_threshold: float,
    temporal_guard: bool,
) -> float | None:
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


def forced_refresh_reason(
    *,
    step_index: int,
    config: H3BCConfig,
    consecutive_hits: int,
    force_refresh_next: bool = False,
    external_force_refresh: bool = False,
) -> str | None:
    """Return a reason when the current step must be exact before cache gating."""
    if step_index <= config.warmup_steps:
        return "warmup"
    if external_force_refresh:
        return "external"
    if step_index in config.force_refresh_steps:
        return "scheduled-step"
    if config.force_refresh_every and step_index % config.force_refresh_every == 0:
        return "periodic"
    if force_refresh_next:
        return "adaptive"
    if consecutive_hits >= config.max_cached_run:
        return "max-cached-run"
    return None
