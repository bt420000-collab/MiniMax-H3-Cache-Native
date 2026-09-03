from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import comfy.patcher_extension

from .h3bc_policy import (
    H3BCConfig,
    forced_refresh_reason,
    normalized_error,
    parse_refresh_steps,
    threshold_multiplier,
    window_phase,
)

VERSION = "2.0.0-alpha2"
MODE_OFF = "OFF — native H3, no hooks"
MODE_REFERENCE = "LOSSLESS / REFERENCE — exact + profiler"
MODE_SAFE = "SAFE — conservative exact refresh"
MODE_BALANCED = "BALANCED — production baseline"
MODE_AGGRESSIVE = "AGGRESSIVE — experimental"
CUSTOM_MODE = "Custom — manual values"
LEGACY_SAFE = "H3BC Safe α — 0.05"
LEGACY_BALANCED = "H3BC Balanced α — 0.07"
LEGACY_FAST = "H3BC Fast α — 0.09"
PREFETCH_MODES = ["inherit", "disable_dynamic_vbars"]


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Preset:
    config: H3BCConfig
    label: str


PRESETS = {
    MODE_SAFE: Preset(
        H3BCConfig(
            threshold=0.030,
            start_percent=0.10,
            end_percent=0.95,
            max_consecutive_hits=1,
            probe_blocks=1,
            dynamic_threshold=True,
            edge_ratio=0.80,
            error_budget_units=0.90,
            audio_guard_ratio=0.75,
            temporal_guard=True,
            warmup_steps=4,
            force_refresh_every=0,
            adaptive_refresh=True,
            adaptive_refresh_ratio=0.70,
        ),
        "SAFE",
    ),
    MODE_BALANCED: Preset(
        H3BCConfig(
            threshold=0.040,
            start_percent=0.10,
            end_percent=0.95,
            max_consecutive_hits=1,
            probe_blocks=1,
            dynamic_threshold=True,
            edge_ratio=0.82,
            error_budget_units=1.00,
            audio_guard_ratio=0.80,
            temporal_guard=True,
            warmup_steps=4,
            force_refresh_every=0,
            adaptive_refresh=True,
            adaptive_refresh_ratio=0.75,
        ),
        "BALANCED",
    ),
    MODE_AGGRESSIVE: Preset(
        H3BCConfig(
            threshold=0.070,
            start_percent=0.08,
            end_percent=0.96,
            max_consecutive_hits=2,
            probe_blocks=1,
            dynamic_threshold=True,
            edge_ratio=0.68,
            error_budget_units=1.40,
            audio_guard_ratio=0.85,
            temporal_guard=True,
            warmup_steps=3,
            force_refresh_every=0,
            adaptive_refresh=True,
            adaptive_refresh_ratio=0.72,
        ),
        "AGGRESSIVE",
    ),
    LEGACY_SAFE: Preset(
        H3BCConfig(
            threshold=0.05, start_percent=0.10, end_percent=0.95,
            max_consecutive_hits=2, probe_blocks=1, dynamic_threshold=True,
            edge_ratio=0.60, error_budget_units=1.10, audio_guard_ratio=0.75,
            temporal_guard=True, warmup_steps=0, adaptive_refresh=False,
        ),
        "legacy Safe alpha1",
    ),
    LEGACY_BALANCED: Preset(
        H3BCConfig(
            threshold=0.07, start_percent=0.10, end_percent=0.95,
            max_consecutive_hits=2, probe_blocks=1, dynamic_threshold=True,
            edge_ratio=0.58, error_budget_units=1.25, audio_guard_ratio=0.80,
            temporal_guard=True, warmup_steps=0, adaptive_refresh=False,
        ),
        "legacy Balanced alpha1",
    ),
    LEGACY_FAST: Preset(
        H3BCConfig(
            threshold=0.09, start_percent=0.10, end_percent=0.95,
            max_consecutive_hits=2, probe_blocks=1, dynamic_threshold=True,
            edge_ratio=0.55, error_budget_units=1.45, audio_guard_ratio=0.85,
            temporal_guard=True, warmup_steps=0, adaptive_refresh=False,
        ),
        "legacy Fast alpha1",
    ),
}


@dataclass
class CacheContext:
    key: tuple = ("default",)
    previous_probe_residual: torch.Tensor | None = None
    tail_residual: torch.Tensor | None = None
    probe_input: torch.Tensor | None = None
    probe_output: torch.Tensor | None = None
    pending_probe_residual: torch.Tensor | None = None
    use_cache: bool = False
    consecutive_hits: int = 0
    budget_spent: float = 0.0
    previous_sigma: float | None = None
    input_signature: tuple | None = None
    audio_slice: tuple[int, int] | None = None
    video_slice: tuple[int, int] | None = None
    latent_frames: int | None = None
    last_metrics: dict | None = None
    step_index: int = 0
    force_refresh_next: bool = False
    external_force_refresh: bool = False

    def clear_tensors(self, *, preserve_step: bool = False):
        step_index = self.step_index if preserve_step else 0
        self.previous_probe_residual = None
        self.tail_residual = None
        self.probe_input = None
        self.probe_output = None
        self.pending_probe_residual = None
        self.use_cache = False
        self.consecutive_hits = 0
        self.budget_spent = 0.0
        self.previous_sigma = None
        self.input_signature = None
        self.audio_slice = None
        self.video_slice = None
        self.latent_frames = None
        self.last_metrics = None
        self.step_index = step_index
        self.force_refresh_next = False
        self.external_force_refresh = False


@dataclass
class BlockProfileState:
    global_sample: torch.Tensor | None = None
    audio_sample: torch.Tensor | None = None
    video_sample: torch.Tensor | None = None


@dataclass
class BlockProfileRecord:
    step: int
    block: int
    global_diff: Any = None
    audio_diff: Any = None
    video_diff: Any = None
    residual_norm: Any = None
    start_event: Any = None
    end_event: Any = None
    cpu_ms: float | None = None


class H3BCReferenceProfiler:
    """Low-memory block residual profiler.

    It stores only deterministic sampled residuals per block, not full block outputs,
    so profiling all 50 H3 blocks does not turn the cache engine into an OOM engine.
    """

    def __init__(self, block_count: int, debug: bool, sample_tokens: int = 64, sample_channels: int = 32):
        self.block_count = int(block_count)
        self.debug = bool(debug)
        self.sample_tokens = int(sample_tokens)
        self.sample_channels = int(sample_channels)
        self.previous: dict[tuple, dict[int, BlockProfileState]] = {}
        self.records: list[BlockProfileRecord] = []
        self._devices: set[torch.device] = set()
        self._finalized: list[dict] | None = None

    def reset(self):
        self.previous.clear()
        self.records.clear()
        self._devices.clear()
        self._finalized = None

    @staticmethod
    def _span_tensor(x: torch.Tensor, span: tuple[int, int] | None):
        if span is None or x.ndim < 2:
            return None
        a, b = span
        if a < 0 or b <= a or b > x.shape[0]:
            return None
        return x[a:b]

    def _sample(self, x: torch.Tensor | None):
        if x is None or not torch.is_tensor(x) or x.numel() == 0:
            return None
        if x.ndim == 1:
            flat = x.reshape(-1, 1)
        else:
            flat = x.reshape(-1, x.shape[-1])
        rows = min(self.sample_tokens, flat.shape[0])
        cols = min(self.sample_channels, flat.shape[1])
        if rows <= 0 or cols <= 0:
            return None
        row_idx = torch.linspace(0, flat.shape[0] - 1, rows, device=flat.device).round().long()
        col_idx = torch.linspace(0, flat.shape[1] - 1, cols, device=flat.device).round().long()
        return flat.index_select(0, row_idx).index_select(1, col_idx).detach()

    @staticmethod
    def _relative_tensor(current: torch.Tensor | None, previous: torch.Tensor | None):
        if current is None or previous is None or current.shape != previous.shape:
            return None
        num = (current - previous).abs().mean()
        den = previous.abs().mean().clamp(min=1e-8)
        return (num / den).detach()

    def begin_timing(self, tensor: torch.Tensor):
        if self.debug and torch.is_tensor(tensor) and tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._devices.add(tensor.device)
            return start, end, None
        return None, None, time.perf_counter()

    def observe_block(
        self,
        *,
        context: CacheContext,
        block_index: int,
        block_input: torch.Tensor,
        block_output: torch.Tensor,
        timing,
    ):
        start_event, end_event, cpu_start = timing
        if end_event is not None:
            end_event.record()
            cpu_ms = None
        else:
            cpu_ms = (time.perf_counter() - cpu_start) * 1000.0 if cpu_start is not None else None

        residual = (block_output - block_input).detach()
        global_sample = self._sample(residual)
        audio_sample = self._sample(self._span_tensor(residual, context.audio_slice))
        video_sample = self._sample(self._span_tensor(residual, context.video_slice))
        states = self.previous.setdefault(context.key, {})
        prev = states.get(block_index, BlockProfileState())
        record = BlockProfileRecord(
            step=context.step_index,
            block=block_index,
            global_diff=self._relative_tensor(global_sample, prev.global_sample),
            audio_diff=self._relative_tensor(audio_sample, prev.audio_sample),
            video_diff=self._relative_tensor(video_sample, prev.video_sample),
            residual_norm=(global_sample.abs().mean().detach() if global_sample is not None else None),
            start_event=start_event,
            end_event=end_event,
            cpu_ms=cpu_ms,
        )
        self.records.append(record)
        states[block_index] = BlockProfileState(global_sample, audio_sample, video_sample)

    @staticmethod
    def _as_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.item())
        return float(value)

    def finalize(self):
        if self._finalized is not None:
            return self._finalized
        if self.debug:
            for device in self._devices:
                torch.cuda.synchronize(device)
        out = []
        for rec in self.records:
            block_ms = rec.cpu_ms
            if rec.start_event is not None and rec.end_event is not None:
                try:
                    block_ms = float(rec.start_event.elapsed_time(rec.end_event))
                except Exception:
                    block_ms = None
            out.append({
                "step": rec.step,
                "block": rec.block,
                "global_diff": self._as_float(rec.global_diff),
                "audio_diff": self._as_float(rec.audio_diff),
                "video_diff": self._as_float(rec.video_diff),
                "residual_norm": self._as_float(rec.residual_norm),
                "block_ms": block_ms,
            })
        self._finalized = out
        return out

    def aggregate(self):
        records = self.finalize()
        by_block: dict[int, dict[str, list[float]]] = {}
        for rec in records:
            bucket = by_block.setdefault(rec["block"], {"global": [], "audio": [], "video": [], "ms": []})
            for key, dst in (("global_diff", "global"), ("audio_diff", "audio"), ("video_diff", "video"), ("block_ms", "ms")):
                value = rec[key]
                if value is not None and math.isfinite(value):
                    bucket[dst].append(value)
        summary = []
        for block, values in sorted(by_block.items()):
            def avg(name):
                xs = values[name]
                return sum(xs) / len(xs) if xs else None
            summary.append({
                "block": block,
                "mean_residual_diff": avg("global"),
                "mean_audio_diff": avg("audio"),
                "mean_video_diff": avg("video"),
                "mean_block_ms": avg("ms"),
                "samples": len(values["global"]),
            })
        return summary


class H3BCAdaptiveCache:
    def __init__(
        self,
        config: H3BCConfig,
        start_sigma: float,
        end_sigma: float,
        block_count: int,
        verbose: bool,
        *,
        run_mode: str = "CUSTOM",
        reference_mode: bool = False,
        debug: bool = False,
    ):
        config.validate()
        if config.probe_blocks >= block_count:
            raise ValueError(f"probe_blocks must be smaller than H3 block count ({block_count})")
        self.config = config
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.block_count = int(block_count)
        self.verbose = bool(verbose)
        self.run_mode = str(run_mode)
        self.reference_mode = bool(reference_mode)
        self.debug = bool(debug)
        self.contexts: dict[tuple, CacheContext] = {}
        self.current: CacheContext | None = None
        self.full_steps = 0
        self.cached_steps = 0
        self.cached_step_numbers: list[int] = []
        self.metrics: list[dict] = []
        self.reason_counts: dict[str, int] = {}
        self.forced_refresh_count = 0
        self.decision_gate_ms = 0.0
        self.cache_apply_cpu_ms = 0.0
        self._last_external_reset_token = object()
        self.profiler = H3BCReferenceProfiler(block_count, debug=(debug or reference_mode))
        self.step_records: list[dict] = []

    def reset(self):
        for context in self.contexts.values():
            context.clear_tensors()
        self.contexts.clear()
        self.current = None
        self.full_steps = 0
        self.cached_steps = 0
        self.cached_step_numbers.clear()
        self.metrics.clear()
        self.reason_counts.clear()
        self.forced_refresh_count = 0
        self.decision_gate_ms = 0.0
        self.cache_apply_cpu_ms = 0.0
        self.profiler.reset()
        self.step_records.clear()

    @staticmethod
    def _input_signature(x):
        tensors = x if isinstance(x, (tuple, list)) else (x,)
        return tuple((tuple(t.shape), t.dtype, t.device) for t in tensors if torch.is_tensor(t))

    @staticmethod
    def _layout_info(minimax_payload):
        if not minimax_payload:
            return None, None, None
        layout = minimax_payload.get("layout")
        if layout is None:
            return None, None, None
        audio_slice = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
        video_slice = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
        latent_frames = layout.signature[1] if len(layout.signature) > 1 else None
        return audio_slice, video_slice, latent_frames

    @staticmethod
    def _external_control(transformer_options: dict):
        control = transformer_options.get("h3bc_control", {})
        return control if isinstance(control, dict) else {}

    def begin_call(self, x, timestep, transformer_options, minimax_payload=None):
        sigma = float(timestep.flatten()[0].item()) / 1000.0
        uuids = transformer_options.get("uuids")
        key = tuple(str(v) for v in uuids) if uuids else ("default",)
        context = self.contexts.setdefault(key, CacheContext(key=key))
        signature = self._input_signature(x)

        control = self._external_control(transformer_options)
        reset_token = control.get("reset_cache", None)
        if reset_token not in (None, False) and reset_token != self._last_external_reset_token:
            context.clear_tensors()
            self._last_external_reset_token = reset_token
            signature = self._input_signature(x)

        if context.input_signature != signature or (context.previous_sigma is not None and sigma > context.previous_sigma + 1e-7):
            context.clear_tensors()
            context.input_signature = signature
        context.step_index += 1
        context.previous_sigma = sigma
        context.audio_slice, context.video_slice, context.latent_frames = self._layout_info(minimax_payload)
        context.probe_input = None
        context.probe_output = None
        context.pending_probe_residual = None
        context.use_cache = False
        context.last_metrics = None

        force_steps = control.get("force_refresh_step", control.get("force_refresh_steps", ()))
        if isinstance(force_steps, int):
            force_steps = (force_steps,)
        try:
            force_steps = {int(v) for v in force_steps} if force_steps else set()
        except Exception:
            force_steps = set()
        context.external_force_refresh = bool(control.get("force_refresh", False) or context.step_index in force_steps)
        self.current = context

    def end_call(self):
        self.current = None

    def _within_window(self, context: CacheContext) -> bool:
        return context.previous_sigma is not None and self.end_sigma <= context.previous_sigma <= self.start_sigma

    @staticmethod
    def _rel_l1(current: torch.Tensor, previous: torch.Tensor, span: tuple[int, int] | None):
        if span is None:
            return None
        a, b = span
        if a < 0 or b > current.shape[0] or b <= a:
            return None
        cur = current[a:b]
        prev = previous[a:b]
        if cur.shape != prev.shape or cur.numel() == 0:
            return None
        numerator = (cur - prev).abs().mean()
        denominator = prev.abs().mean().clamp(min=1e-8)
        return float((numerator / denominator).item())

    @staticmethod
    def _temporal_diff(current: torch.Tensor, previous: torch.Tensor, context: CacheContext):
        if context.video_slice is None or not context.latent_frames:
            return None
        a, b = context.video_slice
        cur = current[a:b]
        prev = previous[a:b]
        frames = int(context.latent_frames)
        if cur.shape != prev.shape or frames <= 0 or cur.shape[0] % frames:
            return None
        rows_per_frame = cur.shape[0] // frames
        cur = cur.reshape(frames, rows_per_frame, -1)
        prev = prev.reshape(frames, rows_per_frame, -1)
        numerator = (cur - prev).abs().mean(dim=(1, 2))
        denominator = prev.abs().mean(dim=(1, 2)).clamp(min=1e-8)
        return float((numerator / denominator).max().item())

    def capture_probe_input(self, x: torch.Tensor):
        context = self.current
        if context is None:
            raise RuntimeError("H3BC probe input captured outside model execution")
        context.probe_input = x.detach()

    @torch.compiler.disable()
    def decide_after_probe(self, probe_output: torch.Tensor):
        gate_start = time.perf_counter()
        context = self.current
        if context is None or context.probe_input is None:
            raise RuntimeError("H3BC probe state is incomplete")
        probe_residual = probe_output - context.probe_input
        previous = context.previous_probe_residual
        tail = context.tail_residual
        can_compare = previous is not None and tail is not None and previous.shape == probe_residual.shape and tail.shape == probe_output.shape
        reason = "exact:no-cache"
        use_cache = False
        metrics = {
            "step": context.step_index,
            "video": None,
            "audio": None,
            "temporal": None,
            "multiplier": 1.0,
            "normalized": None,
            "budget_before": context.budget_spent,
            "effective_video_threshold": None,
            "effective_audio_threshold": None,
            "consecutive_before": context.consecutive_hits,
        }

        refresh_reason = forced_refresh_reason(
            step_index=context.step_index,
            config=self.config,
            consecutive_hits=context.consecutive_hits,
            force_refresh_next=context.force_refresh_next,
            external_force_refresh=context.external_force_refresh,
        )
        if refresh_reason is not None:
            reason = f"refresh:{refresh_reason}"
            self.forced_refresh_count += 1
        elif can_compare and self._within_window(context):
            phase = window_phase(context.previous_sigma, self.start_sigma, self.end_sigma)
            mult = threshold_multiplier(phase, self.config.edge_ratio, self.config.dynamic_threshold)
            video_threshold = self.config.threshold * mult
            audio_threshold = self.config.threshold * self.config.audio_guard_ratio * mult
            temporal_threshold = video_threshold
            video_diff = self._rel_l1(probe_residual, previous, context.video_slice)
            audio_diff = self._rel_l1(probe_residual, previous, context.audio_slice)
            temporal_diff = self._temporal_diff(probe_residual, previous, context) if self.config.temporal_guard else None

            if video_diff is None and audio_diff is None:
                numerator = (probe_residual - previous).abs().mean()
                denominator = previous.abs().mean().clamp(min=1e-8)
                video_diff = float((numerator / denominator).item())

            norm = normalized_error(
                video_diff,
                audio_diff,
                temporal_diff,
                video_threshold,
                audio_threshold,
                temporal_threshold,
                self.config.temporal_guard,
            )
            metrics.update({
                "video": video_diff,
                "audio": audio_diff,
                "temporal": temporal_diff,
                "multiplier": mult,
                "normalized": norm,
                "effective_video_threshold": video_threshold,
                "effective_audio_threshold": audio_threshold,
            })
            if norm is None:
                reason = "exact:invalid-diff"
            elif norm > 1.0:
                reason = "exact:guard"
            elif context.budget_spent + norm > self.config.error_budget_units:
                reason = "exact:error-budget"
            else:
                use_cache = True
                reason = "cache"
        elif can_compare:
            reason = "exact:outside-window"

        context.use_cache = use_cache
        context.last_metrics = metrics
        self.metrics.append(metrics)
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        context.force_refresh_next = False
        context.external_force_refresh = False

        if use_cache:
            context.consecutive_hits += 1
            context.budget_spent += float(metrics["normalized"] or 0.0)
            self.cached_step_numbers.append(context.step_index)
            if self.config.adaptive_refresh and (metrics["normalized"] or 0.0) >= self.config.adaptive_refresh_ratio:
                context.force_refresh_next = True
            context.probe_input = None
            context.probe_output = None
            context.pending_probe_residual = None
        else:
            context.consecutive_hits = 0
            context.budget_spent = 0.0
            context.probe_output = probe_output.detach()
            context.pending_probe_residual = probe_residual.detach().clone()

        gate_ms = (time.perf_counter() - gate_start) * 1000.0
        self.decision_gate_ms += gate_ms
        self.step_records.append({
            "step": context.step_index,
            "action": "CACHE" if use_cache else "EXACT",
            "reason": reason,
            "sigma": context.previous_sigma,
            "consecutive": context.consecutive_hits,
            "decision_gate_ms": gate_ms,
            **{k: metrics.get(k) for k in ("video", "audio", "temporal", "normalized")},
        })

        if self.verbose or self.debug:
            vd = metrics["video"]
            ad = metrics["audio"]
            td = metrics["temporal"]
            nd = metrics["normalized"]
            logging.info(
                "[H3BC] step=%02d %s sigma=%.5f v=%s a=%s t=%s norm=%s cached_run=%d gate=%.2fms",
                context.step_index,
                reason,
                context.previous_sigma,
                "-" if vd is None else f"{vd:.5f}",
                "-" if ad is None else f"{ad:.5f}",
                "-" if td is None else f"{td:.5f}",
                "-" if nd is None else f"{nd:.3f}",
                context.consecutive_hits,
                gate_ms,
            )

    def finish_full_step(self, output: torch.Tensor):
        context = self.current
        if context is None or context.probe_output is None or context.pending_probe_residual is None:
            raise RuntimeError("H3BC full-step state is incomplete")
        context.tail_residual = (output - context.probe_output).detach().clone()
        context.previous_probe_residual = context.pending_probe_residual
        context.probe_input = None
        context.probe_output = None
        context.pending_probe_residual = None
        self.full_steps += 1

    def finish_reference_step(self):
        self.full_steps += 1

    def finish_cached_step(self, probe_output: torch.Tensor):
        context = self.current
        if context is None or context.tail_residual is None:
            raise RuntimeError("H3BC has no cached tail residual")
        start = time.perf_counter()
        result = probe_output + context.tail_residual
        self.cache_apply_cpu_ms += (time.perf_counter() - start) * 1000.0
        self.cached_steps += 1
        return result

    def _profile_estimates(self):
        block_summary = self.profiler.aggregate()
        mean_ms = {row["block"]: row["mean_block_ms"] for row in block_summary if row["mean_block_ms"] is not None}
        if not mean_ms or not self.cached_steps:
            return None, None, None
        skipped_ms_per_hit = sum(mean_ms.get(i, 0.0) for i in range(self.config.probe_blocks, self.block_count))
        gross_saved = skipped_ms_per_hit * self.cached_steps
        overhead = self.decision_gate_ms + self.cache_apply_cpu_ms
        net_saved = gross_saved - overhead
        exact_block_total = sum(mean_ms.get(i, 0.0) for i in range(self.block_count))
        estimated_actual = self.full_steps * exact_block_total + self.cached_steps * sum(
            mean_ms.get(i, 0.0) for i in range(self.config.probe_blocks)
        ) + overhead
        estimated_reference = (self.full_steps + self.cached_steps) * exact_block_total
        speedup = estimated_reference / estimated_actual if estimated_actual > 0 else None
        return gross_saved, net_saved, speedup

    def summary_dict(self):
        steps = self.full_steps + self.cached_steps
        hit_rate = self.cached_steps / steps if steps else 0.0
        gross_saved, net_saved, estimated_speedup = self._profile_estimates()
        return {
            "version": VERSION,
            "mode": self.run_mode,
            "steps": steps,
            "blocks": self.block_count,
            "exact_calls": self.full_steps,
            "cache_hits": self.cached_steps,
            "hit_rate": hit_rate,
            "cached_steps": list(self.cached_step_numbers),
            "forced_refresh": self.forced_refresh_count,
            "decision_gate_ms": self.decision_gate_ms,
            "cache_apply_cpu_ms": self.cache_apply_cpu_ms,
            "gross_compute_saved_ms_est": gross_saved,
            "net_saved_ms_est": net_saved,
            "speedup_est": estimated_speedup,
            "reasons": dict(self.reason_counts),
            "block_profile": self.profiler.aggregate(),
        }

    def summary(self):
        data = self.summary_dict()
        speed = "n/a" if data["speedup_est"] is None else f"{data['speedup_est']:.2f}x"
        saved = "n/a" if data["net_saved_ms_est"] is None else f"{data['net_saved_ms_est']:.1f}ms"
        return (
            f"steps={data['steps']} blocks={data['blocks']} exact={data['exact_calls']} cache={data['cache_hits']} "
            f"hit_rate={data['hit_rate']:.1%} forced_refresh={data['forced_refresh']} "
            f"gate_overhead={data['decision_gate_ms']:.1f}ms cache_apply_cpu={data['cache_apply_cpu_ms']:.1f}ms "
            f"net_saved_est={saved} speedup_est={speed} cached_steps={data['cached_steps']} "
            f"reasons={data['reasons']}"
        )

    def export_debug(self):
        if not self.debug:
            return None
        try:
            import folder_paths
            root = Path(folder_paths.get_output_directory()) / "h3bc"
        except Exception:
            root = Path.cwd() / "output" / "h3bc"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"h3bc_{VERSION}_{stamp}.json"
        payload = self.summary_dict()
        payload["step_records"] = self.step_records
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def make_block_patch(cache: H3BCAdaptiveCache, index: int, last_index: int):
    probe_last = cache.config.probe_blocks - 1

    def patch(args, extra):
        original_block = extra["original_block"]
        block_input = args["img"]

        if cache.reference_mode:
            timing = cache.profiler.begin_timing(block_input)
            output = original_block(args)["img"]
            cache.profiler.observe_block(
                context=cache.current,
                block_index=index,
                block_input=block_input,
                block_output=output,
                timing=timing,
            )
            if index == last_index:
                cache.finish_reference_step()
            return {"img": output}

        if index == 0:
            cache.capture_probe_input(block_input)

        if index <= probe_last:
            timing = cache.profiler.begin_timing(block_input) if cache.debug else (None, None, None)
            output = original_block(args)["img"]
            if cache.debug:
                cache.profiler.observe_block(
                    context=cache.current,
                    block_index=index,
                    block_input=block_input,
                    block_output=output,
                    timing=timing,
                )
            if index == probe_last:
                cache.decide_after_probe(output)
            return {"img": output}

        context = cache.current
        if context is None:
            raise RuntimeError("H3BC has no active context")

        if context.use_cache:
            if index == last_index:
                return {"img": cache.finish_cached_step(args["img"])}
            return {"img": args["img"]}

        timing = cache.profiler.begin_timing(block_input) if cache.debug else (None, None, None)
        output = original_block(args)["img"]
        if cache.debug:
            cache.profiler.observe_block(
                context=context,
                block_index=index,
                block_input=block_input,
                block_output=output,
                timing=timing,
            )
        if index == last_index:
            cache.finish_full_step(output)
        return {"img": output}

    return patch


def make_diffusion_wrapper(cache: H3BCAdaptiveCache):
    def wrapper(executor, *args, **kwargs):
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        minimax_payload = args[4] if len(args) > 4 else kwargs.get("minimax_payload")
        cache.begin_call(args[0], args[1], transformer_options, minimax_payload)
        try:
            return executor(*args, **kwargs)
        finally:
            cache.end_call()
    return wrapper


def make_sample_wrapper(cache: H3BCAdaptiveCache, label: str):
    def wrapper(executor, *args, **kwargs):
        cache.reset()
        logging.info("H3BC enabled: %s", label)
        try:
            return executor(*args, **kwargs)
        finally:
            logging.info("H3BC Summary: %s", cache.summary())
            if cache.reference_mode:
                rows = [r for r in cache.profiler.aggregate() if r["mean_residual_diff"] is not None]
                stable = sorted(rows, key=lambda r: r["mean_residual_diff"])[:8]
                expensive = sorted(
                    [r for r in rows if r["mean_block_ms"] is not None],
                    key=lambda r: r["mean_block_ms"],
                    reverse=True,
                )[:8]
                logging.info("[H3BC] profiler stable blocks: %s", stable)
                logging.info("[H3BC] profiler expensive blocks: %s", expensive)
            path = cache.export_debug()
            if path is not None:
                logging.info("[H3BC] telemetry: %s", path)
            cache.reset()
    return wrapper


class ApplyMiniMaxH3BC:
    @classmethod
    def INPUT_TYPES(cls):
        modes = [
            MODE_OFF,
            MODE_REFERENCE,
            MODE_SAFE,
            MODE_BALANCED,
            MODE_AGGRESSIVE,
            CUSTOM_MODE,
            LEGACY_SAFE,
            LEGACY_BALANCED,
            LEGACY_FAST,
        ]
        return {"required": {
            "model": ("MODEL",),
            "mode": (modes, {"default": MODE_BALANCED}),
            "threshold": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.30, "step": 0.005}),
            "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            "max_consecutive_hits": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "probe_blocks": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "dynamic_threshold": ("BOOLEAN", {"default": True}),
            "edge_ratio": ("FLOAT", {"default": 0.82, "min": 0.05, "max": 1.0, "step": 0.01}),
            "error_budget_units": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 8.0, "step": 0.05}),
            "audio_guard_ratio": ("FLOAT", {"default": 0.80, "min": 0.05, "max": 2.0, "step": 0.05}),
            "temporal_guard": ("BOOLEAN", {"default": True}),
            "prefetch_mode": (PREFETCH_MODES, {"default": "inherit"}),
            "verbose": ("BOOLEAN", {"default": True}),
            "warmup_steps": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
            "force_refresh_every": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            "force_refresh_steps": ("STRING", {"default": "", "multiline": False}),
            "adaptive_refresh": ("BOOLEAN", {"default": True}),
            "adaptive_refresh_ratio": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/optimization"
    DESCRIPTION = (
        "H3BC native H3 cache engine. alpha2 adds exact-refresh policy, lossless reference profiling, "
        "task-boundary reset, external refresh controls, telemetry, and conservative production presets. "
        "Set H3BC_DEBUG=1 for detailed JSON telemetry."
    )

    def apply(
        self,
        model,
        mode,
        threshold,
        start_percent,
        end_percent,
        max_consecutive_hits,
        probe_blocks,
        dynamic_threshold,
        edge_ratio,
        error_budget_units,
        audio_guard_ratio,
        temporal_guard,
        prefetch_mode,
        verbose,
        warmup_steps=4,
        force_refresh_every=0,
        force_refresh_steps="",
        adaptive_refresh=True,
        adaptive_refresh_ratio=0.75,
    ):
        if mode == MODE_OFF:
            return (model,)

        reference_mode = mode == MODE_REFERENCE
        if reference_mode:
            config = H3BCConfig(
                threshold=0.0,
                start_percent=0.0,
                end_percent=1.0,
                max_consecutive_hits=1,
                probe_blocks=1,
                dynamic_threshold=False,
                edge_ratio=1.0,
                error_budget_units=1.0,
                audio_guard_ratio=1.0,
                temporal_guard=True,
                warmup_steps=0,
                adaptive_refresh=False,
            )
            label = MODE_REFERENCE
        elif mode == CUSTOM_MODE:
            config = H3BCConfig(
                threshold=threshold,
                start_percent=start_percent,
                end_percent=end_percent,
                max_consecutive_hits=max_consecutive_hits,
                probe_blocks=probe_blocks,
                dynamic_threshold=dynamic_threshold,
                edge_ratio=edge_ratio,
                error_budget_units=error_budget_units,
                audio_guard_ratio=audio_guard_ratio,
                temporal_guard=temporal_guard,
                warmup_steps=warmup_steps,
                force_refresh_every=force_refresh_every,
                force_refresh_steps=parse_refresh_steps(force_refresh_steps),
                adaptive_refresh=adaptive_refresh,
                adaptive_refresh_ratio=adaptive_refresh_ratio,
            )
            label = (
                f"Custom threshold={threshold:.3f} window={start_percent:.2f}-{end_percent:.2f} "
                f"max_cached_run={max_consecutive_hits} probe={probe_blocks} warmup={warmup_steps} "
                f"refresh_every={force_refresh_every} refresh_steps={config.force_refresh_steps} "
                f"adaptive={adaptive_refresh}:{adaptive_refresh_ratio:.2f} prefetch={prefetch_mode}"
            )
        else:
            preset = PRESETS[mode]
            config = preset.config
            label = f"{preset.label}; prefetch={prefetch_mode}"

        config.validate()
        diffusion_model = model.get_model_object("diffusion_model")
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model" or not hasattr(diffusion_model, "blocks"):
            raise ValueError(f"H3BC only supports native MiniMaxH3Model, got {diffusion_model.__class__.__name__}")
        block_count = len(diffusion_model.blocks)
        if config.probe_blocks >= block_count:
            raise ValueError(f"probe_blocks={config.probe_blocks} must be smaller than H3 block count={block_count}")

        transformer_options = model.model_options.get("transformer_options", {})
        conflict_keys = {
            "easycache": "EasyCache/LazyCache",
            "cache_dit_turbo": "CacheDiT",
            "minimax_h3_block_cache_t8": "MiniMax H3 Block Cache (T8)",
            "h3bc_v2": "another H3BC node",
        }
        for key, name in conflict_keys.items():
            if key in transformer_options:
                raise ValueError(f"H3BC cannot be combined with {name}")
        existing = transformer_options.get("patches_replace", {}).get("dit", {})
        collisions = [i for i in range(block_count) if ("double_block", i) in existing]
        if collisions:
            raise ValueError(
                "H3BC conflicts with another double_block replacement. Connect H3BC directly after the diffusion model loader. "
                f"Colliding blocks: {collisions[:8]}"
            )

        model_sampling = model.get_model_object("model_sampling")
        start_sigma = float(model_sampling.percent_to_sigma(config.start_percent))
        end_sigma = float(model_sampling.percent_to_sigma(config.end_percent))

        patched = model.clone()
        patched_options = patched.model_options.setdefault("transformer_options", {})
        patched_options["h3bc_v2"] = {
            "version": VERSION,
            "mode": mode,
            "probe_blocks": config.probe_blocks,
            "warmup_steps": config.warmup_steps,
            "max_cached_run": config.max_cached_run,
        }
        patched_options.setdefault("h3bc_control", {})
        if prefetch_mode == "disable_dynamic_vbars":
            patched_options["prefetch_dynamic_vbars"] = False

        debug = _env_truthy("H3BC_DEBUG")
        cache = H3BCAdaptiveCache(
            config,
            start_sigma,
            end_sigma,
            block_count,
            verbose,
            run_mode=mode,
            reference_mode=reference_mode,
            debug=debug,
        )
        for index in range(block_count):
            patched.set_model_patch_replace(make_block_patch(cache, index, block_count - 1), "dit", "double_block", index)

        key = f"h3bc_v2_{id(cache)}"
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key, make_diffusion_wrapper(cache))
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key, make_sample_wrapper(cache, label))
        return (patched,)


NODE_CLASS_MAPPINGS = {"ApplyMiniMaxH3BC": ApplyMiniMaxH3BC}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyMiniMaxH3BC": "MiniMax H3BC Native Cache Engine (α2)"}
