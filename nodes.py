from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import comfy.patcher_extension

from .h3bc_policy import H3BCConfig, normalized_error, threshold_multiplier, window_phase


@dataclass(frozen=True)
class Preset:
    config: H3BCConfig
    label: str


PRESETS = {
    "H3BC Safe α — 0.05": Preset(H3BCConfig(
        threshold=0.05, start_percent=0.10, end_percent=0.95,
        max_consecutive_hits=2, probe_blocks=1, dynamic_threshold=True,
        edge_ratio=0.60, error_budget_units=1.10, audio_guard_ratio=0.75,
        temporal_guard=True,
    ), "Safe alpha"),
    "H3BC Balanced α — 0.07": Preset(H3BCConfig(
        threshold=0.07, start_percent=0.10, end_percent=0.95,
        max_consecutive_hits=2, probe_blocks=1, dynamic_threshold=True,
        edge_ratio=0.58, error_budget_units=1.25, audio_guard_ratio=0.80,
        temporal_guard=True,
    ), "Balanced alpha"),
    "H3BC Fast α — 0.09": Preset(H3BCConfig(
        threshold=0.09, start_percent=0.10, end_percent=0.95,
        max_consecutive_hits=2, probe_blocks=1, dynamic_threshold=True,
        edge_ratio=0.55, error_budget_units=1.45, audio_guard_ratio=0.85,
        temporal_guard=True,
    ), "Fast alpha"),
}
CUSTOM_MODE = "Custom — manual values"
PREFETCH_MODES = ["inherit", "disable_dynamic_vbars"]


@dataclass
class CacheContext:
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

    def clear_tensors(self):
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


class H3BCAdaptiveCache:
    def __init__(self, config: H3BCConfig, start_sigma: float, end_sigma: float, block_count: int, verbose: bool):
        config.validate()
        if config.probe_blocks >= block_count:
            raise ValueError(f"probe_blocks must be smaller than H3 block count ({block_count})")
        self.config = config
        self.start_sigma = float(start_sigma)
        self.end_sigma = float(end_sigma)
        self.block_count = int(block_count)
        self.verbose = bool(verbose)
        self.contexts: dict[tuple, CacheContext] = {}
        self.current: CacheContext | None = None
        self.full_steps = 0
        self.cached_steps = 0
        self.cached_step_numbers: list[int] = []
        self.metrics: list[dict] = []
        self.reason_counts: dict[str, int] = {}

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

    def begin_call(self, x, timestep, transformer_options, minimax_payload=None):
        sigma = float(timestep.flatten()[0].item()) / 1000.0
        uuids = transformer_options.get("uuids")
        key = tuple(str(v) for v in uuids) if uuids else ("default",)
        context = self.contexts.setdefault(key, CacheContext())
        signature = self._input_signature(x)
        if context.input_signature != signature or (context.previous_sigma is not None and sigma > context.previous_sigma + 1e-7):
            context.clear_tensors()
            context.input_signature = signature
        context.previous_sigma = sigma
        context.audio_slice, context.video_slice, context.latent_frames = self._layout_info(minimax_payload)
        context.probe_input = None
        context.probe_output = None
        context.pending_probe_residual = None
        context.use_cache = False
        context.last_metrics = None
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
        context.probe_input = x.detach().clone()

    @torch.compiler.disable()
    def decide_after_probe(self, probe_output: torch.Tensor):
        context = self.current
        if context is None or context.probe_input is None:
            raise RuntimeError("H3BC probe state is incomplete")
        probe_residual = probe_output - context.probe_input
        previous = context.previous_probe_residual
        tail = context.tail_residual
        can_compare = previous is not None and tail is not None and previous.shape == probe_residual.shape and tail.shape == probe_output.shape
        reason = "full:no-cache"
        use_cache = False
        metrics = {
            "video": None, "audio": None, "temporal": None,
            "multiplier": 1.0, "normalized": None, "budget_before": context.budget_spent,
            "effective_video_threshold": None, "effective_audio_threshold": None,
        }

        if can_compare and self._within_window(context):
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
                video_diff, audio_diff, temporal_diff,
                video_threshold, audio_threshold, temporal_threshold,
                self.config.temporal_guard,
            )
            metrics.update({
                "video": video_diff, "audio": audio_diff, "temporal": temporal_diff,
                "multiplier": mult, "normalized": norm,
                "effective_video_threshold": video_threshold,
                "effective_audio_threshold": audio_threshold,
            })
            if norm is None:
                reason = "full:invalid-diff"
            elif norm > 1.0:
                reason = "full:guard"
            elif context.consecutive_hits >= self.config.max_consecutive_hits:
                reason = "full:max-hits"
            elif context.budget_spent + norm > self.config.error_budget_units:
                reason = "full:error-budget"
            else:
                use_cache = True
                reason = "cache"
        elif can_compare:
            reason = "full:outside-window"

        context.use_cache = use_cache
        context.last_metrics = metrics
        self.metrics.append(metrics)
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1

        if use_cache:
            context.consecutive_hits += 1
            context.budget_spent += float(metrics["normalized"] or 0.0)
            self.cached_step_numbers.append(self.full_steps + self.cached_steps + 1)
            context.probe_input = None
            context.probe_output = None
            context.pending_probe_residual = None
        else:
            context.consecutive_hits = 0
            context.budget_spent = 0.0
            context.probe_output = probe_output.detach().clone()
            context.pending_probe_residual = probe_residual.detach().clone()

        if self.verbose:
            vd = metrics["video"]
            ad = metrics["audio"]
            td = metrics["temporal"]
            nd = metrics["normalized"]
            logging.info(
                "[H3BC] %s sigma=%.5f v=%s a=%s t=%s norm=%s budget=%.3f/%.3f",
                reason, context.previous_sigma,
                "-" if vd is None else f"{vd:.5f}",
                "-" if ad is None else f"{ad:.5f}",
                "-" if td is None else f"{td:.5f}",
                "-" if nd is None else f"{nd:.3f}",
                context.budget_spent, self.config.error_budget_units,
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

    def finish_cached_step(self, probe_output: torch.Tensor):
        context = self.current
        if context is None or context.tail_residual is None:
            raise RuntimeError("H3BC has no cached tail residual")
        self.cached_steps += 1
        return probe_output + context.tail_residual

    def summary(self):
        steps = self.full_steps + self.cached_steps
        if not steps:
            return "no model steps"
        executed_blocks = self.full_steps * self.block_count + self.cached_steps * self.config.probe_blocks
        theoretical = steps * self.block_count / max(executed_blocks, 1)
        skipped_blocks = self.cached_steps * (self.block_count - self.config.probe_blocks)
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.reason_counts.items()))
        return (
            f"cached {self.cached_steps}/{steps} steps; probe={self.config.probe_blocks}/{self.block_count}; "
            f"skipped blocks={skipped_blocks}; estimated block-stack speedup {theoretical:.2f}x; "
            f"cache steps={self.cached_step_numbers}; {reasons}"
        )


def make_block_patch(cache: H3BCAdaptiveCache, index: int, last_index: int):
    probe_last = cache.config.probe_blocks - 1

    def patch(args, extra):
        original_block = extra["original_block"]

        if index == 0:
            cache.capture_probe_input(args["img"])

        if index <= probe_last:
            output = original_block(args)["img"]
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

        output = original_block(args)["img"]
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
            logging.info("H3BC: %s", cache.summary())
            cache.reset()
    return wrapper


class ApplyMiniMaxH3BC:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "mode": ([*PRESETS, CUSTOM_MODE], {"default": "H3BC Balanced α — 0.07"}),
            "threshold": ("FLOAT", {"default": 0.07, "min": 0.0, "max": 0.30, "step": 0.005}),
            "start_percent": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            "max_consecutive_hits": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
            "probe_blocks": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "dynamic_threshold": ("BOOLEAN", {"default": True}),
            "edge_ratio": ("FLOAT", {"default": 0.58, "min": 0.05, "max": 1.0, "step": 0.01}),
            "error_budget_units": ("FLOAT", {"default": 1.25, "min": 0.1, "max": 8.0, "step": 0.05}),
            "audio_guard_ratio": ("FLOAT", {"default": 0.80, "min": 0.05, "max": 2.0, "step": 0.05}),
            "temporal_guard": ("BOOLEAN", {"default": True}),
            "prefetch_mode": (PREFETCH_MODES, {"default": "inherit"}),
            "verbose": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MiniMax H3/optimization"
    DESCRIPTION = (
        "H3BC v2 adaptive native 20-step cache. Prefix probe + separate audio/video guards + "
        "dynamic edge-aware threshold + cumulative error budget. Alpha presets are intentionally uncalibrated."
    )

    def apply(self, model, mode, threshold, start_percent, end_percent, max_consecutive_hits,
              probe_blocks, dynamic_threshold, edge_ratio, error_budget_units, audio_guard_ratio,
              temporal_guard, prefetch_mode, verbose):
        if mode == CUSTOM_MODE:
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
            )
            label = (
                f"Custom threshold={threshold:.3f} window={start_percent:.2f}-{end_percent:.2f} "
                f"hits={max_consecutive_hits} probe={probe_blocks} dyn={dynamic_threshold} "
                f"edge={edge_ratio:.2f} budget={error_budget_units:.2f} audio={audio_guard_ratio:.2f} "
                f"temporal={temporal_guard} prefetch={prefetch_mode}"
            )
        else:
            preset = PRESETS[mode]
            config = preset.config
            label = f"{mode}; prefetch={prefetch_mode}"

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
            "version": "2.0.0-alpha1",
            "mode": mode,
            "probe_blocks": config.probe_blocks,
        }
        if prefetch_mode == "disable_dynamic_vbars":
            patched_options["prefetch_dynamic_vbars"] = False

        cache = H3BCAdaptiveCache(config, start_sigma, end_sigma, block_count, verbose)
        for index in range(block_count):
            patched.set_model_patch_replace(make_block_patch(cache, index, block_count - 1), "dit", "double_block", index)

        key = f"h3bc_v2_{id(cache)}"
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, key, make_diffusion_wrapper(cache))
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key, make_sample_wrapper(cache, label))
        return (patched,)


NODE_CLASS_MAPPINGS = {"ApplyMiniMaxH3BC": ApplyMiniMaxH3BC}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyMiniMaxH3BC": "MiniMax H3BC v2 Adaptive Cache (Alpha)"}
