"""H3BC alpha4 runtime correctness/calibration layer.

This module keeps the public alpha2 node API stable while repairing MiniMax H3's
in-place hidden-state aliasing semantics and applying the first empirically
calibrated native-H3 thresholds. It intentionally touches only H3BC's cloned
MODEL/block hooks.
"""
from __future__ import annotations

import logging
import math


def apply_alpha4_runtime(n):
    """Apply alpha4 fixes to the loaded ``nodes`` module in-place."""
    if getattr(n, "VERSION", None) == "2.0.0-alpha4":
        return n

    n.VERSION = "2.0.0-alpha4"

    # First empirical calibration from a real native H3 20-step run. H3BC's
    # relative-L1 metric is not numerically interchangeable with Cache-DiT's.
    n.PRESETS[n.MODE_SAFE] = n.Preset(
        n.H3BCConfig(
            threshold=0.090,
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
    )
    n.PRESETS[n.MODE_BALANCED] = n.Preset(
        n.H3BCConfig(
            threshold=0.100,
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
    )
    n.PRESETS[n.MODE_AGGRESSIVE] = n.Preset(
        n.H3BCConfig(
            threshold=0.120,
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
    )

    def capture_probe_input(self, x):
        context = self.current
        if context is None:
            raise RuntimeError("H3BC probe input captured outside model execution")
        # MiniMax H3 DiTBlock.forward mutates hidden state in-place. detach() alone
        # aliases the live tensor and collapses probe residuals to zero.
        context.probe_input = x.detach().clone()

    def finish_full_step(self, output):
        context = self.current
        if context is None or context.probe_output is None or context.pending_probe_residual is None:
            raise RuntimeError("H3BC full-step state is incomplete")
        tail_residual = (output - context.probe_output).detach().clone()
        tail_mean = float(tail_residual.float().abs().mean().item())
        if (not math.isfinite(tail_mean)) or tail_mean <= 1e-12:
            # Fail closed. A real 49-block H3 tail must not be exactly zero.
            context.tail_residual = None
            logging.error(
                "[H3BC] invalid tail residual mean=%s; cache disabled until a valid exact tail is captured",
                tail_mean,
            )
        else:
            context.tail_residual = tail_residual
        context.previous_probe_residual = context.pending_probe_residual
        context.probe_input = None
        context.probe_output = None
        context.pending_probe_residual = None
        self.full_steps += 1

    def make_block_patch(cache, index: int, last_index: int):
        probe_last = cache.config.probe_blocks - 1

        def patch(args, extra):
            original_block = extra["original_block"]
            block_input = args["img"]

            if cache.reference_mode:
                # Profiler must snapshot before the in-place block mutates input.
                block_input_snapshot = block_input.detach().clone()
                timing = cache.profiler.begin_timing(block_input)
                output = original_block(args)["img"]
                cache.profiler.observe_block(
                    context=cache.current,
                    block_index=index,
                    block_input=block_input_snapshot,
                    block_output=output,
                    timing=timing,
                )
                if index == last_index:
                    cache.finish_reference_step()
                return {"img": output}

            if index == 0:
                cache.capture_probe_input(block_input)

            if index <= probe_last:
                block_input_snapshot = block_input.detach().clone() if cache.debug else None
                timing = cache.profiler.begin_timing(block_input) if cache.debug else (None, None, None)
                output = original_block(args)["img"]
                if cache.debug:
                    cache.profiler.observe_block(
                        context=cache.current,
                        block_index=index,
                        block_input=block_input_snapshot,
                        block_output=output,
                        timing=timing,
                    )
                if index == probe_last:
                    cache.decide_after_probe(output)
                    context = cache.current
                    # alpha2 decide_after_probe stores probe_output with detach().
                    # Freeze it immediately before Block 1..N mutate the live tensor.
                    if context is not None and not context.use_cache and context.probe_output is not None:
                        context.probe_output = context.probe_output.detach().clone()
                return {"img": output}

            context = cache.current
            if context is None:
                raise RuntimeError("H3BC has no active context")

            if context.use_cache:
                if index == last_index:
                    return {"img": cache.finish_cached_step(args["img"])}
                return {"img": args["img"]}

            block_input_snapshot = block_input.detach().clone() if cache.debug else None
            timing = cache.profiler.begin_timing(block_input) if cache.debug else (None, None, None)
            output = original_block(args)["img"]
            if cache.debug:
                cache.profiler.observe_block(
                    context=context,
                    block_index=index,
                    block_input=block_input_snapshot,
                    block_output=output,
                    timing=timing,
                )
            if index == last_index:
                cache.finish_full_step(output)
            return {"img": output}

        return patch

    n.H3BCAdaptiveCache.capture_probe_input = capture_probe_input
    n.H3BCAdaptiveCache.finish_full_step = finish_full_step
    n.make_block_patch = make_block_patch

    # Keep the saved-workflow API, but make the visible Custom threshold default
    # match the current BALANCED calibration.
    original_input_types = n.ApplyMiniMaxH3BC.INPUT_TYPES.__func__

    @classmethod
    def input_types(cls):
        spec = original_input_types(cls)
        old = spec["required"]["threshold"]
        opts = dict(old[1])
        opts["default"] = 0.10
        spec["required"]["threshold"] = (old[0], opts)
        return spec

    n.ApplyMiniMaxH3BC.INPUT_TYPES = input_types
    n.ApplyMiniMaxH3BC.DESCRIPTION = (
        "H3BC native H3 cache engine. alpha4 repairs in-place H3 tensor aliasing, "
        "keeps exact-refresh/reference profiling, and applies the first real-H3 residual calibration. "
        "Set H3BC_DEBUG=1 for detailed JSON telemetry."
    )
    n.NODE_DISPLAY_NAME_MAPPINGS["ApplyMiniMaxH3BC"] = "MiniMax H3BC Native Cache Engine (α4)"
    return n
