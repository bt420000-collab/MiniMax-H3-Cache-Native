# H3BC — MiniMax H3 Block Cache Native

**Version:** 2.0.0-alpha1  
**Status:** engineering alpha, not yet quality-calibrated

H3BC is a training-free inference cache for **native MiniMax H3** in ComfyUI. It keeps the original H3 model, sampler, scheduler and step count intact. It does **not** use a Turbo/PDD/Distillation LoRA and does not monkey-patch `MiniMaxH3Model._forward`.

The intended production target is **Stock H3 20-step with less repeated DiT work**.

## What changed from H3BC v0.1

v0.1 used a lightweight input signature and cached the whole DiT-stack residual. v2 replaces the decision path with a real neural probe:

1. Run the first `probe_blocks` H3 transformer blocks normally.
2. Measure how the current probe residual changed relative to the last real step.
3. Evaluate **video**, **audio**, and optional **max-frame temporal** change separately.
4. Tighten the threshold near both ends of the cache window and relax it near the middle.
5. Track a normalized cumulative **error budget** in addition to the hard consecutive-hit limit.
6. On a cache hit, reuse the residual of the remaining transformer stack.

Default alpha presets currently use one real probe block and cache blocks 1–49 when the guards allow it.

## Why H3-specific AV guards

MiniMax H3 is a packed audio-video DiT. A single global mean can hide a local motion change or an audio change. H3BC v2 therefore computes separate relative-L1 checks for the packed **video** and **audio** target ranges. With `temporal_guard=true`, the most-changed latent video frame also participates in the gate.

A cache hit requires every enabled guard to remain within its effective threshold.

## Dynamic threshold

When enabled, the threshold multiplier follows a smooth middle-heavy curve across the active cache window:

- window edges: `threshold × edge_ratio`
- middle: `threshold × 1.0`

This keeps early/late denoising more conservative without giving up the redundant middle region.

## Error budget

A cache hit consumes:

`max(video_ratio, audio_ratio, temporal_ratio)`

where each ratio is `observed_diff / effective_threshold`.

The cache is forced back to a full H3 step if the accumulated budget would exceed `error_budget_units`, even if the current individual hit is still under threshold. A real step resets the budget.

This is deliberately stricter than a simple `max_consecutive_hits` counter.

## Presets (alpha, not calibrated)

| Preset | Base threshold | Intent |
|---|---:|---|
| H3BC Safe α | 0.05 | first fidelity baseline |
| H3BC Balanced α | 0.07 | default engineering test |
| H3BC Fast α | 0.09 | exploratory speed test |

These values are **not claimed as production quality settings yet**. Fixed-seed A/B testing against uncached Stock H3 20-step is required.

## Connection

```text
Load Diffusion Model
        │
        ▼
MiniMax H3BC v2 Adaptive Cache (Alpha)
        │
        ├──► Basic Scheduler
        └──► Basic Guider
```

Connect the patched MODEL everywhere the unpatched H3 MODEL was previously used.

Do not stack H3BC with FirstBlockCache, CacheDiT, EasyCache/LazyCache, T8 Block Cache, or another `double_block` replacement.

## Prefetch mode

Current ComfyUI MiniMax H3 creates its block prefetch queue before the block loop. A skipped block can therefore avoid compute while still participating in dynamic weight prefetch.

`prefetch_mode=inherit` keeps ComfyUI behavior unchanged and is the default.

`prefetch_mode=disable_dynamic_vbars` sets `prefetch_dynamic_vbars=False` on the patched model. This can help low-VRAM/cache-heavy runs, but can also make full steps slower. Treat it as an A/B switch, not a universally faster setting.

## First validation matrix

Use the same model, prompt, seed, duration, resolution, sampler and 20-step scheduler for every run:

1. Stock H3 20-step, no cache.
2. H3BC Safe α, prefetch inherit.
3. H3BC Balanced α, prefetch inherit.
4. H3BC Fast α, prefetch inherit.
5. Best quality candidate again with `disable_dynamic_vbars`.

Record:

- total wall-clock time
- H3BC cached/full steps
- cache step indices
- per-step video/audio/temporal diffs
- final video visual difference
- motion/face/hand/detail stability
- speech/audio timing and timbre

## Current limitations

- This is an approximation. Cached and uncached runs follow different numerical trajectories.
- Alpha presets have only static/unit validation here; no GPU H3 quality benchmark is claimed.
- Current v2 alpha reuses the full tail residual after the probe prefix. A future v2.1 may test a real tail-refinement segment, but that is intentionally not mixed into this first validation build.
- Dynamic ComfyUI weight prefetch cannot be perfectly skipped per cache decision without a deeper block-loop integration; this build avoids core monkey-patching.

## Compatibility design

H3BC uses ComfyUI's public model patch mechanisms:

- `set_model_patch_replace(..., "dit", "double_block", index)`
- `WrappersMP.DIFFUSION_MODEL`
- `WrappersMP.OUTER_SAMPLE`

It does not replace MiniMax H3 forward code.

## Research / implementation references

The design space was informed by:

- the original `lihaoyun6/ComfyUI-MiniMaxH3-Cache`
- `duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache`
- Hugging Face Diffusers FirstBlockCache
- ComfyUI native MiniMax H3 block replacement hooks

H3BC v2 code in this package is an independent implementation.

## License

MIT. See `LICENSE`.
