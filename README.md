# H3BC - MiniMax H3 Cache Native

**Version:** 2.0.0-alpha2  
**Status:** engineering alpha, production-safety/profiling phase

H3BC is a lightweight, training-free runtime cache for **native MiniMax H3** in ComfyUI. It preserves the original H3 model, scheduler, sampler contract and denoising step count. It is not FastH3, PDD, Turbo LoRA, or a distilled low-step model.

The current target is:

> **Stock MiniMax H3 20-step + conservative residual cache + frequent exact refresh.**

H3BC is intentionally independent from H3VM, Compute PM, PDD, FastH3, and production workflow plugins.

## alpha2 priorities

alpha2 moves H3BC from an experimental cache node toward a measurable cache engine:

- explicit **OFF** mode with no hooks at all;
- **LOSSLESS / REFERENCE** mode: exact H3 compute plus block profiler;
- production-oriented **SAFE / BALANCED / AGGRESSIVE** policies;
- first-class **Exact Refresh Policy**;
- `warmup_steps`;
- `max_cached_run` (the existing `max_consecutive_hits` widget is retained as the compatible UI name);
- `force_refresh_every`;
- explicit `force_refresh_steps`;
- adaptive next-step refresh;
- external `force_refresh` / `reset_cache` control contract;
- task-boundary cache reset;
- block residual profiling with bounded sampled state;
- cache decision overhead and estimated net saving telemetry;
- `H3BC_DEBUG=1` JSON telemetry export.

## Current data path

```text
Native H3 MODEL
      |
      v
H3BC MODEL wrapper
      |
      +-- OFF
      |     +-- return original MODEL unchanged
      |
      +-- LOSSLESS / REFERENCE
      |     +-- every block EXACT
      |          +-- sampled residual profiler
      |
      +-- SAFE / BALANCED / AGGRESSIVE
            |
            v
       Block 0..probe N EXACT
            |
            +-- warmup / forced refresh? -----> EXACT tail
            +-- AV / temporal guard fail? ----> EXACT tail
            +-- error budget exceeded? -------> EXACT tail
            +-- max_cached_run reached? ------> EXACT tail
            +-- confidence high --------------> reuse cached tail residual
```

The tail cache is still whole-tail residual reuse after the real probe prefix. Per-block selective cache comes later, after profiling tells us which H3 blocks are actually worth caching.

## Modes

### OFF

Returns the incoming MODEL object directly. No clone, wrapper, block replacement, cache state, or telemetry is installed.

Use this as the true latency baseline.

### LOSSLESS / REFERENCE

Runs every H3 block exactly and records sampled step-to-step block residual behavior. It does not skip any block.

This mode is intended for profiling and quality reference. Profiling adds overhead, so do **not** use its wall-clock time as the native speed baseline. Use OFF for latency comparison.

### SAFE

Current starting profile:

```text
warmup_steps = 4
threshold = 0.030
max_cached_run = 1
probe_blocks = 1
adaptive_refresh = true
```

### BALANCED

Current production-baseline candidate:

```text
warmup_steps = 4
threshold = 0.040
max_cached_run = 1
probe_blocks = 1
adaptive_refresh = true
```

This intentionally follows the conservative production pattern: one cache opportunity, then exact re-anchor.

### AGGRESSIVE

Experimental only:

```text
warmup_steps = 3
threshold = 0.070
max_cached_run = 2
probe_blocks = 1
adaptive_refresh = true
```

Quality loss is possible and expected to be workload dependent.

The old alpha1 preset strings remain accepted so existing saved workflows do not immediately break.

## Exact Refresh Policy

Cache admission is no longer controlled by threshold alone.

For every denoising call:

```text
Step N
  |
  +-- warmup? -------------------------- EXACT
  +-- external force refresh? ---------- EXACT
  +-- explicit force_refresh_steps? ---- EXACT
  +-- force_refresh_every interval? ---- EXACT
  +-- adaptive next refresh? ----------- EXACT
  +-- cached_run >= max_cached_run? ---- EXACT
  +-- guard/error budget fail? --------- EXACT
  +-- otherwise ------------------------ CACHE
```

Any exact step refreshes the probe anchor and tail residual.

## H3-specific cache gate

H3BC uses a real prefix neural probe. After the probe prefix it compares the current probe residual with the last exact probe residual.

The gate treats packed H3 modalities separately:

- video relative-L1;
- audio relative-L1;
- optional max-frame temporal video relative-L1.

A cache hit requires every enabled guard to stay inside its effective threshold.

The threshold can be tightened near the start/end of the active denoising window and relaxed in the middle.

## Reference Block Profiler

The profiler answers the next engineering question: **which H3 blocks are both expensive and step-to-step redundant?**

For each exact block it measures a deterministic sampled block residual:

```text
residual = block_output - block_input
```

and records, when available:

- step index;
- block index;
- full sampled residual diff;
- video sampled residual diff;
- audio sampled residual diff;
- residual norm;
- block compute timing.

To keep VRAM bounded, it stores only small deterministic residual samples per block, not all 50 full block residual tensors.

At task end REFERENCE mode logs the most stable blocks and most expensive blocks.

## Telemetry

Normal runs log step-level cache decisions and a task summary.

Set:

```text
H3BC_DEBUG=1
```

before starting ComfyUI to enable detailed JSON telemetry. Files are written under:

```text
ComfyUI/output/h3bc/
```

A summary includes:

```text
steps
blocks
exact_calls
cache_hits
hit_rate
cached_steps
forced_refresh
decision_gate_ms
cache_apply_cpu_ms
gross_compute_saved_ms_est
net_saved_ms_est
speedup_est
reason counts
per-block profile
```

The estimated saved time is only produced when block timing data exists. H3BC deliberately does not pretend that skipped-block count equals real wall-clock speedup.

## External refresh/reset contract

H3BC reserves this standard transformer option:

```python
transformer_options["h3bc_control"] = {
    "force_refresh": False,
    "force_refresh_step": 8,       # or force_refresh_steps=[8, 12]
    "reset_cache": "shot-42",     # use a changing token/nonce
    "cache_policy": "balanced",   # reserved for future external policy control
}
```

`reset_cache` is treated as a token so a persistent control dictionary does not accidentally reset every denoising step.

The OUTER_SAMPLE wrapper also resets cache state before and after every generation task, so cache state does not leak across shots by default.

## Memory behavior

H3BC does not cache every block output.

The acceleration path currently keeps only the last exact probe residual plus last exact tail residual, with temporary probe state during an exact step. The block profiler keeps small sampled residuals instead of full-size per-block tensors.

Future work will evaluate selected-block cache, cache dtype compression, and pinned-CPU spill only after real profiling shows that they are useful.

## Connection

```text
Load Diffusion Model
        |
        v
MiniMax H3BC Native Cache Engine (alpha2)
        |
        +---> Basic Scheduler
        +---> Basic Guider
```

Do not stack H3BC with another H3 `double_block` cache/replacement node.

## Compatibility / failure isolation

H3BC uses ComfyUI model-patch mechanisms:

- `set_model_patch_replace(..., "dit", "double_block", index)`;
- `WrappersMP.DIFFUSION_MODEL`;
- `WrappersMP.OUTER_SAMPLE`.

It does not replace `MiniMaxH3Model._forward` and does not modify H3VM or any production plugin.

Deleting the H3BC custom-node directory restores the original system path.

## Validation order

First production validation should use identical model/prompt/seed/resolution/frames/sampler/scheduler:

1. OFF, true native runtime baseline.
2. LOSSLESS / REFERENCE, exact output + block profile.
3. SAFE.
4. BALANCED.
5. AGGRESSIVE only after SAFE/BALANCED are visually inspected.

Evaluate not just speed but:

- face identity/drift;
- hands;
- fast motion;
- camera motion;
- fine texture;
- small text/logo;
- temporal wobble;
- ghosting;
- color drift;
- lip/expression timing;
- audio/video temporal coupling.

## Current limitations

- H3BC is still an approximation when cache hits occur.
- alpha2 production presets are starting points, not universal safety guarantees.
- Per-block selective cache is **not enabled yet**. First we profile real H3 block redundancy/cost.
- Attention-vs-MLP sub-residual profiling is not implemented yet; alpha2 profiles full H3 block residuals first.
- Dynamic ComfyUI weight prefetch cannot be perfectly avoided for a cache hit without deeper core integration; `disable_dynamic_vbars` remains an A/B switch.
- `decision_gate_ms` includes the synchronization needed to make the runtime cache decision, because that cost is real and must not be hidden.

## Research references

The design space is informed by:

- native ComfyUI MiniMax H3 block replacement hooks;
- FirstBlockCache-style real prefix probing;
- Cache-DiT / DBCache exact-refresh strategy;
- vLLM-Omni MiniMax H3 conservative Cache-DiT serving work;
- the original MiniMax H3 residual cache implementations.

H3BC remains a lightweight H3-specific independent implementation and does not depend on vLLM-Omni or Cache-DiT at runtime.

## License

MIT. See `LICENSE`.
