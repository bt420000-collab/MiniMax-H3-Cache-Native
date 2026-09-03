# H3BC Architecture Notes

## Layer boundary

H3BC owns **single-model single-worker runtime cache only**.

It does not own:

- multi-GPU model partitioning;
- worker scheduling;
- PDD/Turbo/FastH3 model changes;
- production workflow orchestration.

## alpha1 -> alpha2 gap closed

alpha1 already had:

- native ComfyUI `double_block` replacement;
- real prefix probe;
- AV/temporal guards;
- dynamic threshold;
- cumulative error budget;
- task-level reset via OUTER_SAMPLE.

alpha1 did not have production-grade refresh semantics or a real reference profiler. alpha2 adds:

- explicit OFF path;
- exact REFERENCE profiler path;
- warmup exact steps;
- mandatory max-cached-run refresh;
- periodic/scheduled refresh;
- adaptive next-step refresh;
- external refresh/reset contract;
- sampled per-block residual profiling;
- cache overhead accounting.

## Current cache hook data flow

```text
DIFFUSION_MODEL wrapper
  begin_call()
    - identify context / UUID
    - reset on shape or sigma restart
    - apply external reset/refresh signals
    - advance 1-based denoise step

block 0..probe_last
  original block EXACT
  -> probe residual
  -> exact-refresh policy
  -> AV / temporal diff gate
  -> error budget gate

if CACHE
  blocks probe..48 bypass
  block 49 returns probe_output + cached_tail_residual

if EXACT
  blocks probe..49 original compute
  -> cache new tail residual
  -> cache new probe residual anchor

OUTER_SAMPLE wrapper
  reset before task
  execute task
  print/export telemetry
  reset after task
```

## First technical target after alpha2

Use REFERENCE telemetry to rank blocks by two axes:

1. mean step-to-step residual change;
2. mean block compute cost.

The first selective-cache candidates should be blocks that are both **expensive and stable**. No per-block cache policy should be enabled before this evidence exists.
