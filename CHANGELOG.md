# Changelog

## 2.0.0-alpha4

- Fixed critical in-place tensor aliasing in native MiniMax H3 probe snapshots and REFERENCE/debug profiler inputs.
- Freezes probe input and post-probe state before later H3 blocks mutate the live hidden tensor.
- Adds fail-closed zero/non-finite tail residual validation.
- Adds an in-place mutation regression test matching MiniMax H3 block semantics.
- Applies the first real-H3 20-step residual calibration: SAFE 0.09, BALANCED 0.10, AGGRESSIVE 0.12.
- Preserves frequent exact refresh for SAFE/BALANCED with warmup=4 and max_cached_run=1.
- Documents that H3BC relative-L1 thresholds are workload-specific and not numerically interchangeable with Cache-DiT thresholds.

## 2.0.0-alpha2

- Added explicit `OFF` mode that returns the incoming MODEL unchanged and installs no H3BC hooks.
- Added `LOSSLESS / REFERENCE` exact-compute mode with bounded per-block residual profiling.
- Added production-oriented `SAFE`, `BALANCED`, and `AGGRESSIVE` modes.
- Added first-class exact-refresh policy: `warmup_steps`, `max_cached_run`, periodic refresh, explicit refresh steps, and adaptive next-step refresh.
- Added external `h3bc_control` contract for `force_refresh`, `force_refresh_step(s)`, and nonce-based `reset_cache`.
- Kept task-boundary reset before and after each OUTER_SAMPLE execution.
- Added H3BC_DEBUG JSON telemetry export with per-step decisions and per-block profile data.
- Added cache decision-gate overhead accounting and debug-only estimated gross/net saved compute.
- Added low-memory sampled residual profiler; it does not retain full residual tensors for all 50 blocks.
- Retained legacy alpha1 preset strings so existing saved workflows remain loadable.
- Updated example workflow to the alpha2 BALANCED exact-refresh baseline.
- Expanded policy/static/mock tests for refresh and profiler contracts.

## 2.0.0-alpha1

- Rebuilt H3BC as an independent native ComfyUI plugin.
- Replaced input-signature cache decision with configurable real prefix neural probe.
- Added separate video and audio relative-L1 guards.
- Added max-frame temporal video guard.
- Added edge-aware dynamic threshold curve.
- Added normalized cumulative error budget.
- Added ComfyUI cache-conflict detection.
- Added optional dynamic-vbar prefetch disable A/B switch.
- Added unit/static validation and an updated I2V example workflow.
