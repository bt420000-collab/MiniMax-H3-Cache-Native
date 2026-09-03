# Examples

`H3BC_v2_node_fragment.json` records the alpha2 BALANCED node settings and intended MODEL wiring.

For GPU validation, use your existing native MiniMax H3 20-step workflow and insert H3BC immediately after the diffusion model loader:

```text
Load Diffusion Model
        |
        v
MiniMax H3BC Native Cache Engine (alpha2)
        |
        +--> Basic Scheduler
        +--> Basic Guider
```

Validation order:

1. OFF for the true native latency baseline.
2. LOSSLESS / REFERENCE for exact compute plus block profiling.
3. SAFE.
4. BALANCED.
5. AGGRESSIVE only after visual inspection.

Keep the stock sampler, scheduler, step count, prompt, seed, duration and resolution fixed for A/B tests.

A full public benchmark workflow will be added after the first real-GPU quality/performance calibration, so the repository does not canonize an unvalidated alpha workflow as a production reference.
