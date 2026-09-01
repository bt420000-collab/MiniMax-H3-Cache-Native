# Examples

`H3BC_v2_node_fragment.json` records the alpha1 H3BC node settings and intended MODEL wiring.

For GPU validation, use your existing native MiniMax H3 workflow and insert H3BC immediately after the diffusion model loader:

```text
Load Diffusion Model
        |
        v
MiniMax H3BC v2 Adaptive Cache (Alpha)
        |
        +--> Basic Scheduler
        +--> Basic Guider
```

Keep the stock sampler, scheduler, step count, prompt, seed, duration and resolution fixed for A/B tests.

A full benchmark workflow will be added after the first real-GPU quality/performance calibration, so the public repository does not canonize an unvalidated alpha workflow as a production reference.
