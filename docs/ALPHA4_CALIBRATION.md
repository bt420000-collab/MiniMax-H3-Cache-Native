# H3BC alpha4 calibration note

The first real native MiniMax H3 20-step validation after the in-place snapshot fix showed a clear U-shaped first-block residual curve.

Observed temporal-guard values on the calibration clip were approximately:

```text
step 05  0.10889
step 08  0.07959
step 10  0.07471
step 11  0.07422
step 12  0.07471
step 14  0.08154
step 16  0.10059
step 20  0.24316
```

The previous BALANCED 0.04 profile produced zero cache hits. This does not imply the vLLM Cache-DiT 0.04 serving profile is wrong; it means H3BC's first-block relative-L1 metric has a different numeric scale.

alpha4 therefore starts with:

```text
SAFE       0.09, warmup 4, max_cached_run 1
BALANCED   0.10, warmup 4, max_cached_run 1
AGGRESSIVE 0.12, warmup 3, max_cached_run 2
```

On the calibration trace, the intent is roughly one SAFE middle-step opportunity, around three alternating BALANCED opportunities, and a wider AGGRESSIVE middle band. Exact hit locations remain workload-dependent.

Do not treat these thresholds as universal. Revalidate across quantization, resolution, conditioning type, prompt/motion class, and hardware.
