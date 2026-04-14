# Online VMP / BONG Experiment Notes

This folder summarizes the XOR online VMP experiments.

The most important result is in:

- `how_site_memory_works.md`

The failed or weaker experiments are documented in:

- `failed_bong/01_posterior_as_prior.md`
- `failed_bong/02_damping_and_svi_scaling.md`
- `failed_bong/03_constant_site_memory.md`

Only the working scheduled site-memory Julia script is kept under `test_dataset/`. The failed/auxiliary scripts were removed from `test_dataset` after the experiments; their docs keep the observed results for later review.

## Best Current Experiment

Run from the repository root:

```bash
julia --project=. test_dataset/xor_online_site_memory_schedule_sweep.jl
```

Best setting:

```text
method = fixed-batch site memory with cavity updates
schedule = alpha_t = alpha_0 / sqrt(epoch)
batch_size = 24
batch_iterations = 5
epochs = 10
obs_precision = 1e3
alpha_mean_initial = 0.5
alpha_gate_initial = 0.05
projection_iterations = 1
```

Observed result:

```text
Final MSE = 0.197937
Best MSE = 0.197937 at update 200
Tail-20 mean MSE = 0.198223
Baseline MSE = 0.221875
```

The same script also reruns the best row at the end and writes the best heatmap.

Heatmap:

```text
test_dataset/viz/online_sweep_site_memory_schedule/best_sqrt_site_b24_amean0p5_agate0p05_heatmap.png
```
