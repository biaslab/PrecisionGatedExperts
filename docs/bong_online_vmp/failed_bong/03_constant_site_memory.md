# Failed / Partial Experiment 3: Site Memory Without A Schedule

## What Was Tested

Fixed-batch site memory with cavity updates:

```text
q_global(w) proportional_to p0(w) * product_b t_b(w)
```

When revisiting batch `b`:

```text
q_cavity(w) proportional_to q_global(w) / t_b(w)
```

Then:

```text
run VMP with q_cavity and batch b
replace/refine site b
rebuild q_global
```

This was the first method that actually retained nonlinear structure.

However, constant site damping over too many epochs was unstable.

## Historical Initial Site-Memory Sweep

The auxiliary constant-site-memory script was removed from `test_dataset` after cleanup. Historical output:

```text
test_dataset/viz/online_sweep_site_memory/summary.csv
```

Best initial site-memory row:

```text
batch_size = 32
batch_iterations = 5
epochs = 5
alpha_mean = 0.5
alpha_gate = 0.05
obs_precision = 1e3
final MSE = 0.203278
best MSE = 0.202756 at update 2
tail-20 mean MSE = 0.204180
```

This was a strong positive result, not a failure.

The failure was full site replacement:

```text
alpha_mean = 1.0
alpha_gate = 1.0
```

It found good early points but then exploded numerically.

## Historical Refinement Sweep

The auxiliary refinement script was removed from `test_dataset` after cleanup. Historical output:

```text
test_dataset/viz/online_sweep_site_memory_refine/summary.csv
```

Best 5-epoch constant-damping row:

```text
batch_size = 24
batch_iterations = 5
epochs = 5
alpha_mean = 0.5
alpha_gate = 0.05
final MSE = 0.199848
best MSE = 0.198038 at update 86
tail-20 mean MSE = 0.199501
```

Heatmap:

```text
test_dataset/viz/online_sweep_site_memory_refine/best_site_b24_amean0p5_agate0p05_heatmap.png
```

## Historical Long Constant-Damping Failure

The auxiliary long-check script was removed from `test_dataset` after cleanup. Historical output:

```text
test_dataset/viz/online_sweep_site_memory_long/summary.csv
```

10-epoch constant-damping results:

```text
batch_size = 24
batch_iterations = 5
alpha_mean = 0.5
alpha_gate = 0.05
final MSE = 27.690129
best MSE = 0.198038 at update 86
tail-20 mean MSE = 20.443908

batch_size = 32
batch_iterations = 8
alpha_mean = 0.5
alpha_gate = 0.05
final MSE = 8.888544
best MSE = 0.198976 at update 2
tail-20 mean MSE = 5.363628
```

## Interpretation

Site memory is the key structural improvement.

But constant site damping keeps applying large structural updates after the nonlinear solution is already present.

Failure pattern:

```text
good solution appears
continued constant site updates destabilize it
final predictions can explode
```

This led to the successful scheduled site-memory experiment:

```text
alpha_t = alpha_0 / sqrt(epoch)
```

That schedule preserves the benefit of site memory while avoiding late constant-update blow-up.
