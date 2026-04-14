# Failed Experiment 1: Plain Posterior-As-Prior Online VMP

## What Was Tested

The original BONG-like loop:

```text
take mini-batch
run VMP
use posterior as next prior
repeat
```

In distribution form:

```text
q_next(w) approx project[q_current(w) * likelihood(batch)]
```

This was tested with random mini-batches, different batch sizes, and different numbers of VMP iterations per mini-batch.

## Historical Reproduction Note

The auxiliary sweep script was removed from `test_dataset` after cleanup. The recorded outputs from the run are listed below if the local artifacts are still present.

Historical output:

```text
test_dataset/viz/online_sweep_batch_iter/summary.csv
test_dataset/viz/online_sweep_batch_iter/final_mse_by_batch_and_iterations.png
test_dataset/viz/online_sweep_batch_iter/best_mse_by_batch_and_iterations.png
test_dataset/viz/online_sweep_batch_iter/tail20_mean_mse_by_batch_and_iterations.png
test_dataset/viz/online_sweep_batch_iter/min_var_w_a_by_batch_and_iterations.png
```

## Important Results

Best transient result in that sweep:

```text
batch_size = 16
batch_iterations = 5
best MSE = 0.181799 at update 1
final MSE = 0.221025
tail-20 mean MSE = 0.222191
```

Best final result in that sweep:

```text
batch_size = 16
batch_iterations = 2
final MSE = 0.221012
best MSE = 0.217965 at update 8
tail-20 mean MSE = 0.222243
```

Baseline:

```text
constant 0.52 baseline MSE = 0.221875
```

## Interpretation

More inner VMP iterations help discover nonlinear structure early.

But the model jumps out of the good nonlinear solution and drifts back toward baseline.

The failure pattern:

```text
best early MSE can be good
final/tail MSE returns near baseline
gate structure becomes broad / weak / nearly linear
```

So this experiment showed:

```text
online VMP can find nonlinear structure,
but posterior-as-prior replacement does not retain it.
```
