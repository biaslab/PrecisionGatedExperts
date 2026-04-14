# Failed Experiment 2: Damping And SVI-Style Scaling Without Site Memory

## What Was Tested

These experiments tried to fix the posterior-as-prior update without keeping per-batch sites.

Tested mechanisms:

1. Natural-parameter damping:

```text
eta_next = eta_old + alpha * (eta_post - eta_old)
```

2. Separate damping for output weights and gate weights:

```text
alpha_mean
alpha_gate
```

3. SVI-style mini-batch likelihood scaling:

```text
obs_precision_candidate = obs_precision * n_train / batch_size
```

4. Optional tempering:

```text
obs_precision_candidate = obs_precision * beta * n_train / batch_size
```

5. Candidate posterior from current prior vs original initial prior.

## Historical Damping Sweep

The auxiliary damping script was removed from `test_dataset` after cleanup. Historical output:

```text
test_dataset/viz/online_sweep_damping/summary.csv
```

Best damping-only row:

```text
batch_size = 16
batch_iterations = 5
alpha_mean = 0.5
alpha_gate = 0.05
rho = 1.0
final MSE = 0.220891
best MSE = 0.186889 at update 1
tail-20 mean MSE = 0.221015
```

This is only a small improvement. It still does not create a strong stable online nonlinear model.

## Historical Current-Prior SVI Scaling Sweep

The auxiliary SVI scaling script was removed from `test_dataset` after cleanup. Historical output:

```text
test_dataset/viz/online_sweep_svi_scaling/summary.csv
```

Best current-prior SVI-style row:

```text
batch_size = 16
batch_iterations = 5
alpha_mean = 0.5
alpha_gate = 0.05
beta = 1.0
likelihood_scale = 30.0
final MSE = 0.220804
best MSE = 0.185580 at update 1
tail-20 mean MSE = 0.220919
```

This is mildly better than damping alone, but still weak compared with site memory.

## Historical Initial-Prior SVI Candidate Sweep

The auxiliary initial-prior SVI script was removed from `test_dataset` after cleanup. Historical output:

```text
test_dataset/viz/online_sweep_svi_initial/summary.csv
```

Best-looking final row:

```text
batch_size = 32
beta = 0.1
likelihood_scale = 1.5
final MSE = 0.220342
best MSE = 0.204798 at update 3
tail-20 mean MSE = 0.225849
```

It preserved some useful early structure but was bad in the tail.

## Interpretation

Damping helps a little because it prevents each mini-batch from fully replacing the posterior.

SVI-style scaling is conceptually nicer for random mini-batches, but it still does not solve the main problem here.

Why it failed:

```text
there is still only one global posterior
mini-batches still do not own persistent contributions
repeated passes do not revise specific old evidence
gate structure can still be washed out
```

Conclusion:

```text
damping/scaling alone is not enough;
the important missing piece is site memory.
```
