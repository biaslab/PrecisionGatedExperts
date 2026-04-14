# How The Working Online VMP Update Works

## Short Version

The best current online method is not plain posterior-as-prior filtering.

It uses:

```text
fixed-batch site memory
cavity updates
damped site replacement
sqrt-epoch damping decay
```

The best script is:

```text
test_dataset/xor_online_site_memory_schedule_sweep.jl
```

The scheduled sweep script also writes the best heatmap at the end.

## Why Plain Online VMP Failed

The earlier online update was:

```text
take mini-batch
run VMP
replace prior by posterior
repeat
```

In distribution form:

```text
q_next(w) approx project[q_current(w) * likelihood(batch)]
```

This is reasonable for one-pass filtering over new data. It is a bad match for repeated random mini-batches over the same dataset, because revisiting the same data behaves like adding it again as new evidence.

That can:

- double-count repeated mini-batches
- make the posterior too confident
- let each local mini-batch rewrite the global gate posterior
- wash out nonlinear gate specialization
- drift back toward broad always-on gates and nearly linear predictions

This is different from neural-network SGD. SGD revisits data to optimize the same finite-data objective. Posterior-as-prior filtering revisits data as if it were fresh evidence unless we correct for it.

## Site Memory

Site memory represents the global posterior as a product:

```text
q_global(w) proportional_to p0(w) * t1(w) * t2(w) * ... * tB(w)
```

where:

```text
p0(w) = original prior
tb(w) = approximate site factor for batch b
B = number of fixed batches
```

In exponential-family natural parameters, that product becomes a sum:

```text
eta_global = eta_prior + eta_site_1 + eta_site_2 + ... + eta_site_B
```

For the Gaussian weighted-mean precision form used here:

```text
q(w) proportional_to exp(xi' w - 1/2 w' Lambda w)
```

the stored natural terms are:

```text
xi
Lambda
```

Multiplying Gaussian-form factors corresponds to:

```text
xi_global = xi_prior + sum_b xi_site_b
Lambda_global = Lambda_prior + sum_b Lambda_site_b
```

## Cavity Update

When revisiting batch `b`, the algorithm removes that batch's old contribution:

```text
q_cavity(w) proportional_to q_global(w) / tb(w)
```

In natural parameters:

```text
eta_cavity = eta_global - eta_site_b
```

Then it runs VMP on that batch using the cavity as the prior:

```text
q_post_b(w) approx VMP(prior = q_cavity, likelihood = batch b)
```

The proposed new site is:

```text
t_new_b(w) proportional_to q_post_b(w) / q_cavity(w)
```

In natural parameters:

```text
eta_site_new_b = eta_post_b - eta_cavity
```

Then the global posterior is rebuilt:

```text
q_global(w) proportional_to q_cavity(w) * t_b(w)
```

or:

```text
eta_global = eta_cavity + eta_site_b
```

## Damped Site Replacement

Full site replacement was too aggressive and numerically unstable:

```text
eta_site_b = eta_site_new_b
```

The working update damps the site replacement:

```text
eta_site_b <- (1 - alpha_t) * eta_site_b + alpha_t * eta_site_new_b
```

There are separate step sizes:

```text
alpha_mean
alpha_gate
```

The best base values so far are:

```text
alpha_mean_initial = 0.5
alpha_gate_initial = 0.05
```

The gate step is smaller because `w_a` controls the nonlinear partition. If every mini-batch rewrites gates too strongly, the model loses the nonlinear structure.

## Square-Root Schedule

Constant damping worked for 5 epochs, but became unstable at 10 epochs.

The best schedule so far is:

```text
alpha_t = alpha_0 / sqrt(epoch)
```

This gives:

- larger early steps to discover nonlinear structure
- smaller later steps to consolidate without destroying it

The best current setting:

```text
batch_size = 24
batch_iterations = 5
epochs = 10
obs_precision = 1e3
alpha_mean_initial = 0.5
alpha_gate_initial = 0.05
schedule = sqrt_epoch
projection_iterations = 1
```

Result:

```text
Final MSE = 0.197937
Best MSE = 0.197937 at update 200
Tail-20 mean MSE = 0.198223
Baseline MSE = 0.221875
```

The key diagnostic is that the best MSE occurs at the final update. This means the method is not just finding a good early checkpoint and jumping out.

## Reproduce

Run the scheduled sweep:

```bash
julia --project=. test_dataset/xor_online_site_memory_schedule_sweep.jl
```

Expected best row:

```text
schedule = sqrt_epoch
batch_size = 24
batch_iterations = 5
alpha_mean = 0.5
alpha_gate = 0.05
final_mse = 0.197937
best_mse = 0.197937
best_update = 200
tail20_mean_mse = 0.198223
```

Expected output:

```text
Final test MSE = 0.197937
Best test MSE = 0.197937 at update 200
Final mean gate slope norm = 32.141263
```

Generated files:

```text
docs/bong_online_vmp/results/site_memory_schedule/summary.csv
docs/bong_online_vmp/results/site_memory_schedule/best_sqrt_site_b24_amean0p5_agate0p05_metrics.csv
docs/bong_online_vmp/results/site_memory_schedule/best_sqrt_site_b24_amean0p5_agate0p05_heatmap.png
```

The archived `summary.csv` should contain this best row:

```text
schedule = sqrt_epoch
batch_size = 24
batch_iterations = 5
alpha_mean = 0.5
alpha_gate = 0.05
final_mse = 0.197937
best_mse = 0.197937
tail20_mean_mse = 0.198223
tail20_below_baseline_fraction = 1.0
```

This points to the conclusion:

```text
fixed-batch site memory plus sqrt-decayed site damping retains nonlinear structure;
plain posterior-as-prior online VMP did not.
```

## Main Takeaway

The improvement came from changing the online inference structure:

```text
bad:
  every mini-batch replaces the whole posterior

good:
  every fixed mini-batch owns a site factor
  revisiting a batch refines that batch's site
  global posterior is the product of all sites and the prior
```

The square-root damping schedule makes the site refinement stable over longer runs.
