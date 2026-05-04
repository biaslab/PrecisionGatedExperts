# PGE online (Kalman-like) VMP filtering

## Question

The paper trains VMP on the entire batch. For deployment we'd want **streaming / online** inference: process observations in small batches, carry forward the posterior over globals `(w, τ, β)`, then use it as the next batch's prior. This is the Bayesian analogue of a Kalman filter.

Open question: does the posterior converge to a sensible estimate as we keep feeding batches? In non-linear / non-conjugate Kalman-style filtering, this often fails because each recursive Gaussian approximation compounds error.

## Setup

- Data: `N_total` synthetic observations from `generate_synthetic_data` (same rig we used for Gibbs / NUTS / VMP comparisons).
- Batch size: **10 observations per step**.
- Per batch:
  - Priors are the **posteriors from the previous batch** (or the original broad priors at t=0).
  - Run VMP dynamic to convergence (or fixed budget, say 20 iters).
  - Record: final FE, wall-clock time, posterior means vs ground truth.
- Compare to **full-batch inference** (all `N_total` observations at once) as a reference.

## Files

- `run_online.jl` — the experiment driver. Writes per-batch metrics to `results/online_trajectory.csv` and a summary figure to `results/online_trajectory.{png,pdf}`.

## How to run

```bash
julia --project=. docs/pge_online_filtering/run_online.jl
```

## Success criteria

- **FE/N should decrease and stabilize** as batches accumulate.
- **Posterior mean error** (vs truth) should **not blow up** — filtering shouldn't diverge.
- **Total wall-clock** of 10 batches of 10 obs each should be **comparable to or faster than** a single 100-obs batch run.
