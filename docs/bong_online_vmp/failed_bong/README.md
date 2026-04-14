# Failed / Weaker BONG-Like Experiments

These notes document experiments that were useful but did not solve stable online XOR learning.

Read them in this order:

1. `01_posterior_as_prior.md`
2. `02_damping_and_svi_scaling.md`
3. `03_constant_site_memory.md`

The failed/auxiliary scripts are not kept under `test_dataset` after cleanup. These files are historical notes: they explain what was tested, the observed metrics, and where existing output artifacts were written.

The main conclusion from the failed experiments:

```text
plain posterior-as-prior online VMP can discover nonlinear structure,
but it usually does not retain it.
```

The successful direction is documented in:

```text
../how_site_memory_works.md
```
