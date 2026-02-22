# Synthetic Hard-Switch Dataset

This folder contains a simple controlled dataset for checking whether gating can learn non-trivial switching.

## Idea

- `regime` alternates every 200 points.
- Target `OT` follows one latent pattern in regime 1 and another in regime 2.
- `pred_a` is strong in regime 1 and weak in regime 2.
- `pred_b` is strong in regime 2 and weak in regime 1.
- `pred_avg` is a plain average baseline.

If your gating model is working, it should:
- prefer `pred_a` in regime 1,
- prefer `pred_b` in regime 2,
- and beat `pred_avg`.

## Generate

```bash
julia --project=. test_dataset/generate_hard_switch_dataset.jl
```

This writes:

- `test_dataset/hard_switch_univariate.csv`

