# Noisy Experts Diagonal — Paper Results

## Configuration

All configs match the dynamic model paper results exactly (same features, iterations, priors) plus κ prior.

### ETTh1 / ETTh2 (univariate, uniwindow, n_features=33)

| Prior | Value |
|-------|-------|
| β | GammaShapeRate(1.0, 1e3) |
| τ | GammaShapeRate(1.0, 1e-3) |
| κ | GammaShapeRate(1.0, 1.0) |
| w | MvNormalMeanScalePrecision(n_features=33, scale=0.01) |
| inference_iterations | 5 |
| prediction_iterations | 1 |

### Exchange Rate (multivariate, window, n_features=57)

| Prior | h96 | h192 | h336 | h720 |
|-------|-----|------|------|------|
| β | GammaShapeRate(1.0, 1e6) | same | same | same |
| τ | GammaShapeRate(1.0, 1.0) | same | same | GammaShapeRate(1.0, 0.1) |
| κ | GammaShapeRate(1.0, 1.0) | same | same | same |
| w | MvNormalMeanScalePrecision(57, 1.0) | same | same | same |
| inference_iterations | 5 | 5 | 5 | 5 |
| prediction_iterations | 2 | 1 | 2 | 2 |

## TODO

- [x] 1. Create 12 YAML session configs
- [x] 2. Run ETTh1 h96
- [x] 3. Run ETTh1 h192
- [x] 4. Run ETTh1 h336
- [x] 5. Run ETTh1 h720
- [x] 6. Run ETTh2 h96
- [x] 7. Run ETTh2 h192
- [x] 8. Run ETTh2 h336
- [x] 9. Run ETTh2 h720
- [x] 10. Run exchange_rate h96
- [x] 11. Run exchange_rate h192
- [x] 12. Run exchange_rate h336
- [x] 13. Run exchange_rate h720
- [x] 14. Copy results with clean names to paper/results/noisy_experts_diagonal/

## Results

| # | Dataset | Horizon | MSE | MAE | RMSE | R² | NLL | Status |
|---|---------|---------|-----|-----|------|----|-----|--------|
| 1 | ETTh1 | 96 | 0.136 | 0.295 | 0.369 | 0.156 | -0.444 | done |
| 2 | ETTh1 | 192 | 0.119 | 0.275 | 0.346 | 0.246 | -0.398 | done |
| 3 | ETTh1 | 336 | 0.100 | 0.253 | 0.316 | 0.361 | -0.346 | done |
| 4 | ETTh1 | 720 | 0.127 | 0.288 | 0.356 | 0.158 | -0.424 | done |
| 5 | ETTh2 | 96 | 0.368 | 0.507 | 0.606 | 0.526 | -1.018 | done |
| 6 | ETTh2 | 192 | 0.365 | 0.496 | 0.604 | 0.522 | -1.031 | done |
| 7 | ETTh2 | 336 | 0.369 | 0.499 | 0.608 | 0.500 | -1.043 | done |
| 8 | ETTh2 | 720 | 0.339 | 0.471 | 0.582 | 0.510 | -0.940 | done |
| 9 | exchange_rate | 96 | 0.333 | 0.441 | 0.577 | 0.362 | -10.97 | done |
| 10 | exchange_rate | 192 | 0.708 | 0.663 | 0.842 | -0.434 | -15.88 | done |
| 11 | exchange_rate | 336 | 0.809 | 0.675 | 0.900 | -0.454 | -23.31 | done |
| 12 | exchange_rate | 720 | 1.491 | 0.930 | 1.221 | -1.762 | -34.49 | done |
