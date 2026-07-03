# probabilistic_ensemble_forecasting

Check out our paper here!

[![arXiv](https://img.shields.io/badge/arXiv-2605.29467-b31b1b.svg)](https://arxiv.org/abs/2605.29467)

## Running experiments

All experiments are configured via YAML session files and launched through `run_experiment`:

```bash
julia --project=. -e 'using ProbabilisticEnsembling; run_experiment("sessions/dynamic/dynamic_ETTh1_96.yaml")'
```

Pre-configured session files are available under `sessions/` for three ensemble types:

- `sessions/static/` - static ensemble
- `sessions/dynamic/` - dynamic ensemble
- `sessions/hierarchical/` - hierarchical ensemble

Each YAML file specifies the dataset, horizon, expert models, priors, and iteration counts. Example (`sessions/dynamic/dynamic_ETTh1_96.yaml`):

```yaml
params:
  prediction_type: "univariate"
  column: "OT"
  model_type: "dynamic"
  dataset: "ETTh1"
  dataset_path: "data/ETTh1.csv"
  horizon: 96
  inference_iterations: 500
  prediction_iterations: 1
  experts:
    - "models/ETTh1_h96_s96_CNN_enzyme.jld2"
    - "models/ETTh1_h96_s96_MLP_enzyme.jld2"
    - "models/ETTh1_h96_s96_LSTM_enzyme.jld2"
    - "models/ETTh1_h96_s96_DLinear_enzyme.jld2"
    - "models/ETTh1_h96_s96_NConv_enzyme.jld2"
  priors:
    β:
      type: "GammaShapeRate"
      shape: 1.0
      rate: 1.0
    τ:
      type: "GammaShapeRate"
      shape: 1.0
      rate: 1.0
    w:
      type: "MvNormalMeanScalePrecision"
      n_features: 22
      scale: 1.0
```

Quantile forecasters configuration:
- `selected_quantiles` (or `quantiles`): explicit quantile list in percent (`[10, 90]`) or unit interval (`[0.1, 0.9]`).
- `number_of_quantiles`: generates uniformly spaced quantiles in `(0, 1)`; e.g. `4` -> `[20, 40, 60, 80]`.
- If neither field is provided, defaults to `[10, 90]`.
- `experts` may be empty (`experts: []`) for quantile-only runs.

## Datasets

to download traffic.csv and electricity.csv: https://drive.google.com/drive/folders/1x3lrzu0qMUXMAJPxg6gIWV_4h6sGFKn7?usp=sharing

This repository already includes the datasets under `data/` as CSV files:

- `data/ETTh1.csv`
- `data/ETTh2.csv`
- `data/electricity.csv`
- `data/traffic.csv`
- `data/exchange_rate.csv`

The training/inference scripts auto-detect these CSVs from `data/`. Trained models are saved to `models/`.

| Dataset (file)         | Dims | Horizons                 | Split (train, val, test) |Propostions splits| Frequency |
|------------------------|-----:|--------------------------|---------------------------|--|-----------|
| ETTh1 (`ETTh1.csv`)    |    7 | {96, 192, 336, 720}      | (8545, 2881, 2881)        |6:2:2| 15 min    |
| ETTh2 (`ETTh2.csv`)    |    7 | {96, 192, 336, 720}      | (8545, 2881, 2881)        |6:2:2| 15 min    |
| Electricity (`electricity.csv`) | 321 | {96, 192, 336, 720} | (18317, 2633, 5261)      |7:1:2| Hourly    |
| Traffic (`traffic.csv`)        | 862 | {96, 192, 336, 720} | (12185, 1757, 3509)       |7:1:2| Hourly    |
| Exchange (`exchange_rate.csv`) |   8 | {96, 192, 336, 720} | (5120, 665, 1422)         |7:1:2| Daily     |

Notes
- Splits are chronological: the first block is used for training, the next for validation, and the final for testing.

## Neural ensemble (Adaptive Mixture of Local Experts)

A separate pipeline implements the softmax gating network from Jacobs, Jordan, Nowlan & Hinton (1991). Instead of Bayesian inference it trains a Lux neural network via MLE to learn context-dependent expert weights.

### Running

```bash
julia --project=. -e 'using ProbabilisticEnsembling; run_neural_ensemble_experiment("sessions/neural_ensemble/neural_ensemble_ETTh1_96.yaml")'
```

Session files live under `sessions/neural_ensemble/`. Example (`neural_ensemble_ETTh1_96.yaml`):

```yaml
params:
  pipeline: "neural_ensemble"
  prediction_type: "univariate"
  column: "OT"
  dataset: "ETTh1"
  dataset_path: "data/ETTh1.csv"
  experts:
    - "models/ETTh1_h96_s96_CNN_enzyme.jld2"
    - "models/ETTh1_h96_s96_MLP_enzyme.jld2"
    - "models/ETTh1_h96_s96_LSTM_enzyme.jld2"
    - "models/ETTh1_h96_s96_DLinear_enzyme.jld2"
    - "models/ETTh1_h96_s96_NConv_enzyme.jld2"
  train_set: true
  feature_type: "simple"
  quantiles: [10, 90]
  gating:
    layers: 1
    hidden_dim: 64
    n_epochs: 100
    patience: 50
    min_delta: 1.0e-6
    learning_rate: 1.0e-3
  save_dir: "saved_neural_ensemble_models"
```

### Configuration fields

| Field | Description |
|-------|-------------|
| `prediction_type` | `"univariate"` (single column) or `"multivariate"` (all columns) |
| `column` | Target column name, required for univariate (e.g. `"OT"`) |
| `train_set` | `true`: train gating on train split, monitor val. `false`: train on val, monitor train |
| `feature_type` | `"simple"`, `"window"`, `"uniwindow"`, `"fft"`, `"fft:5"`, `"ae"`, `"vae"` |
| `quantiles` | Quantile baselines added as extra constant experts, in percent (e.g. `[10, 90]`) |
| `gating.layers` | `1` for linear gating, `>1` for MLP |
| `gating.hidden_dim` | Hidden layer size for MLP gating |
| `gating.n_epochs` | Maximum training epochs |
| `gating.patience` | Early stopping patience |
| `gating.min_delta` | Minimum improvement for early stopping |
| `gating.learning_rate` | Adam optimizer learning rate |
| `save_dir` | Directory for saved trained models |

The `horizon` is inferred from the first expert model's metadata.

### Predicting from a saved model

Trained models are saved to `saved_neural_ensemble_models/` as JLD2 files. To re-run predictions on the test set from a saved model:

```bash
julia --project=. -e '
using ProbabilisticEnsembling
results = predict_from_trained_neural_ensemble("saved_neural_ensemble_models/ETTh1_h96_neural_ensemble_09f181e9.jld2")
println(results.ensemble_metrics)
'
```

## Comparing trained models

Use `scripts/compare_models.jl` to visually compare dynamic and static ensemble predictions side by side. The script loads saved results from `paper/results/`, runs inference with the trained posteriors, and produces a combined plot.

```bash
julia --project scripts/compare_models.jl <dataset> <horizon> [--dim <d>]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `dataset` | Dataset name: `ETTh1`, `ETTh2`, `exchange_rate`, `electricity`, `traffic` |
| `horizon` | Prediction horizon: `96`, `192`, `336`, `720` |
| `--dim d` | Which dimension to plot for multivariate datasets (default: `1`). Ignored for univariate datasets. |
| `--show-val` | Prepend validation ground truth to the predictions panel with a dashed vertical boundary line. Useful for seeing what the dynamic model learned from. |

**Examples:**

```bash
# Univariate — no --dim needed
julia --project scripts/compare_models.jl ETTh1 96
julia --project scripts/compare_models.jl ETTh2 336

# Multivariate — pick a dimension to visualise
julia --project scripts/compare_models.jl exchange_rate 192 --dim 3
julia --project scripts/compare_models.jl electricity 96 --dim 1
julia --project scripts/compare_models.jl traffic 720 --dim 5

# Show validation set ground truth before test predictions
julia --project scripts/compare_models.jl exchange_rate 192 --dim 3 --show-val
```

The output plot (`compare_<dataset>_h<horizon>_dim<dim>.png`) contains:

- **Predictions panel** — ground truth vs dynamic (blue) and static (red) ensemble means with 95% confidence bands. MSE and MAE are shown in the legend.
- **Dynamic influence panel** — time-varying normalised expert weights (γ) with 95% credible intervals.
- **Dynamic TopShare panel** — dominance of the strongest expert over time (`max γᵢ / Σγ`).
- **Static influence panel** — bar chart of normalised expert weights.

If only one model type is available for a dataset/horizon pair, the script still runs with whatever is present.
