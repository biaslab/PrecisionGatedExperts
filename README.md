# probabilistic_ensemble_forecasting

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
