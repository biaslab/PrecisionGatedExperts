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
