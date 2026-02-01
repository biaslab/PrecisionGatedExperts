# probabilistic_ensemble_forecasting

You can training scripts with:

```bash
julia --project=. scripts/train_ett_lstm_enzyme.jl
```

Run inference with trained model: 

```bash
julia --project=. scripts/infer_ett_enzyme.jl models/ETTh1_h96_CNN_enzyme.jld2
```

Run static ensemble model with neural models: 

```bash
julia --project=. scripts/neural_ensemble_rxinfer.jl models/ETTh1_h96_CNN_enzyme.jld2 models/ETTh1_h96_LSTM_enzyme.jld2
```

Run the dynamic ensemble with neural models:

Models must come from the same dataset and use the same horizon. The dataset name is stored at the top of each model file’s name. The dataset used at inference time is determined by the models you pass in. You can pass two or more models as arguments after the script name.
Right now constant models like quantile 10 and quantile 90 added by default to each combination of models. 

Example (CNN + LSTM trained on ETTh1 with horizon 96):

```bash
julia --project=. scripts/dynamic_neural_ensemble_rxinfer.jl models/ETTh1_h96_CNN_enzyme.jld2 models/ETTh1_h96_LSTM_enzyme.jld2
```

General usage:

```bash
julia --project=. scripts/dynamic_neural_ensemble_rxinfer.jl <model_path_1> <model_path_2> ... <model_path_n>
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
- Horizons denote one-step-ahead offsets; for example, horizon 96 predicts the value 96 steps after the end of each input window.
