# probabilistic_ensemble_forecasting

You can training scripts with:

```bash
julia --project=. scripts/train_ett_lux.jl
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
