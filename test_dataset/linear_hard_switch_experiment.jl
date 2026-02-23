using ProbabilisticEnsembling
using DataFrames
using CSV
using StableRNGs
using Random: shuffle
using ExponentialFamily
using Plots

rng = StableRNG(1234)
train_size = 0.1
data = CSV.read("test_dataset/hard_switch_univariate.csv", DataFrame)
data = shuffle(rng, data)
n_train_last = convert(Int, ceil(train_size * size(data)[1]))

data_train = data[1:n_train_last, :]
scale = 1e-3
priors = Dict{Symbol, Any}(
    :w => [
        MvNormalMeanScalePrecision(zeros(2), scale),   # expert A: high for x<1.5
        MvNormalMeanScalePrecision(zeros(2), scale)    # expert B: high for x>1.5
    ],
    :τ => [GammaShapeRate(1.0, 1.0), GammaShapeRate(1.0, 1.0)],
    :β => [GammaShapeRate(1.0, 1e12), GammaShapeRate(1.0, 1e12)]
);

model = ProbabilisticEnsembling.univariate_dynamic_ensemble(
    n_forecasters = 2,
    n_obs = n_train_last,
    priors = priors
)

y_val = data_train[!, :OT];

constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false);
init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors);
predictions_matrix = permutedims(hcat(data_train[!, :pred_a], data_train[!, :pred_b]))  # 2 x n_obs
# Affine gating features [1, x] allow a threshold between x=1 and x=2.
features = [[1.0, x + 1e-12] for x in data_train[:, :regime]]
data = (y = data_train[:, :OT], features = features, predictions = predictions_matrix);

spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type = ProbabilisticEnsembling.Dynamic(),
    inference_iterations = 20,
    subsample_size = nothing,
    subsample_percentage = nothing
)

result = ProbabilisticEnsembling.run_training_rxinfer(spec, model, data; constraints = constraints, initialization = init, showprogress=true)

@show map(mean, result.posteriors[:w][end])

# --- Predict on two samples to inspect precision selector per regime ---
w_posteriors = result.posteriors[:w][end]
τ_posteriors = result.posteriors[:τ][end]
β_posteriors = result.posteriors[:β][end]
posterior_priors = Dict{Symbol,Any}(:w => w_posteriors, :τ => τ_posteriors, :β => β_posteriors)

predictions_probe = zeros(2, 2)  # 2 forecasters x 2 observations (dummy, not used for weights)
features_probe = [[1.0, 1.0], [1.0, 2.0]]  # [bias, regime] for regime 1 and regime 2
prediction_array = [missing, missing]

infer_probe = ProbabilisticEnsembling.predict_with_model(
    spec.prediction_type, spec.model_type, posterior_priors;
    n_forecasters = 2,
    n_steps = 2,
    prediction_array = prediction_array,
    predictions_test = predictions_probe,
    features_test = features_probe,
    prediction_iterations = spec.inference_iterations,
)

γ_probe = infer_probe.posteriors[:γ][end]
@show "Regime 1 weights:" mean.(γ_probe[:, 1]) ./ sum(mean.(γ_probe[:, 1]))
@show "Regime 2 weights:" mean.(γ_probe[:, 2]) ./ sum(mean.(γ_probe[:, 2]))
