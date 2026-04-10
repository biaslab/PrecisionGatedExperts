using ProbabilisticEnsembling
using DataFrames
using CSV
using StableRNGs
using Random: shuffle
using ExponentialFamily
using ExponentialFamilyProjection
using ClosedFormExpectations
using ClosedFormExpectations: EnzymeBackend
using Enzyme
using RxInfer
using BayesBase
using Statistics

# -----------------------------------------------------------------------
# Test: DynamicReLU ensemble on hard-switch dataset
#
# The hard-switch dataset has two "regimes" with two experts (pred_a, pred_b).
# In regime 1, expert A is accurate; in regime 2, expert B is accurate.
# The model should learn feature-dependent precisions that select the right
# expert per regime.
# -----------------------------------------------------------------------

rng = StableRNG(1234)
train_size = 0.1
data = CSV.read("test_dataset/hard_switch_univariate.csv", DataFrame)
data = shuffle(rng, data)
n_train_last = convert(Int, ceil(train_size * size(data)[1]))

data_train = data[1:n_train_last, :]
scale = 1e-3
priors = Dict{Symbol,Any}(
    :w => [
        MvNormalMeanScalePrecision(zeros(2), scale),
        MvNormalMeanScalePrecision(zeros(2), scale),
    ],
    :τ => [GammaShapeRate(1.0, 1e-3), GammaShapeRate(1.0, 1e-3)],
)

model = ProbabilisticEnsembling.univariate_dynamic_relu_ensemble(
    n_forecasters = 2,
    n_obs = n_train_last,
    priors = priors,
)

constraints = ProbabilisticEnsembling.univariate_dynamic_relu_ensemble_constraints(priors, false)
init = ProbabilisticEnsembling.univariate_dynamic_relu_ensemble_init(priors)

predictions_matrix = permutedims(hcat(data_train[!, :pred_a], data_train[!, :pred_b]))  # 2 x n_obs
features = [[1.0, x + 1e-12] for x in data_train[:, :regime]]
train_data = (y = data_train[:, :OT], features = features, predictions = predictions_matrix)

spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type = ProbabilisticEnsembling.DynamicReLU(),
    inference_iterations = 40,
    subsample_size = nothing,
    subsample_percentage = nothing,
)

@info "Starting DynamicReLU training..."
result = infer(
    model = model,
    data = train_data,
    constraints = constraints,
    initialization = init,
    iterations = spec.inference_iterations,
    free_energy = true,
    showprogress = true,
)

@show map(mean, result.posteriors[:w][end])

# --- Predict on two samples to inspect precision selector per regime ---
w_posteriors = result.posteriors[:w][end]
τ_posteriors = result.posteriors[:τ][end]
posterior_priors = Dict{Symbol,Any}(:w => w_posteriors, :τ => τ_posteriors)

predictions_probe = zeros(2, 2)
features_probe = [[1.0, 1.0], [1.0, 2.0]]  # regime 1 and regime 2
prediction_array = [missing, missing]

infer_probe = ProbabilisticEnsembling.predict_with_model(
    spec.prediction_type,
    spec.model_type,
    posterior_priors;
    n_forecasters = 2,
    n_steps = 2,
    prediction_array = prediction_array,
    predictions_test = predictions_probe,
    features_test = features_probe,
    prediction_iterations = 20,
)

γ_probe = infer_probe.posteriors[:γ][end]
weights_r1 = mean.(γ_probe[:, 1]) ./ sum(mean.(γ_probe[:, 1]))
weights_r2 = mean.(γ_probe[:, 2]) ./ sum(mean.(γ_probe[:, 2]))
@show "Regime 1 weights:" weights_r1
@show "Regime 2 weights:" weights_r2

# Check: in regime 1 expert A should dominate, in regime 2 expert B
if weights_r1[1] > weights_r1[2]
    @info "Regime 1: Expert A dominates (correct)" w_A = weights_r1[1] w_B = weights_r1[2]
else
    @warn "Regime 1: Expert B dominates (unexpected)" w_A = weights_r1[1] w_B = weights_r1[2]
end

if weights_r2[2] > weights_r2[1]
    @info "Regime 2: Expert B dominates (correct)" w_A = weights_r2[1] w_B = weights_r2[2]
else
    @warn "Regime 2: Expert A dominates (unexpected)" w_A = weights_r2[1] w_B = weights_r2[2]
end
