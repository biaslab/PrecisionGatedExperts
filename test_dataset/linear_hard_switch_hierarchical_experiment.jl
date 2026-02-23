using ProbabilisticEnsembling
using DataFrames
using CSV
using StableRNGs
using Random: shuffle
using ExponentialFamily
using Plots
using Statistics

# Minimal hierarchical hard-switch experiment on synthetic dataset.
# Run:
#   julia --project=. test_dataset/linear_hard_switch_hierarchical_experiment.jl

rng = StableRNG(1234)
train_size = 0.1
inference_iterations = 60

data = CSV.read("test_dataset/hard_switch_univariate.csv", DataFrame)
data = shuffle(rng, data)
n_train = Int(ceil(train_size * nrow(data)))
data_train = data[1:n_train, :]

# Sparse hierarchical priors (best-guess defaults).
# - affine features [1, regime] for thresholding
# - small alpha encourages sparse precision allocation
scale = 1e-3
alpha = 0.01
priors = Dict{Symbol,Any}(
    :w => [
        MvNormalMeanScalePrecision(zeros(2), scale),
        MvNormalMeanScalePrecision(zeros(2), scale),
    ],
    :τ => [GammaShapeRate(1, 1), GammaShapeRate(1.0, 1.0)],
    :ρ => [GammaShapeRate(1.0, 1e12), GammaShapeRate(1.0, 1e12)],
    :α => alpha,
)

model = ProbabilisticEnsembling.hierarchical_model(
    n_forecasters = 2,
    n_obs = n_train,
    priors = priors,
)

features = [[1.0, x + 1e-12] for x in data_train[:, :regime]]
predictions_matrix = permutedims(hcat(data_train[!, :pred_a], data_train[!, :pred_b])) # 2 x n_obs
fit_data = (y = data_train[!, :OT], features = features, predictions = predictions_matrix)

spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type = ProbabilisticEnsembling.Hierarchical(),
    inference_iterations = inference_iterations,
    subsample_size = nothing,
    subsample_percentage = nothing,
)

constraints = ProbabilisticEnsembling.hierarchical_constraints(priors, false)
init = ProbabilisticEnsembling.hierarchical_init(priors)

result = ProbabilisticEnsembling.run_training_rxinfer(
    spec,
    model,
    fit_data;
    constraints = constraints,
    initialization = init,
    showprogress = true,
)

@show map(mean, result.posteriors[:w][end])

# Probe selector with fixed regimes.
w_post = result.posteriors[:w][end]
τ_post = result.posteriors[:τ][end]
ρ_post = result.posteriors[:ρ][end]
posterior_priors = Dict{Symbol,Any}(:w => w_post, :τ => τ_post, :ρ => ρ_post, :α => alpha)

predictions_probe = zeros(2, 2) # dummy; not used for weights when y is missing
features_probe = [[1.0, 1.0], [1.0, 2.0]]
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
    prediction_iterations = spec.inference_iterations,
)

γ_probe = infer_probe.posteriors[:γ][end]
@show "Regime 1 weights:" mean.(γ_probe[:, 1]) ./ sum(mean.(γ_probe[:, 1]))
@show "Regime 2 weights:" mean.(γ_probe[:, 2]) ./ sum(mean.(γ_probe[:, 2]))

plot(result.free_energy, title = "Hierarchical Free Energy", xlabel = "Iteration", ylabel = "Free Energy")
savefig("test_dataset/viz/hierarchical_linear_hard_switch_free_energy.png")
