using ProbabilisticEnsembling
using DataFrames
using CSV
using StableRNGs
using Random: shuffle
using ExponentialFamily
using ExponentialFamilyProjection
using ClosedFormExpectations
using Enzyme
using RxInfer
using BayesBase
using Plots
using Statistics

# E[log(x)] for LogNormal(μ, σ) is μ (the log-mean parameter).
# Required by ReactiveMP's AverageEnergy rule for GammaShapeRate.
BayesBase.mean(::typeof(log), dist::LogNormal) = dist.μ

# Custom constraints using LogNormal instead of Gamma for q(γ).
# This is valid because ClosedFormExpectations.jl now supports LogNormal as q,
# i.e. it can compute E_q[log p(x)] in closed form when q ~ LogNormal.
@constraints function univariate_dynamic_lognormal_constraints(priors, prediction)
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
    q(
        z,
    )::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())),
    )
    q(
        γ,
    )::ProjectedTo(
        LogNormal,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())),
    )
    if prediction
        for (i, prior) in enumerate(deepcopy(priors[:w]))
            q(w[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:τ])
            q(τ[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:β])
            q(β[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
    end
end

# Initialization: q(γ) = LogNormal(0, 1) → mean ≈ 1.65, positive support, matches Gamma(1,1) spirit.
@initialization function univariate_dynamic_lognormal_init(priors)
    q(w) = deepcopy(priors[:w])
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = LogNormal(0.0, 1.0)
    q(τ) = priors[:τ]
    q(β) = priors[:β]
end

rng = StableRNG(1234)
train_size = 0.1
data = CSV.read("test_dataset/hard_switch_univariate.csv", DataFrame)
data = shuffle(rng, data)
n_train_last = convert(Int, ceil(train_size * size(data)[1]))

data_train = data[1:n_train_last, :]
scale = 1e-3
priors = Dict{Symbol, Any}(
    :w => [
        MvNormalMeanScalePrecision(zeros(2), scale),
        MvNormalMeanScalePrecision(zeros(2), scale)
    ],
    :τ => [GammaShapeRate(1.0, 1.0), GammaShapeRate(1.0, 1.0)],
    :β => [GammaShapeRate(1.0, 1e4), GammaShapeRate(1.0, 1e4)]
);

model = ProbabilisticEnsembling.univariate_dynamic_ensemble(
    n_forecasters = 2,
    n_obs = n_train_last,
    priors = priors
)

constraints = univariate_dynamic_lognormal_constraints(priors, false)
init = univariate_dynamic_lognormal_init(priors)

predictions_matrix = permutedims(hcat(data_train[!, :pred_a], data_train[!, :pred_b]))  # 2 x n_obs
features = [[1.0, x + 1e-12] for x in data_train[:, :regime]]
data = (y = data_train[:, :OT], features = features, predictions = predictions_matrix);

spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type = ProbabilisticEnsembling.Dynamic(),
    inference_iterations = 40,
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

predictions_probe = zeros(2, 2)
features_probe = [[1.0, 1.0], [1.0, 2.0]]  # regime 1 and regime 2
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

plot(result.free_energy, title = "Free Energy (LogNormal γ)")
