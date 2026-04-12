#!/usr/bin/env julia

# XOR with direct precision-weighted ReLU gating (no NormalMixture).
#
# 4 neurons, each with:
#   μ_k[j] ~ softdot(features, w_mean[k], τ)   -- learned mean
#   za_k[j] ~ softdot(features, w_a[k], τ)      -- learned gate pre-activation
#   γ_k[j] ~ ReLU(za_k[j])                       -- precision = max(0, za)
#   y[j] ~ NormalMeanPrecision(μ_k[j], γ_k[j])   -- each neuron contributes to y
#
# Symmetry breaking: asymmetric weight initialization (biases ±2, gate dirs Q1-Q4)

using CSV
using DataFrames
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using ClosedFormExpectations: EnzymeBackend
using ProbabilisticEnsembling
using RxInfer
using Statistics
using LinearAlgebra
using StableRNGs
using Random

const N_NEURONS = 4

@model function xor_relu_direct(n_neurons, features, y, priors)
    local w_mean, w_a, z_mean, za, γ, τ, out

    τ ~ priors[:τ]

    for k = 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k] ~ priors[:w_a][k]
    end

    for j = 1:length(y)
        for k = 1:n_neurons
            z_mean[k, j] ~ softdot(features[j], w_mean[k], τ)
            za[k, j] ~ softdot(features[j], w_a[k], τ) where {meta = LowRankMeta()}
            γ[k, j] ~ ReLU(za[k, j])
            out[j] ~ NormalMeanPrecision(z_mean[k, j], γ[k, j])
        end
        y[j] ~ NormalMeanPrecision(out[j], 1)
    end
end

@constraints function xor_relu_direct_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, out) = q(w_mean, z_mean)q(w_a)q(za)q(γ)q(τ)q(out)
    q(za)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
end

@initialization function xor_relu_direct_init(priors)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(0.0, 1.0)
    q(za) = GammaShapeScale(2.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]
end

function make_priors(; n_features::Int = 3, seed::Int = 42)
    rng = StableRNG(seed)

    # Tight prior on w_mean (precision=10) — prevents biases from collapsing away from ±2
    # Loose prior on w_a (precision=0.01) — allows gate weights to grow large
    w_mean = [MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-3, n_features))) for _ in 1:N_NEURONS]
    w_a = [MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-3, n_features))) for _ in 1:N_NEURONS]

    return Dict{Symbol,Any}(
        :w_mean => w_mean,
        :w_a => w_a,
        :τ => GammaShapeRate(1000.0, 1.0),
    )
end

build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

function split_dataset(df; train_fraction = 0.3, seed = 2026)
    rng = StableRNG(seed)
    idx = randperm(rng, nrow(df))
    n = round(Int, train_fraction * nrow(df))
    df[idx[1:n], :], df[idx[n+1:end], :]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

df = CSV.read("test_dataset/xor_simple_dataset.csv", DataFrame)
df_train, df_test = split_dataset(df; seed = 2027)
df_train = df_train[1:min(500, nrow(df_train)), :]

println("=" ^ 70)
println("XOR ReLU Direct: precision-weighted, $(N_NEURONS) neurons")
println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
println("=" ^ 70)

priors = make_priors()
features_train = build_features(df_train)

# # Use SubsampledData to prevent full-batch variance collapse
# # Small batches give different gradient signals to different neurons
# subsample_size = 30
# y_sub = ProbabilisticEnsembling.SubsampledData(collect(df_train.OT), subsample_size)
# features_sub = ProbabilisticEnsembling.SubsampledData(features_train, subsample_size)

result = infer(
    model = xor_relu_direct(n_neurons = N_NEURONS, priors = priors),
    data = (y = df_train.OT, features = features_train),
    constraints = xor_relu_direct_constraints(),
    initialization = xor_relu_direct_init(priors),
    iterations = 50,
    free_energy = false,
    showprogress = true,
    options = (limit_stack_depth = 100,),
)

try
    fe = result.free_energy
    println("\nFree energy: first=$(round(fe[1]; digits=2)) last=$(round(fe[end]; digits=2))")
catch
    println("\n(Free energy not computed)")
end

println("\nLearned weights:")
for k in 1:N_NEURONS
    wm = round.(mean(result.posteriors[:w_mean][end][k]); digits = 4)
    wa = round.(mean(result.posteriors[:w_a][end][k]); digits = 4)
    println("  neuron $k: w_mean=$wm  w_a=$wa")
end

# Evaluate
features_test = build_features(df_test)
w_means_post = [mean(result.posteriors[:w_mean][end][k]) for k in 1:N_NEURONS]
w_a_post = [mean(result.posteriors[:w_a][end][k]) for k in 1:N_NEURONS]

y_pred = map(features_test) do f
    μs = [dot(w_means_post[k], f) for k in 1:N_NEURONS]
    γs = [max(0.0, dot(w_a_post[k], f)) for k in 1:N_NEURONS]
    total = sum(γs)
    total < 1e-10 && return 0.0
    sum(γs .* μs) / total
end

y_test = df_test.OT
mse = mean((y_pred .- y_test) .^ 2)
println("\nTest MSE (precision-weighted) = $(round(mse; digits=4))")
println("(Baseline: predicting 0 gives MSE ≈ 4.0)")

for (q, cond, label) in [
    (1, (r) -> r.x1 > 0 && r.x2 > 0, "Q1 → +2"),
    (2, (r) -> r.x1 < 0 && r.x2 > 0, "Q2 → -2"),
    (3, (r) -> r.x1 < 0 && r.x2 < 0, "Q3 → +2"),
    (4, (r) -> r.x1 > 0 && r.x2 < 0, "Q4 → -2"),
]
    mask = [cond(df_test[i, :]) for i in 1:nrow(df_test)]
    yp = y_pred[mask]
    yt = y_test[mask]
    println("  $label: avg_pred=$(round(mean(yp);digits=3)) avg_true=$(round(mean(yt);digits=3)) MSE=$(round(mean((yp.-yt).^2);digits=4))")
end
