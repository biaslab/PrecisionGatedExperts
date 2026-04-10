#!/usr/bin/env julia

# XOR with Kalman-like deep CT + ReLU mean injection.
#
# Architecture: same CT applied repeatedly, fixed means injected at each depth level.
#
#   s[0] = features (padded to 4D)
#   for l = 1:L:
#       s[l] ~ CT(s[l-1], a, W)                    -- SAME a, W every level
#       for k = 1:K:
#           za[k,l] ~ softdot(s[l], e[k], τ)       -- project state onto LEARNED direction e[k]
#           γ[k,l]  ~ ReLU(za[k,l])                 -- ReLU precision
#           s[l]    ~ NormalMeanPrecision(μ[k], γ)   -- inject fixed mean
#
#   y ~ NormalMeanPrecision(output_from_s[L], τ_obs)
#
# Kalman filter analogy:
#   CT = prediction step (state dynamics)
#   NormalMeanPrecision with ReLU precision = observation step (mean injection)
#
# Learned parameters:
#   a_shared, W_shared — CT transition (shared across all depths)
#   e[k]               — projection directions for ReLU tapping (learned per neuron)
#   w_out              — output projection
#
# Fixed (not learned):
#   μ[k] = [-2.5, -0.5, +0.5, +2.5] — injected means
#
# All random init.

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
using Distributions: Wishart

const N_FEATURES = 3   # [1, x1, x2]
const N_HIDDEN = 4
const N_NEURONS = N_HIDDEN
# Fixed mean vectors (one per neuron, size N_HIDDEN)
# Random normal vectors — the model learns which to activate where
const FIXED_MEANS_RNG = StableRNG(999)
const FIXED_MEANS = [randn(FIXED_MEANS_RNG, N_HIDDEN) for _ in 1:N_HIDDEN]

const ct_meta_first = CTMeta(a -> reshape(a, N_HIDDEN, N_FEATURES))   # 3→4 (first layer)
const ct_meta_shared = CTMeta(a -> reshape(a, N_HIDDEN, N_HIDDEN))    # 4→4 (subsequent)

# ------------------------------------------------------------------
# Missing rule: MvNormalMeanScalePrecision(:γ) with message on out + PointMass on μ
# Needed for structured q(s, za, γ) where s gets MvNormal message from CT
# and μ is data (PointMass FIXED_MEANS[k]).
#
# VMP message to γ from N(out | μ, γ^{-1}I):
#   log ν(γ) = n/2 log(γ) - γ/2 E[||out - μ||²]
#   E[||out-μ||²] = tr(V_out) + ||m_out - μ||²
#   → GammaShapeRate(n/2 + 1, E[||out-μ||²] / 2)
# ------------------------------------------------------------------
@rule MvNormalMeanScalePrecision(:γ, Marginalisation) (
    m_out::MultivariateNormalDistributionsFamily,
    q_μ::PointMass,
) = begin
    m_mean, m_cov = mean_cov(m_out)
    μ = mean(q_μ)
    n = length(μ)
    diff = m_mean - μ
    sq_dist = tr(m_cov) + dot(diff, diff)
    return GammaShapeRate(n / 2 + 1, sq_dist / 2)
end

@model function xor_kalman_ct_relu(n_neurons, n_obs, features, y, priors)
    # First layer: 3→4 (own weights)
    a_first ~ priors[:a_first]
    W_first ~ priors[:W_first]
    # Shared CT weights for subsequent layers: 4→4 (same at every depth level)
    a_shared ~ priors[:a_shared]
    W_shared ~ priors[:W_shared]

    # Learnable projection directions — each neuron learns what to tap from state
    local e
    for k = 1:n_neurons
        e[k] ~ priors[:e][k]
    end

    τ_hi ~ priors[:τ_hi]
    τ_obs ~ priors[:τ_obs]
    w_out ~ priors[:w_out]

    local s, za, γ, pred_out

    for j = 1:n_obs
        # Depth 1: first CT maps 3D features → 4D hidden
        s[1, j] ~ ContinuousTransition(features[j], a_first, W_first) where {
            meta = ct_meta_first
        }
        for k = 1:n_neurons
            za[k, 1, j] ~ softdot(s[1, j], e[k], τ_hi)
            γ[k, 1, j] ~ ReLU(za[k, 1, j])
            # Inject fixed mean on the vector state s via MvNormalMeanScalePrecision
            # γ (scalar) acts as scale precision for all dims
            s[1, j] ~ MvNormalMeanScalePrecision(FIXED_MEANS[k], γ[k, 1, j])
        end

        # Depth 2
        s[2, j] ~ ContinuousTransition(s[1, j], a_shared, W_shared) where {
            meta = ct_meta_shared
        }
        for k = 1:n_neurons
            za[k, 2, j] ~ softdot(s[2, j], e[k], τ_hi)
            γ[k, 2, j] ~ ReLU(za[k, 2, j])
            s[2, j] ~ MvNormalMeanScalePrecision(FIXED_MEANS[k], γ[k, 2, j])
        end

        # Depth 3
        s[3, j] ~ ContinuousTransition(s[2, j], a_shared, W_shared) where {
            meta = ct_meta_shared
        }
        for k = 1:n_neurons
            za[k, 3, j] ~ softdot(s[3, j], e[k], τ_hi)
            γ[k, 3, j] ~ ReLU(za[k, 3, j])
            s[3, j] ~ MvNormalMeanScalePrecision(FIXED_MEANS[k], γ[k, 3, j])
        end

        # Output: project final state to scalar prediction
        pred_out[j] ~ softdot(s[3, j], w_out, τ_hi)
        y[j] ~ NormalMeanPrecision(pred_out[j], τ_obs)
    end
end

@constraints function xor_kalman_constraints()
    # Structured q(s) groups all states; za and γ mean-field with ProjectedTo
    q(a_first, W_first, a_shared, W_shared, e, s, za, γ, w_out, pred_out, τ_hi, τ_obs) =
        q(a_first)q(W_first)q(a_shared)q(W_shared)q(e)q(s)q(za, γ)q(w_out)q(pred_out)q(τ_hi)q(τ_obs)
    q(za)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
end

@initialization function xor_kalman_init(priors)
    q(a_first) = priors[:a_first]
    q(W_first) = priors[:W_first]
    q(a_shared) = priors[:a_shared]
    q(W_shared) = priors[:W_shared]
    q(e) = deepcopy(priors[:e])
    q(τ_hi) = priors[:τ_hi]
    q(τ_obs) = priors[:τ_obs]
    q(w_out) = priors[:w_out]
    q(s) = MvNormalMeanCovariance(zeros(N_HIDDEN), diagm(ones(N_HIDDEN)))
    q(za) = GammaShapeScale(2.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(pred_out) = NormalMeanVariance(0.0, 1.0)
    # Message init to break structured q(s, za, γ) deadlock
    μ(s) = MvNormalMeanCovariance(zeros(N_HIDDEN), diagm(ones(N_HIDDEN)))
    μ(za) = GammaShapeScale(2.0, 1.0)
    μ(γ) = GammaShapeScale(2.0, 1.0)
end

function make_priors(; seed::Int = 42)
    rng = StableRNG(seed)
    n_a_first = N_HIDDEN * N_FEATURES  # 4×3 = 12
    n_a_shared = N_HIDDEN * N_HIDDEN   # 4×4 = 16

    return Dict{Symbol,Any}(
        :a_first => MvNormalMeanCovariance(randn(rng, n_a_first) * 0.3, diagm(fill(1.0, n_a_first))),
        :W_first => Wishart(N_HIDDEN + 2, diagm(fill(10.0, N_HIDDEN))),
        :a_shared => MvNormalMeanCovariance(randn(rng, n_a_shared) * 0.3, diagm(fill(1.0, n_a_shared))),
        :W_shared => Wishart(N_HIDDEN + 2, diagm(fill(10.0, N_HIDDEN))),
        # Learnable projection directions (random init, one per neuron)
        :e => [MvNormalMeanScalePrecision(randn(rng, N_HIDDEN) * 0.5, 1.0) for _ in 1:N_NEURONS],
        :w_out => MvNormalMeanScalePrecision(randn(rng, N_HIDDEN) * 0.3, 1.0),
        :τ_hi => GammaShapeRate(1e4, 1.0),
        :τ_obs => GammaShapeRate(100.0, 1.0),
    )
end

build_features(df) = [Float64[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

function split_dataset(df; train_fraction = 0.3, seed = 2026)
    rng = StableRNG(seed)
    idx = randperm(rng, nrow(df))
    n = round(Int, train_fraction * nrow(df))
    df[idx[1:n], :], df[idx[n+1:end], :]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

df = CSV.read("test_dataset/xor_dataset.csv", DataFrame)
df_train, df_test = split_dataset(df; seed = 2027)
df_train = df_train[1:min(100, nrow(df_train)), :]

println("=" ^ 70)
println("XOR Kalman CT+ReLU: deep CT chain + ReLU precision taps")
println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
println("  depth=3, n_neurons=$N_NEURONS, fixed_means=$FIXED_MEANS")
println("=" ^ 70)

priors = make_priors()
features_train = build_features(df_train)

@info "Running inference..."
result = infer(
    model = xor_kalman_ct_relu(n_neurons = N_NEURONS, n_obs = nrow(df_train), priors = priors),
    data = (y = df_train.OT, features = features_train),
    constraints = xor_kalman_constraints(),
    initialization = xor_kalman_init(priors),
    iterations = 30,
    free_energy = false,
    showprogress = true,
    options = (limit_stack_depth = 500,),
)

println("\nLearned projection directions e[k]:")
for k in 1:N_NEURONS
    ek = round.(mean(result.posteriors[:e][end][k]); digits = 4)
    println("  N$k (μ_fixed=$(FIXED_MEANS[k])): e=$ek")
end

println("\nLearned output projection w_out:")
println("  $(round.(mean(result.posteriors[:w_out][end]); digits=4))")

# Evaluate — run predictions through the learned model manually
features_test = build_features(df_test)
y_test = df_test.OT

a_first_post = mean(result.posteriors[:a_first][end])
A_first = reshape(a_first_post, N_HIDDEN, N_FEATURES)
a_shared_post = mean(result.posteriors[:a_shared][end])
A_shared = reshape(a_shared_post, N_HIDDEN, N_HIDDEN)
e_post = [mean(result.posteriors[:e][end][k]) for k in 1:N_NEURONS]
w_out_post = mean(result.posteriors[:w_out][end])

y_pred = map(features_test) do f
    # Depth 1
    h1 = A_first * f
    for k in 1:N_NEURONS
        γk = max(0.0, dot(h1, e_post[k]))
        if γk > 1e-10
            # Kalman update: h1 ← (h1 * τ_prior + FIXED_MEANS[k] * γk) / (τ_prior + γk)
            # Simplified: just accumulate precision-weighted corrections
            h1 = h1 .+ (FIXED_MEANS[k] .- h1) .* (γk / (1.0 + γk))
        end
    end
    # Depth 2
    h2 = A_shared * h1
    for k in 1:N_NEURONS
        γk = max(0.0, dot(h2, e_post[k]))
        if γk > 1e-10
            h2 = h2 .+ (FIXED_MEANS[k] .- h2) .* (γk / (1.0 + γk))
        end
    end
    # Depth 3
    h3 = A_shared * h2
    for k in 1:N_NEURONS
        γk = max(0.0, dot(h3, e_post[k]))
        if γk > 1e-10
            h3 = h3 .+ (FIXED_MEANS[k] .- h3) .* (γk / (1.0 + γk))
        end
    end
    # Output
    return dot(h3, w_out_post)
end

mse = mean((y_pred .- y_test) .^ 2)
println("\nTest MSE = $(round(mse; digits=4))")
println("(Baseline: 4.0 | CT+ReLU depth-1: 3.90 | Interaction feature: 1.87)")

for (q, cond, label) in [
    (1, (r) -> r.x1 > 0 && r.x2 > 0, "Q1 → +2"),
    (2, (r) -> r.x1 < 0 && r.x2 > 0, "Q2 → -2"),
    (3, (r) -> r.x1 < 0 && r.x2 < 0, "Q3 → +2"),
    (4, (r) -> r.x1 > 0 && r.x2 < 0, "Q4 → -2"),
]
    mask = [cond(df_test[i, :]) for i in 1:nrow(df_test)]
    yp = y_pred[mask]; yt = y_test[mask]
    println("  $label: avg_pred=$(round(mean(yp);digits=3)) avg_true=$(round(mean(yt);digits=3)) MSE=$(round(mean((yp.-yt).^2);digits=4))")
end
