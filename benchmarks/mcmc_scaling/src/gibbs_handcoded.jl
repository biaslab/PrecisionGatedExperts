using Distributions
using LinearAlgebra
using Random
using Statistics

# ============================================================
# Conjugate conditional samplers for globals (= VMP messages + rand)
#
# These compute the SAME posterior parameters as the VMP messages
# to w, τ, β in the RxInfer model. The only difference: VMP returns
# the distribution, Gibbs draws a sample from it.
# ============================================================

"""
Sample w[i] | z[i,:], features, τ[i]  (conjugate Gaussian)

The softdot factor z[i,j] ~ Normal(w[i]'φ[j], 1/τ[i]) is conjugate with
the Gaussian prior w[i] ~ MvNormal(0, (1/prior_prec)*I).

Posterior:  Λ_post = prior_prec*I + τ[i] * Σ_j φ[j]φ[j]'
            η_post = τ[i] * Σ_j φ[j]*z[i,j]
            w[i] ~ MvNormal(Λ_post \\ η_post, inv(Λ_post))

This is the SAME computation as the VMP message μ→w[i].
"""
function sample_w_conditional(
    z_i::AbstractVector,     # z[i, :] — length N
    features::AbstractVector, # Vector of feature vectors, each length d
    τ_i::Real;
    prior_prec::Real = 0.01, # MvNormalMeanScalePrecision(scale=0.01) → precision = 0.01
)
    d = length(features[1])
    N = length(z_i)

    # Posterior precision: Λ = prior_prec*I + τ * Φ*Φ'
    Λ_post = Matrix{Float64}(prior_prec * I, d, d)
    η_post = zeros(d)

    for j in 1:N
        φ = features[j]
        Λ_post .+= τ_i .* (φ * φ')
        η_post .+= τ_i .* φ .* z_i[j]
    end

    μ_post = Λ_post \ η_post
    Σ_post = Symmetric(inv(Λ_post))
    return rand(MvNormal(μ_post, Σ_post))
end

"""
Sample τ[i] | w[i], z[i,:], features  (conjugate Gamma)

The softdot factor: z[i,j] ~ Normal(w[i]'φ[j], 1/τ[i]), with τ as precision.
Likelihood ∝ τ^(N/2) exp(-τ/2 Σ_j (z[i,j] - w[i]'φ[j])²)
Combined with prior Gamma(shape, rate):
  shape_post = shape + N/2
  rate_post  = rate + Σ_j (z[i,j] - w[i]'φ[j])² / 2

This is the SAME computation as the VMP message μ→τ[i].
"""
function sample_τ_conditional(
    z_i::AbstractVector,
    w_i::AbstractVector,
    features::AbstractVector;
    prior_shape::Real = 1.0,
    prior_rate::Real = 1e-3,  # GammaShapeRate(1, 1e-3)
)
    N = length(z_i)
    ss = 0.0
    for j in 1:N
        r = z_i[j] - dot(w_i, features[j])
        ss += r * r
    end
    shape_post = prior_shape + N / 2
    rate_post = prior_rate + ss / 2
    return rand(Gamma(shape_post, 1.0 / rate_post))
end

"""
Sample β[i] | z[i,:]  (conjugate Gamma)

The Gamma(1, β) prior on γ[i,j] = exp(z[i,j]):
  p(γ|β) = β exp(-β γ)  (shape=1)
Likelihood over j: ∝ β^N exp(-β Σ_j exp(z[i,j]))
Combined with prior Gamma(shape, rate):
  shape_post = shape + N
  rate_post  = rate + Σ_j exp(z[i,j])

This is the SAME computation as the VMP message μ→β[i].
"""
function sample_β_conditional(
    z_i::AbstractVector;
    prior_shape::Real = 1.0,
    prior_rate::Real = 1e3,  # GammaShapeRate(1, 1e3)
)
    N = length(z_i)
    γ_sum = sum(exp(z_i[j]) for j in 1:N)
    shape_post = prior_shape + N
    rate_post = prior_rate + γ_sum
    return rand(Gamma(shape_post, 1.0 / rate_post))
end

# ============================================================
# 1D slice sampler for z[i,j] (non-conjugate conditional)
#
# The conditional p(z | w[i], τ[i], β[i], y[j], pred[i,j]) has three factors:
#   1. Normal(w[i]'φ[j], 1/τ[i])   — softdot
#   2. z - β[i]*exp(z)              — Gamma(1,β) prior on exp(z) + Jacobian
#   3. Normal(pred[i,j], 1/√exp(z)) — likelihood
#
# This mirrors the per-edge fixed-point solve of Theorem 2.
# VMP solves this with a deterministic fixed-point iteration;
# Gibbs solves it with slice sampling. Same 1D problem.
# ============================================================

"""
Log-density of z[i,j] given all other variables (the 1D non-conjugate conditional).
"""
function z_logdensity(
    z::Real, w_i, τ_i, β_i, y_j, features_j, pred_ij,
)
    # Factor 1: softdot
    μ_softdot = dot(w_i, features_j)
    ld = -0.5 * τ_i * (z - μ_softdot)^2

    # Factor 2: Gamma(1, β) prior on γ = exp(z), change of variables
    ld += z - β_i * exp(z)

    # Factor 3: likelihood y ~ Normal(pred, 1/√γ) where γ = exp(z)
    γ = exp(z)
    ld += 0.5 * log(γ) - 0.5 * γ * (y_j - pred_ij)^2

    return ld
end

"""
Univariate slice sampler (Neal 2003, stepping-out + shrinkage).
"""
function slice_sample(logf, x0::Real; w::Real = 2.0, max_steps::Int = 32)
    # Vertical level
    lf0 = logf(x0)
    y = lf0 - randexp()

    # Step out
    L = x0 - w * rand()
    R = L + w
    for _ in 1:max_steps
        logf(L) >= y || break
        L -= w
    end
    for _ in 1:max_steps
        logf(R) >= y || break
        R += w
    end

    # Shrink
    for _ in 1:100  # safety limit
        x1 = L + rand() * (R - L)
        if logf(x1) >= y
            return x1
        end
        if x1 < x0
            L = x1
        else
            R = x1
        end
    end
    return x0  # fallback
end

# ============================================================
# Main Gibbs sampler
# ============================================================

struct GibbsResult
    w_samples::Vector{Matrix{Float64}}   # [sweep][expert, dim]
    τ_samples::Vector{Vector{Float64}}   # [sweep][expert]
    β_samples::Vector{Vector{Float64}}   # [sweep][expert]
    z_samples::Vector{Matrix{Float64}}   # [sweep][expert, obs]
    elapsed_seconds::Float64
    time_per_sweep::Float64
end

"""
Run the handcoded blocked Gibbs sampler.

Block 1: conjugate direct sampling for (w, τ, β) — NO MCMC, exact conditional.
Block 2: independent 1D slice sampling for each z[i,j].

Returns samples and timing.
"""
function run_gibbs(
    y, features, predictions;
    n_experts::Int,
    d::Int,
    n_warmup::Int = 200,
    n_samples::Int = 500,
    rng = Random.default_rng(),
)
    N = length(y)
    n_total = n_warmup + n_samples

    # Initialize from prior-ish values
    w = [randn(rng, d) * 0.1 for _ in 1:n_experts]
    τ = [100.0 for _ in 1:n_experts]
    β = [0.01 for _ in 1:n_experts]
    z = zeros(n_experts, N)
    # Initialize z from softdot mean
    for i in 1:n_experts, j in 1:N
        z[i, j] = dot(w[i], features[j])
    end

    # Storage (only post-warmup)
    w_samples = Vector{Matrix{Float64}}(undef, n_samples)
    τ_samples = Vector{Vector{Float64}}(undef, n_samples)
    β_samples = Vector{Vector{Float64}}(undef, n_samples)
    z_samples = Vector{Matrix{Float64}}(undef, n_samples)

    t0 = time()

    for sweep in 1:n_total
        # === Block 1: Sample globals from conjugate conditionals ===
        for i in 1:n_experts
            z_i = @view z[i, :]
            w[i] = sample_w_conditional(z_i, features, τ[i])
            τ[i] = sample_τ_conditional(z_i, w[i], features)
            β[i] = sample_β_conditional(z_i)
        end

        # === Block 2: Sample each z[i,j] independently via slice sampling ===
        for j in 1:N
            for i in 1:n_experts
                logf = z_val -> z_logdensity(
                    z_val, w[i], τ[i], β[i], y[j], features[j], predictions[i, j],
                )
                z[i, j] = slice_sample(logf, z[i, j])
            end
        end

        # Store post-warmup samples
        if sweep > n_warmup
            idx = sweep - n_warmup
            w_samples[idx] = reduce(hcat, w)'  |> collect  # n_experts × d
            τ_samples[idx] = copy(τ)
            β_samples[idx] = copy(β)
            z_samples[idx] = copy(z)
        end
    end

    elapsed = time() - t0
    return GibbsResult(
        w_samples, τ_samples, β_samples, z_samples,
        elapsed, elapsed / n_total,
    )
end

"""
Compute basic diagnostics from Gibbs samples.
"""
function gibbs_diagnostics(result::GibbsResult, data)
    n_samples = length(result.w_samples)
    n_experts = size(result.w_samples[1], 1)
    d = size(result.w_samples[1], 2)

    # ESS for each w component (the mixing bottleneck)
    min_ess = Inf
    for i in 1:n_experts
        for k in 1:d
            samples = [result.w_samples[s][i, k] for s in 1:n_samples]
            ess_val = ess_simple(samples)
            min_ess = min(min_ess, ess_val)
        end
    end

    # Correctness: compare posterior means of w to ground truth
    w_correct = true
    if hasproperty(data, :w_true)
        for i in 1:n_experts
            w_mean = mean(result.w_samples[s][i, :] for s in 1:n_samples)
            w_std = std([result.w_samples[s][i, k] for s in 1:n_samples, k in 1:d], dims=1)[:]
            for k in 1:d
                if abs(w_mean[k] - data.w_true[i][k]) > 3.0 * max(w_std[k], 0.01)
                    w_correct = false
                end
            end
        end
    end

    return (; min_ess, w_correct, ess_per_s = min_ess / result.elapsed_seconds)
end

"""
Simple ESS estimate using autocorrelation (Geyer's initial monotone sequence).
"""
function ess_simple(x::AbstractVector)
    n = length(x)
    n < 4 && return Float64(n)

    x_centered = x .- mean(x)
    var_x = var(x; corrected=false)
    var_x < 1e-30 && return 1.0

    # Compute autocorrelations up to lag n/2
    max_lag = n ÷ 2
    sum_rho = 0.0
    for lag in 1:max_lag
        rho = 0.0
        for t in 1:(n - lag)
            rho += x_centered[t] * x_centered[t + lag]
        end
        rho /= (n * var_x)
        if rho < 0.05
            break
        end
        sum_rho += rho
    end

    return n / (1.0 + 2.0 * sum_rho)
end
