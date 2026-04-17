using Turing
using LinearAlgebra
using Distributions

"""
PGE ensemble model for joint NUTS sampling.

Translates the RxInfer factor graph:
  z[i,j] ~ softdot(features[j], w[i], τ[i])   → Normal(w'φ, 1/τ)
  γ[i,j] ~ GammaShapeRate(1.0, β[i])           → Gamma prior on γ
  z[i,j] ~ Log(γ[i,j])                          → exp-link: γ = exp(z)
  y[j]   ~ NormalMeanPrecision(pred[i,j], γ[i,j]) → likelihood via equality node

The z[i,j] ~ Normal(...) statement contributes the softdot log-density.
The first @addlogprob!: Gamma(1, β) prior on γ = exp(z), with Jacobian.
  log p_Gamma(exp(z)) + log|Jacobian| = (1-1)*z - β*exp(z) + z = z - β*exp(z) + const
The second @addlogprob!: each expert's likelihood on shared y[j].
"""
@model function pge_ensemble(y, features, predictions, n_experts, d)
    N = length(y)

    # Global parameters (same priors as YAML config)
    w = Matrix{Real}(undef, n_experts, d)
    τ = Vector{Real}(undef, n_experts)
    β = Vector{Real}(undef, n_experts)

    for i in 1:n_experts
        w[i, :] ~ MvNormal(zeros(d), (1.0 / 0.01) * I)  # MvNormalMeanScalePrecision(scale=0.01) → cov = 100·I
        τ[i] ~ Gamma(1.0, 1.0 / 1e-3)                     # GammaShapeRate(1, 1e-3) → Gamma(shape=1, scale=1000)
        β[i] ~ Gamma(1.0, 1.0 / 1e3)                       # GammaShapeRate(1, 1e3) → Gamma(shape=1, scale=0.001)
    end

    # Local latent variables
    z = Matrix{Real}(undef, n_experts, N)

    for j in 1:N
        for i in 1:n_experts
            # Factor 1 (softdot): z ~ Normal(w'φ, 1/τ)
            z[i, j] ~ Normal(dot(w[i, :], features[j]), 1.0 / sqrt(τ[i]))

            # Factor 2 (Gamma(1,β) prior on γ=exp(z), change of variables)
            # log Gamma(exp(z); shape=1, rate=β) + log|Jacobian=exp(z)|
            # = (1-1)*log(exp(z)) - β*exp(z) + log(β) + z
            # = z - β*exp(z) + const
            Turing.@addlogprob!(z[i, j] - β[i] * exp(z[i, j]))

            # Factor 3 (likelihood: expert i contributes to shared y[j])
            Turing.@addlogprob!(logpdf(
                Normal(predictions[i, j], 1.0 / sqrt(exp(z[i, j]))),
                y[j],
            ))
        end
    end
end
