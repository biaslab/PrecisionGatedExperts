using Turing
using LinearAlgebra
using Distributions

"""
PGE ensemble model with blocked Gibbs sampler configuration.

Same joint distribution as `pge_ensemble` in turing_model.jl, but designed
for Turing's Gibbs compositor with two blocks:

  Block 1 (globals | locals): Sample (w, τ, β | z).
    - w-conditional is conjugate: Gaussian prior × Gaussian softdot → Gaussian posterior
    - τ-conditional is conjugate: Gamma prior × Normal precision → Gamma posterior
    - β-conditional is conjugate: Gamma prior × Gamma rate → Gamma posterior
    These are the SAME updates as the VMP messages to w, τ, β.

  Block 2 (locals | globals): Sample each z[i,j] independently.
    - 1D non-conjugate: Normal(softdot) × Gamma-on-exp(z) × Normal(likelihood)
    - Solvable by slice sampling or adaptive MH.
    This mirrors the per-edge fixed-point solve of Theorem 2.

The structural equivalence (Gibbs ≈ VMP + sampling noise) is the point.
"""
@model function pge_ensemble_gibbs(y, features, predictions, n_experts, d)
    N = length(y)

    # --- Block 1: Global parameters (conjugate conditionals) ---
    w = Matrix{Real}(undef, n_experts, d)
    τ = Vector{Real}(undef, n_experts)
    β = Vector{Real}(undef, n_experts)

    for i in 1:n_experts
        w[i, :] ~ MvNormal(zeros(d), (1.0 / 0.01) * I)
        τ[i] ~ Gamma(1.0, 1.0 / 1e-3)
        β[i] ~ Gamma(1.0, 1.0 / 1e3)
    end

    # --- Block 2: Local latent variables (1D non-conjugate each) ---
    z = Matrix{Real}(undef, n_experts, N)

    for j in 1:N
        for i in 1:n_experts
            z[i, j] ~ Normal(dot(w[i, :], features[j]), 1.0 / sqrt(τ[i]))
            Turing.@addlogprob!(z[i, j] - β[i] * exp(z[i, j]))
            Turing.@addlogprob!(logpdf(
                Normal(predictions[i, j], 1.0 / sqrt(exp(z[i, j]))),
                y[j],
            ))
        end
    end
end

"""
Build the Gibbs sampler with block structure exploiting conditional independence.

Block 1: HMC for globals (low-dim, 7×65 + 7 + 7 = 469+14 params)
Block 2: MH for locals (each z[i,j] is 1D non-conjugate)

The key insight: Block 1's conjugate updates are identical to VMP messages.
Gibbs just adds sampling noise on top.
"""
function build_gibbs_sampler(; n_leapfrog=10, step_size=0.05)
    return Gibbs(
        (@varname(w), @varname(τ), @varname(β)) => HMC(step_size, n_leapfrog),
        @varname(z) => MH(),
    )
end
