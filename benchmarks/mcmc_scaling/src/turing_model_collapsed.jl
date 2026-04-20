using Turing
using LinearAlgebra
using Distributions
using LogExpFunctions: logsumexp

"""
Gauss-Hermite nodes/weights via Golub-Welsch.
Returns (nodes, weights) for ∫ f(x) exp(-x²) dx ≈ Σ w_k f(x_k).
"""
function gausshermite_gw(K::Int)
    β = [sqrt(n / 2) for n in 1:(K - 1)]
    T = SymTridiagonal(zeros(K), β)
    F = eigen(T)
    nodes = F.values
    weights = sqrt(π) .* (F.vectors[1, :]) .^ 2
    return nodes, weights
end

"""
Collapsed PGE ensemble: z[i,j] is marginalized via 1D Gauss-Hermite.

NUTS sees only (w, τ, β). Dimension = n_experts * (d + 2), independent of N.

For each (i, j), compute
  log ∫ p(y_j | exp(z)) · β_i · exp(-β_i exp(z)) · N(z | w_i'φ_j, 1/τ_i) · exp(z) dz
via K-point Gauss-Hermite substitution z = μ + √2 σ ξ, giving
  log Σ_k w_k/√π · [p(y_j | exp(z_k)) · β_i · exp(-β_i exp(z_k)) · exp(z_k)]
"""
@model function pge_ensemble_collapsed(
    y, features, predictions, n_experts::Int, d::Int, gh_nodes, gh_weights;
    prior_prec_w = 0.01, prior_rate_τ = 1e-3, prior_rate_β = 1e3,
)
    N = length(y)
    K = length(gh_nodes)

    # Priors (vectorized, concretely typed)
    w ~ filldist(MvNormal(zeros(d), (1/prior_prec_w) * I), n_experts)  # d × n_experts
    τ ~ filldist(Gamma(1.0, 1.0 / prior_rate_τ), n_experts)
    β ~ filldist(Gamma(1.0, 1.0 / prior_rate_β), n_experts)

    # μ[i,j] = w_i' φ_j, as one matmul
    Φ = reduce(hcat, features)                                # d × N
    μ = transpose(w) * Φ                                      # n_experts × N

    logsqrtπ = 0.5 * log(π)

    for j in 1:N
        for i in 1:n_experts
            σ = 1.0 / sqrt(τ[i])
            μ_ij = μ[i, j]
            pred_ij = predictions[i, j]
            y_j = y[j]
            β_i = β[i]

            # Build log-integrand at each GH node
            terms = map(1:K) do k
                ξ = gh_nodes[k]
                z = μ_ij + sqrt(2.0) * σ * ξ
                γ = exp(z)
                # log p(y|pred, 1/γ) = 0.5 log(γ/2π) - 0.5 γ (y-pred)²
                ll_y = 0.5 * z - 0.5 * log(2π) - 0.5 * γ * (y_j - pred_ij)^2
                # log [β · exp(-β γ) · γ] = log β - β γ + z
                ll_g = log(β_i) - β_i * γ + z
                # GH weight normalization: w_k / √π (reference N(0,1) → standard Normal)
                log(gh_weights[k]) - logsqrtπ + ll_y + ll_g
            end

            Turing.@addlogprob! logsumexp(terms)
        end
    end
end
