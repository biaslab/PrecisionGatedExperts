using Random
using LinearAlgebra
using Distributions

"""
Generate synthetic data from the PGE ensemble generative model.

Uses realistic (not prior-scale) parameter values so that z stays in [-3, 3]
and γ = exp(z) stays in [0.05, 20]. The Turing model then uses the original
broad priors — this ensures the problem is well-posed (not prior-dominated).
"""
function generate_synthetic_data(;
    N::Int,
    n_experts::Int = 7,
    d::Int = 65,
    rng = StableRNG(42),
    τ_range::Tuple{<:Real, <:Real} = (100.0, 2000.0),
    β_range::Tuple{<:Real, <:Real} = (0.001, 0.01),
)
    # Ground truth parameters (realistic scale, NOT prior scale)
    w_true = [randn(rng, d) * 0.1 for _ in 1:n_experts]
    τ_true = [rand(rng, Uniform(τ_range[1], τ_range[2])) for _ in 1:n_experts]
    β_true = [rand(rng, Uniform(β_range[1], β_range[2])) for _ in 1:n_experts]

    # Features: [1.0; randn(64)] mimics 65-dim VAE output
    features = [vcat(1.0, randn(rng, d - 1)) for _ in 1:N]

    # Expert predictions: sinusoidal patterns + noise (mimics forecaster outputs)
    predictions = zeros(n_experts, N)
    for i in 1:n_experts
        freq = rand(rng, Uniform(0.5, 3.0))
        phase = rand(rng, Uniform(0.0, 2π))
        for j in 1:N
            predictions[i, j] = sin(freq * j / N * 2π + phase) + randn(rng) * 0.1
        end
    end

    # Forward-sample latent variables and observations
    z_true = zeros(n_experts, N)
    γ_true = zeros(n_experts, N)
    y = zeros(N)

    for j in 1:N
        # Accumulate precision-weighted contributions from all experts
        total_precision = 0.0
        precision_weighted_mean = 0.0

        for i in 1:n_experts
            # softdot: z ~ Normal(w'φ, 1/τ)
            z_true[i, j] = dot(w_true[i], features[j]) + randn(rng) / sqrt(τ_true[i])

            # γ = exp(z), but also has Gamma(1, β) prior
            # Sample γ from its conditional: combine Gamma prior and exp-link
            # For data generation, just use γ = exp(z) (the deterministic link)
            γ_true[i, j] = exp(z_true[i, j])

            total_precision += γ_true[i, j]
            precision_weighted_mean += γ_true[i, j] * predictions[i, j]
        end

        # y is the equality-node observation: product of n_experts Gaussians
        # Product of Normal(pred[i,j], 1/γ[i,j]) gives Normal(weighted_mean, 1/total_prec)
        combined_mean = precision_weighted_mean / total_precision
        combined_var = 1.0 / total_precision
        y[j] = combined_mean + randn(rng) * sqrt(combined_var)
    end

    return (;
        y,
        features,
        predictions,
        n_experts,
        d,
        N,
        # Ground truth for correctness checks
        w_true,
        τ_true,
        β_true,
        z_true,
        γ_true,
    )
end
