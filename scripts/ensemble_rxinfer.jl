"""
Probabilistic Ensemble Forecasting with RxInfer

This example demonstrates how to create a probabilistic ensemble model using RxInfer
where multiple forecasting functions are combined with learned precision weights.

## Mathematical Model

Given n forecasting functions f₁, ..., fₙ, we model the relationship:

    p(y, γ₁, ..., γₙ | x) = p(γ₁)⋯p(γₙ) ∏ⱼ ∏ᵢ N(yⱼ | fᵢ(xⱼ), γᵢ⁻¹)

where:
- y is the target observation
- x is the input (treated as fixed/known)
- fᵢ(x) is the prediction from the i-th forecasting function
- γᵢ is the precision (inverse variance) of the i-th function
- Each γᵢ has a Gamma prior: γᵢ ~ Gamma(α₀, β₀)

## Key Insight: Conjugate Update

For each forecaster i with N validation observations, the posterior precision is:

    γᵢ | data ~ Gamma(α₀ + N/2, β₀ + SSE_i/2)

where SSE_i = ∑ⱼ(yⱼ - fᵢ(xⱼ))² is the sum of squared errors.

## Ensemble Prediction

The optimal ensemble prediction uses precision-weighted averaging:

    ŷ_ensemble = (∑ᵢ E[γᵢ] * fᵢ(x)) / (∑ᵢ E[γᵢ])

with uncertainty:

    Var[y_ensemble] ≈ 1 / (∑ᵢ E[γᵢ])

This automatically downweights unreliable forecasters!
"""

using RxInfer
using Distributions
using Random
using Statistics
using LinearAlgebra
using JLD2
using Lux
using Plots

# Include utilities from the main package
using ProbabilisticEnsembling


@model function ensemble_precision_model(n_forecasters, X, y, priors)
    local γ

    for i in 1:n_forecasters
        γ[i] ~ priors[i]
    end

    for i in 1:n_forecasters
        for j in 1:length(y)
            y[j] ~ NormalMeanPrecision(X[i, j], γ[i])
        end
    end
end

"""
    demo_synthetic_ensemble()

Demonstrate ensemble with synthetic forecasters on sine wave data.
"""
function demo_synthetic_ensemble()
    Random.seed!(42)
    println("="^70)
    println("  Probabilistic Ensemble Forecasting Demo (Synthetic Data)")
    println("="^70)

    # Generate data
    n_train = 200
    n_test = 100
    noise_level = 0.1

    x_train = range(0, 4π, length=n_train)
    y_train = sin.(x_train) + cos.(x_train) .+ noise_level .* randn(n_train)

    x_test = range(0, 4π, length=n_test)
    y_test = sin.(x_test) + cos.(x_test) .+ noise_level .* randn(n_test)

    # Define forecasters with varying quality
    forecasters = [
        ("Perfect sin(x)", x -> sin(x)),
        ("Biased +0.3", x -> sin(x) + 0.3),
        ("Scaled 0.8x", x -> 0.8 * sin(x)),
        ("Phase shift", x -> sin(x + 0.5)),
        ("Noisy", x -> sin(x) + 0.4 * randn()),
    ]


    n_forecasters = length(forecasters)
    predictions_train = zeros(n_forecasters, n_train)
    predictions_test = zeros(n_forecasters, n_test)
    for (i, (_, f)) in enumerate(forecasters)
        predictions_train[i, :] = f.(x_train)
        predictions_test[i, :] = f.(x_test)
    end

    # Infer precisions using RxInfer
    starting_priors = [GammaShapeRate(1.0, 1.0) for _ in 1:n_forecasters]  # α₀=2.0, β₀=1.0
    result = infer(
        model = ensemble_precision_model(n_forecasters=n_forecasters, priors=starting_priors),
        data = (y = y_train, X = predictions_train),
        iterations = 20
    )

    @info result.posteriors[:γ][end]
    @show map(mean, result.posteriors[:γ][end])

    # Compute ensemble predictions
    prediction_array = [missing for _ in 1:n_test]
    infer_predict = infer(
        model = ensemble_precision_model(n_forecasters=n_forecasters, priors=result.posteriors[:γ][end]),
        data = (y = prediction_array, X = predictions_test),
        iterations = 1
    )

    @show map(mean, infer_predict.predictions[:y][end])

end

function main()
    demo_synthetic_ensemble()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
