"""
Multivariate Probabilistic Ensemble Forecasting with RxInfer

This example demonstrates how to create a multivariate probabilistic ensemble model
using RxInfer where multiple forecasting functions are combined with learned precision weights.

## Mathematical Model

Given n forecasting functions f₁, ..., fₙ producing d-dimensional outputs, we model:

    p(y, γ₁, ..., γₙ | x) = p(γ₁)⋯p(γₙ) ∏ⱼ ∏ᵢ N(yⱼ | fᵢ(xⱼ), (1/γᵢ)G⁻¹)

where:
- y is the d-dimensional target observation
- x is the input (treated as fixed/known)
- fᵢ(x) is the d-dimensional prediction from the i-th forecasting function
- γᵢ is the scalar precision of the i-th function (shared across all dimensions)
- G is a fixed precision matrix (we use G = I, identity)
- Each γᵢ has a Gamma prior: γᵢ ~ Gamma(α₀, β₀)

Using MvNormalMeanScalePrecision(μ, γ, G) gives: y ~ N(μ, (1/γ)G⁻¹)
With G = I: y ~ N(μ, (1/γ)I) - isotropic covariance

## Key Insight: Conjugate Update

For each forecaster i with N validation observations (d-dimensional each):

    γᵢ | data ~ Gamma(α₀ + Nd/2, β₀ + SSE_i/2)

where SSE_i = ∑ⱼ ||yⱼ - fᵢ(xⱼ)||² (sum of squared errors across all dimensions)

## Ensemble Prediction

The optimal ensemble prediction uses precision-weighted averaging:

    ŷ_ensemble = (∑ᵢ E[γᵢ] * fᵢ(x)) / (∑ᵢ E[γᵢ])

with uncertainty:

    Cov[y_ensemble] ≈ (1 / ∑ᵢ E[γᵢ]) * I

This automatically downweights unreliable forecasters!
"""

using RxInfer
using Distributions
using Random
using Statistics
using LinearAlgebra
using Plots

# Include utilities from the main package
using ProbabilisticEnsembling


@model function mv_ensemble_precision_model(n_forecasters, n_samples, means, y, priors)
    local γ

    for i in 1:n_forecasters
        γ[i] ~ priors[i]
    end

    for i in 1:n_forecasters
        for j in 1:n_samples
            # means is indexed as means[(i-1)*n_samples + j] to get the j-th sample for forecaster i
            idx = (i - 1) * n_samples + j
            y[j] ~ MvNormalMeanScalePrecision(means[idx], γ[i])
        end
    end
end

"""
    demo_synthetic_ensemble()

Demonstrate multivariate ensemble with synthetic forecasters on sine wave data.
"""
function demo_synthetic_ensemble()
    Random.seed!(42)

    @info "Multivariate Probabilistic Ensemble Forecasting Demo (Synthetic Data)"

    # Generate data
    n_train = 200
    n_test = 100
    d = 3  # output dimension
    noise_level = 0.1

    x_train = range(0, 4π, length=n_train)
    # True function: [sin(x) + cos(x), cos(x) - sin(x), sin(2x)]
    y_train = [[sin(x) + cos(x), cos(x) - sin(x), sin(2x)] .+ noise_level .* randn(d) for x in x_train]

    x_test = range(0, 4π, length=n_test)
    y_test = [[sin(x) + cos(x), cos(x) - sin(x), sin(2x)] .+ noise_level .* randn(d) for x in x_test]

    # Define forecasters with varying quality
    # Note: true function is [sin(x)+cos(x), cos(x)-sin(x), sin(2x)], so most are imperfect
    forecasters = [
        ("sin(x)", x -> [sin(x), sin(x), sin(x)]),
        ("sin+0.3", x -> [sin(x) + 0.3, sin(x) + 0.3, sin(x) + 0.3]),
        ("0.8sin", x -> [0.8sin(x), 0.8sin(x), 0.8sin(x)]),
        ("linear", x -> [0.5x, 0.5x, 0.5x]),
        ("cos(x)", x -> [cos(x), cos(x), cos(x)]),
        ("shifted", x -> [cos(x - 0.5), cos(x - 0.5), cos(x - 0.5)]),
    ]

    n_forecasters = length(forecasters)
    # Shape: (n_forecasters, d, n_samples)
    predictions_train = zeros(n_forecasters, d, n_train)
    predictions_test = zeros(n_forecasters, d, n_test)
    for (i, (_, f)) in enumerate(forecasters)
        for (j, x) in enumerate(x_train)
            predictions_train[i, :, j] = f(x)
        end
        for (j, x) in enumerate(x_test)
            predictions_test[i, :, j] = f(x)
        end
    end

    @info "Data Summary" n_train n_test output_dim=d true_function="[sin+cos, cos-sin, sin(2x)] + noise" noise_level

    # =========================================================================
    # Step 1: Infer precisions using RxInfer
    # =========================================================================
    @info "Step 1: Learning Precision Parameters from Training Data"

    # Convert predictions to flattened vector of vectors for the model
    # means_train[(i-1)*n_train + j] = prediction from forecaster i for sample j
    means_train = Vector{Vector{Float64}}(undef, n_forecasters * n_train)
    for i in 1:n_forecasters
        for j in 1:n_train
            means_train[(i-1)*n_train + j] = predictions_train[i, :, j]
        end
    end

    # Vague priors for precision
    starting_priors = [GammaShapeRate(1.0, 1e-12) for _ in 1:n_forecasters]

    result = infer(
        model = mv_ensemble_precision_model(n_forecasters=n_forecasters, n_samples=n_train, priors=starting_priors),
        data = (y = y_train, means = means_train),
        iterations = 20
    )

    γ_posteriors = result.posteriors[:γ][end]
    γ_means = map(mean, γ_posteriors)

    @info "Learned Precision Posteriors (higher = more reliable)"
    for (i, (name, _)) in enumerate(forecasters)
        # Compute mean squared error across all dimensions
        mse_train = mean([sum((predictions_train[i, :, j] .- y_train[j]).^2) for j in 1:n_train]) / d
        @info "Forecaster $i" name E_γ=round(γ_means[i], digits=4) train_MSE=round(mse_train, digits=4)
    end

    # Compute normalized weights
    weights = γ_means ./ sum(γ_means)
    @info "Ensemble Weights (normalized precisions)"
    for (i, (name, _)) in enumerate(forecasters)
        @info "Weight $i" name weight=round(weights[i], digits=4)
    end

    # =========================================================================
    # Step 2: Generate ensemble predictions on test data
    # =========================================================================
    @info "Step 2: Generating Ensemble Predictions on Test Data"

    # Convert test predictions to flattened vector of vectors
    means_test = Vector{Vector{Float64}}(undef, n_forecasters * n_test)
    for i in 1:n_forecasters
        for j in 1:n_test
            means_test[(i-1)*n_test + j] = predictions_test[i, :, j]
        end
    end

    prediction_array = [missing for _ in 1:n_test]
    infer_predict = infer(
        model = mv_ensemble_precision_model(n_forecasters=n_forecasters, n_samples=n_test, priors=γ_posteriors),
        data = (y = prediction_array, means = means_test),
        iterations = 1
    )

    # Extract ensemble predictions
    ensemble_predictions = infer_predict.predictions[:y][end]
    ensemble_mean = [mean(p) for p in ensemble_predictions]
    ensemble_cov = [cov(p) for p in ensemble_predictions]

    # Extract scalar uncertainties (average diagonal of covariance)
    ensemble_std = [sqrt(mean(diag(c))) for c in ensemble_cov]

    @show mean(ensemble_std)

    # =========================================================================
    # Step 3: Compare all methods
    # =========================================================================
    @info "Step 3: Performance Comparison on Test Data"

    # Compute MSE for each individual forecaster
    @info "Individual Forecaster Performance (Test Set)"
    individual_mses = Float64[]
    for (i, (name, _)) in enumerate(forecasters)
        mse_i = mean([sum((predictions_test[i, :, j] .- y_test[j]).^2) for j in 1:n_test]) / d
        push!(individual_mses, mse_i)
        @info "Forecaster $i" name test_MSE=round(mse_i, digits=6)
    end

    # Simple average baseline
    simple_avg = [vec(mean(predictions_test[:, :, j], dims=1)) for j in 1:n_test]
    simple_avg_mse = mean([sum((simple_avg[j] .- y_test[j]).^2) for j in 1:n_test]) / d

    # Ensemble MSE
    ensemble_mse = mean([sum((ensemble_mean[j] .- y_test[j]).^2) for j in 1:n_test]) / d

    # Best individual
    best_idx = argmin(individual_mses)
    best_mse = individual_mses[best_idx]

    @info "Method Comparison" simple_avg_MSE=round(simple_avg_mse, digits=6) best_individual_MSE=round(best_mse, digits=6) best_forecaster=forecasters[best_idx][1] ensemble_MSE=round(ensemble_mse, digits=6)

    # Improvement metrics
    improvement_vs_avg = (simple_avg_mse - ensemble_mse) / simple_avg_mse * 100
    improvement_vs_best = (best_mse - ensemble_mse) / best_mse * 100

    @info "Improvements" vs_simple_avg="$(round(improvement_vs_avg, digits=1))%" vs_best_individual="$(round(improvement_vs_best, digits=1))%"

    # =========================================================================
    # Step 4: Visualization
    # =========================================================================
    @info "Step 4: Generating Visualization"

    # Convert predictions to matrices for plotting
    y_test_mat = hcat(y_test...)'  # n_test x d
    ensemble_mean_mat = hcat(ensemble_mean...)'  # n_test x d
    simple_avg_mat = hcat(simple_avg...)'  # n_test x d

    # Plot 1: Predictions comparison (showing first dimension as representative)
    p1 = plot(x_test, y_test_mat[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Ensemble vs Individual Predictions (Dim 1)",
        xlabel="x", ylabel="y",
        legend=:topright, size=(600, 400)
    )

    # Add ensemble with uncertainty ribbon
    plot!(p1, x_test, ensemble_mean_mat[:, 1],
        ribbon=2*ensemble_std,
        label="Ensemble ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )

    # Add individual forecasters
    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan]
    for i in 1:n_forecasters
        plot!(p1, x_test, predictions_test[i, 1, :],
            label=forecasters[i][1],
            ls=:dash, alpha=0.6, color=colors[i]
        )
    end

    # Plot 2: Weights bar chart
    p2 = bar(1:n_forecasters, weights,
        title="Learned Ensemble Weights",
        xlabel="Forecaster", ylabel="Weight",
        xticks=(1:n_forecasters, [f[1] for f in forecasters]),
        legend=false, color=:steelblue,
        xrotation=45, size=(600, 400)
    )

    # Plot 3: MSE comparison bar chart
    all_mses = vcat(individual_mses, [simple_avg_mse, ensemble_mse])
    all_labels = vcat([f[1] for f in forecasters], ["Simple Avg", "Ensemble"])
    bar_colors = vcat(fill(:gray, n_forecasters), [:orange, :green])

    p3 = bar(1:length(all_mses), all_mses,
        title="MSE Comparison (Test Set, avg over dims)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45, size=(600, 400)
    )

    # Plot 4: Prediction errors over x (first dimension)
    p4 = plot(title="Prediction Errors vs True Values (Dim 1)",
        xlabel="x", ylabel="Error (pred - true)",
        legend=:topright, size=(600, 400)
    )
    plot!(p4, x_test, ensemble_mean_mat[:, 1] .- y_test_mat[:, 1], label="Ensemble", lw=2, color=:blue)
    plot!(p4, x_test, simple_avg_mat[:, 1] .- y_test_mat[:, 1], label="Simple Avg", lw=1, color=:orange, ls=:dash)
    plot!(p4, x_test, predictions_test[best_idx, 1, :] .- y_test_mat[:, 1],
        label="Best Individual", lw=1, color=:green, ls=:dot)
    hline!(p4, [0], color=:black, ls=:dash, label="", alpha=0.5)

    # Combine plots
    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))

    savefig(plt, "ensemble_multivariate_synthetic_demo.png")
    @info "Saved visualization" file="ensemble_multivariate_synthetic_demo.png"

    # Return results
    return (
        γ_posteriors = γ_posteriors,
        weights = weights,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_mse = ensemble_mse,
        simple_avg_mse = simple_avg_mse,
        individual_mses = individual_mses,
        forecasters = forecasters
    )
end

function main()
    demo_synthetic_ensemble()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
