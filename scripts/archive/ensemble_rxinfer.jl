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

    @info "Probabilistic Ensemble Forecasting Demo (Synthetic Data)"

    # Generate data
    n_train = 200
    n_test = 100
    noise_level = 0.1

    x_train = range(0, 4π, length=n_train)
    y_train = sin.(x_train) + cos.(x_train) .+ noise_level .* randn(n_train)

    x_test = range(0, 4π, length=n_test)
    y_test = sin.(x_test) + cos.(x_test) .+ noise_level .* randn(n_test)

    # Define forecasters with varying quality
    # Note: true function is sin(x) + cos(x), so all are imperfect
    forecasters = [
        ("sin(x)", x -> sin(x)),
        ("sin+0.3", x -> sin(x) + 0.3),
        ("0.8sin", x -> 0.8 * sin(x)),
        ("linear", x -> 0.5 * x),
        ("cos(x)", x -> cos(x)),
        ("shifted cos", x -> cos.(x .- 0.5)),
    ]

    n_forecasters = length(forecasters)
    predictions_train = zeros(n_forecasters, n_train)
    predictions_test = zeros(n_forecasters, n_test)
    for (i, (_, f)) in enumerate(forecasters)
        predictions_train[i, :] = f.(x_train)
        predictions_test[i, :] = f.(x_test)
    end

    @info "Data Summary" n_train n_test true_function="sin(x) + cos(x) + noise" noise_level

    # =========================================================================
    # Step 1: Infer precisions using RxInfer
    # =========================================================================
    @info "Step 1: Learning Precision Parameters from Training Data"

    starting_priors = [GammaShapeRate(1, 1e-12) for _ in 1:n_forecasters]
    result = infer(
        model = ensemble_precision_model(n_forecasters=n_forecasters, priors=starting_priors),
        data = (y = y_train, X = predictions_train),
        iterations = 20
    )

    γ_posteriors = result.posteriors[:γ][end]
    γ_means = map(mean, γ_posteriors)

    @info "Learned Precision Posteriors (higher precision = more reliable)"
    for (i, (name, _)) in enumerate(forecasters)
        mse_train = mean((predictions_train[i, :] .- y_train).^2)
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

    prediction_array = [missing for _ in 1:n_test]
    infer_predict = infer(
        model = ensemble_precision_model(n_forecasters=n_forecasters, priors=γ_posteriors),
        data = (y = prediction_array, X = predictions_test),
        iterations = 1
    )

    # Extract ensemble predictions (mean and std)
    ensemble_predictions = infer_predict.predictions[:y][end]
    ensemble_mean = map(mean, ensemble_predictions)
    ensemble_std = map(std, ensemble_predictions)

    @show mean(ensemble_std)

    # =========================================================================
    # Step 3: Compare all methods
    # =========================================================================
    @info "Step 3: Performance Comparison on Test Data"

    # Compute MSE for each individual forecaster
    @info "Individual Forecaster Performance (Test Set)"
    individual_mses = Float64[]
    for (i, (name, _)) in enumerate(forecasters)
        mse_i = mean((predictions_test[i, :] .- y_test).^2)
        push!(individual_mses, mse_i)
        @info "Forecaster $i" name test_MSE=round(mse_i, digits=6)
    end

    # Simple average baseline
    simple_avg = vec(mean(predictions_test, dims=1))
    simple_avg_mse = mean((simple_avg .- y_test).^2)

    # Ensemble MSE
    ensemble_mse = mean((ensemble_mean .- y_test).^2)

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

    # Plot 1: Predictions comparison
    p1 = plot(x_test, y_test,
        label="True (sin+cos)", lw=2, color=:black, ls=:dot,
        title="Ensemble vs Individual Predictions",
        xlabel="x", ylabel="y",
        legend=:topright, size=(600, 400)
    )

    # Add ensemble with uncertainty ribbon
    plot!(p1, x_test, ensemble_mean,
        ribbon=2*ensemble_std,
        label="Ensemble ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )

    # Add individual forecasters
    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan]
    for i in 1:n_forecasters
        plot!(p1, x_test, predictions_test[i, :],
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
        title="MSE Comparison (Test Set)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45, size=(600, 400)
    )

    # Plot 4: Prediction errors over x
    p4 = plot(title="Prediction Errors vs True Values",
        xlabel="x", ylabel="Error (pred - true)",
        legend=:topright, size=(600, 400)
    )
    plot!(p4, x_test, ensemble_mean .- y_test, label="Ensemble", lw=2, color=:blue)
    plot!(p4, x_test, simple_avg .- y_test, label="Simple Avg", lw=1, color=:orange, ls=:dash)
    plot!(p4, x_test, predictions_test[best_idx, :] .- y_test,
        label="Best Individual", lw=1, color=:green, ls=:dot)
    hline!(p4, [0], color=:black, ls=:dash, label="", alpha=0.5)

    # Combine plots
    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))

    savefig(plt, "ensemble_synthetic_demo.png")
    @info "Saved visualization" file="ensemble_synthetic_demo.png"

    # Return results for further analysis
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
