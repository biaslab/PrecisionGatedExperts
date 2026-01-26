"""
Dynamic Probabilistic Ensemble Forecasting with RxInfer

This script demonstrates how to create a dynamic probabilistic ensemble model using RxInfer
where the weight of each forecaster depends on the input features. This allows the ensemble
to automatically trust different forecasters in different regions of input space.

## Mathematical Model

We make precision depend on input features through a linear-exponential link:

    z_i(x) = w_i^T x           (linear projection)
    γ_i(x) = exp(z_i(x))       (ensures positivity)

The full generative model:

    w_i ~ N(0, Σ_w)                           for i = 1, ..., n
    z_{i,j} = w_i^T x_j                       (deterministic)
    γ_{i,j} = exp(z_{i,j})
    y_j ~ N(f_i(x_j), γ_{i,j}^{-1})           for all i, j

## Why This Works

- If w_i^T x is large positive → γ_i(x) is large → forecaster i gets high weight
- If w_i^T x is large negative → γ_i(x) ≈ 0 → forecaster i is effectively ignored
- The gating is learned from data: forecasters that perform well in certain regions
  will have w_i that activates them there

## Comparison with Static Ensemble

| Aspect              | Static Ensemble      | Dynamic Ensemble                    |
|---------------------|----------------------|-------------------------------------|
| Precision           | γ_i (global)         | γ_i(x) = exp(w_i^T x)               |
| Weights             | Constant everywhere  | Input-dependent                     |
| Parameters          | n precisions         | n × d weight coefficients           |
| Non-conjugacy       | None                 | Log link requires projection        |
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Random
using Statistics
using LinearAlgebra
using Plots

# Include utilities from the main package
using ProbabilisticEnsembling

# =============================================================================
# Dynamic Ensemble Model Definition
# =============================================================================

@model function dynamic_ensemble_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors)
    # features: Vector of vectors, each of length n_features
    # predictions: [n_forecasters × n_obs] - forecaster outputs
    # w_priors: vector of MvNormal priors for gating weights

    local w, z, γ, τ

    # Gating weights for each forecaster
    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
    end

    # For each observation
    for j in 1:n_obs
        # For each forecaster
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])

            # γ = exp(z) via Log node: z = Log(γ)
            # Use GammaShapeScale for CFE compatibility
            γ[i, j] ~ GammaShapeScale(1.0, 1.0)
            z[i, j] ~ Log(γ[i, j])

            # Likelihood: observation given forecaster i's prediction
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function dynamic_ensemble_constraints()
    # Mean-field factorization: all variables factorize
    q(w, z, γ, τ) = q(w)q(z)q(γ)q(τ)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ) :: ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

@initialization function dynamic_ensemble_init()
    q(w) = MvNormalMeanPrecision(zeros(2), diagm(ones(2)))
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
end

# =============================================================================
# Static Ensemble Model (for comparison)
# =============================================================================

@model function static_ensemble_model(n_forecasters, predictions, y, priors)
    local γ

    for i in 1:n_forecasters
        γ[i] ~ priors[i]
    end

    for i in 1:n_forecasters
        for j in eachindex(y)
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i])
        end
    end
end

# =============================================================================
# Feature Design Functions
# =============================================================================

"""
    create_features(x, feature_type::Symbol)

Create feature vectors from raw input x.

Available feature types:
- :raw       - Just x (linear dependence)
- :bias      - [1, x] (baseline + linear)
- :quadratic - [1, x, x²] (quadratic dependence)
- :trig      - [1, sin(x), cos(x)] (periodic features)
"""
function create_features(x, feature_type::Symbol)
    n = length(x)

    if feature_type == :raw
        return [Float64[xi] for xi in x]
    elseif feature_type == :bias
        return [Float64[1.0, xi] for xi in x]
    elseif feature_type == :quadratic
        return [Float64[1.0, xi, xi^2] for xi in x]
    elseif feature_type == :trig
        return [Float64[1.0, sin(xi), cos(xi)] for xi in x]
    else
        error("Unknown feature type: $feature_type")
    end
end

# =============================================================================
# Demo Function
# =============================================================================

"""
    demo_dynamic_ensemble()

Demonstrate the dynamic ensemble inference with synthetic data.
"""
function demo_dynamic_ensemble(true_function, feature_type=:bias)
    Random.seed!(42)

    @info "Dynamic Probabilistic Ensemble Forecasting Demo"
    @info "================================================"

    # =========================================================================
    # Step 1: Generate synthetic data with region-dependent truth
    # =========================================================================
    @info "Step 1: Generating Synthetic Data with Region-Dependent Structure"

    n_train = 300
    noise_level = 0.15

    x_train = collect(range(0, 2π, length=n_train))
    y_train = true_function.(x_train) .+ noise_level .* randn(n_train)

    # Define forecasters
    forecasters = [
        ("sin(x)", x -> sin(x)),
        ("sin+0.3", x -> sin(x) + 0.3),
        ("0.8sin", x -> 0.8 * sin(x)),
        ("linear", x -> 0.5 * x),
        ("cos(x)", x -> cos(x)),
        # ("shifted cos", x -> cos.(x .- 0.5)),
    ]

    n_forecasters = length(forecasters)

    # Generate predictions
    predictions_train = zeros(n_forecasters, n_train)
    for (i, (_, f)) in enumerate(forecasters)
        predictions_train[i, :] = f.(x_train)
    end

    @info "Data Summary" n_train n_forecasters true_function="transition from sin to cos" noise_level
   
    # Create feature vectors for gating
    features_train = create_features(x_train, feature_type)
    n_features = length(features_train[1])

    @info "Feature Design" feature_type n_features

    # =========================================================================
    # Step 2: Train Dynamic Ensemble using RxInfer
    # =========================================================================
    @info "Step 2: Training Dynamic Ensemble using RxInfer"

    # Initial priors on gating weights
    w_priors_init = [MvNormalMeanPrecision(zeros(n_features), 0.1 * diagm(ones(n_features))) for _ in 1:n_forecasters]
    # Priors on τ (softdot precision)
    τ_priors_init = [ GammaShapeScale(1.0, 1.0)  for _ in 1:n_forecasters ]

    @info "Running variational inference with projection..."

    dynamic_result = infer(
        model = dynamic_ensemble_model(
            n_forecasters = n_forecasters,
            n_obs = n_train,
            w_priors = w_priors_init,
            τ_priors = τ_priors_init
        ),
        data = (y = y_train, features = features_train, predictions = predictions_train),
        constraints = dynamic_ensemble_constraints(),
        initialization = dynamic_ensemble_init(),
        iterations = 30,
        free_energy = true,
        showprogress = true
    )

    # Extract learned weight posteriors
    w_posteriors = dynamic_result.posteriors[:w][end]
    τ_posteriors = dynamic_result.posteriors[:τ][end]

    @info "Learned Gating Weight Posteriors"
    for (i, (name, _)) in enumerate(forecasters)
        w_mean = mean(w_posteriors[i])
        @info "  Forecaster $i" name w_bias=round(w_mean[1], digits=4) w_slope=round(w_mean[2], digits=4)
    end

    @info "Learned Softdot Precision Posteriors"
    for (i, (name, _)) in enumerate(forecasters)
        τ_means = mean(τ_posteriors[i])
        @info "  Forecaster higher level precision $i" name τ_mean=round(τ_means, digits=4)
    end

    # =========================================================================
    # Step 3: Generate predictions on test data
    # =========================================================================
    @info "Step 3: Generating Ensemble Predictions on Test Data"

    n_test = 50
    x_test = collect(range(0, 4π, length=n_test))
    y_test = true_function.(x_test) .+ noise_level .* randn(n_test)

    # Generate forecaster predictions for test data
    predictions_test = zeros(n_forecasters, n_test)
    for (i, (_, f)) in enumerate(forecasters)
        predictions_test[i, :] = f.(x_test)
    end

    # Create features for test data
    features_test = create_features(x_test, feature_type)
    y_missing = [missing for _ in 1:n_test]

    # Dynamic ensemble predictions using RxInfer
    dynamic_predict = infer(
        model = dynamic_ensemble_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            w_priors = w_posteriors,
            τ_priors = τ_posteriors
        ),
        data = (y = y_missing, features = features_test, predictions = predictions_test),
        constraints = dynamic_ensemble_constraints(),
        initialization = dynamic_ensemble_init(),
        iterations = 10
    )

    # Extract dynamic ensemble predictions
    dynamic_predictions = dynamic_predict.predictions[:y][end]
    dynamic_mean = map(mean, dynamic_predictions)
    dynamic_std = map(std, dynamic_predictions)

    @info "Dynamic Prediction Statistics" mean_std=round(mean(dynamic_std), digits=4)

    # =========================================================================
    # Step 4: Train and predict with Static Ensemble for comparison
    # =========================================================================
    @info "Step 4: Training Static Ensemble for Comparison"

    # Train static ensemble
    static_priors = [GammaShapeRate(1.0, 1e-12) for _ in 1:n_forecasters]
    static_result = infer(
        model = static_ensemble_model(n_forecasters=n_forecasters, priors=static_priors),
        data = (y = y_train, predictions = predictions_train),
        iterations = 20
    )
    γ_posteriors = static_result.posteriors[:γ][end]

    @info "Learned Static Precision Posteriors"
    for (i, (name, _)) in enumerate(forecasters)
        γ_mean = mean(γ_posteriors[i])
        @info "  Forecaster $i" name γ_mean=round(γ_mean, digits=4)
    end

    # Static ensemble predictions
    static_predict = infer(
        model = static_ensemble_model(n_forecasters=n_forecasters, priors=γ_posteriors),
        data = (y = y_missing, predictions = predictions_test),
        iterations = 1
    )
    static_predictions = static_predict.predictions[:y][end]
    static_mean = map(mean, static_predictions)
    static_std = map(std, static_predictions)

    @info "Static Prediction Statistics" mean_std=round(mean(static_std), digits=4)

    # =========================================================================
    # Step 5: Performance Comparison
    # =========================================================================
    @info "Step 5: Performance Comparison (Train: 0-2π, Test: 0-4π)"

    # Compute MSEs
    dynamic_mse = mean((dynamic_mean .- y_test).^2)
    static_mse = mean((static_mean .- y_test).^2)

    individual_mses = Float64[]
    for i in 1:n_forecasters
        mse_i = mean((predictions_test[i, :] .- y_test).^2)
        push!(individual_mses, mse_i)
    end

    simple_avg = vec(mean(predictions_test, dims=1))
    simple_avg_mse = mean((simple_avg .- y_test).^2)

    @info "MSE Comparison"
    for (i, (name, _)) in enumerate(forecasters)
        @info "  Forecaster $i" name MSE=round(individual_mses[i], digits=6)
    end
    @info "  Simple Average" MSE=round(simple_avg_mse, digits=6)
    @info "  Static Ensemble" MSE=round(static_mse, digits=6)
    @info "  Dynamic Ensemble" MSE=round(dynamic_mse, digits=6)

    # Compute improvements
    best_idx = argmin(individual_mses)
    best_mse = individual_mses[best_idx]
    best_name = forecasters[best_idx][1]

    improvement_vs_simple_avg = (simple_avg_mse - dynamic_mse) / simple_avg_mse * 100
    improvement_vs_static = (static_mse - dynamic_mse) / static_mse * 100
    improvement_vs_best = (best_mse - dynamic_mse) / best_mse * 100

    @info "Dynamic Ensemble Improvements"
    @info "  vs Simple Average" improvement="$(round(improvement_vs_simple_avg, digits=1))%"
    @info "  vs Static Ensemble" improvement="$(round(improvement_vs_static, digits=1))%"
    @info "  vs Best Individual ($best_name)" improvement="$(round(improvement_vs_best, digits=1))%"

    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    @info "Step 6: Generating Visualization"

    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan]

    # Plot 1: Predictions comparison
    p1 = plot(x_test, y_test,
        label="True", lw=2, color=:black, ls=:dot,
        title="Dynamic vs Static Ensemble (Train: 0-2π, Test: 0-4π)",
        xlabel="x", ylabel="y",
        legend=:topright
    )

    # Dynamic ensemble with uncertainty
    plot!(p1, x_test, dynamic_mean,
        ribbon=2*dynamic_std,
        label="Dynamic ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )

    # Static ensemble
    plot!(p1, x_test, static_mean,
        ribbon=2*static_std,
        label="Static ±2σ",
        lw=2, color=:red, fillalpha=0.2
    )

    # Add vertical line at training boundary
    vline!(p1, [2π], color=:gray, ls=:dash, label="Train boundary", lw=2)

    # Plot 2: Individual forecasters
    p2 = plot(x_test, y_test,
        label="True", lw=2, color=:black, ls=:dot,
        title="Individual Forecasters",
        xlabel="x", ylabel="y",
        legend=:topright
    )

    for (i, (name, _)) in enumerate(forecasters)
        plot!(p2, x_test, predictions_test[i, :],
            label=name, ls=:dash, alpha=0.7, color=colors[i]
        )
    end
    vline!(p2, [2π], color=:gray, ls=:dash, label="", lw=2)

    # Plot 3: Dynamic weights over x (computed from learned w)
    p3 = plot(title="Dynamic Precision Weights γ(x) = exp(w'x)",
        xlabel="x", ylabel="Precision γ",
        legend=:topright
    )

    for (i, (name, _)) in enumerate(forecasters)
        w_mean = mean(w_posteriors[i])
        # Compute γ(x) = exp(w'x) for each test point
        γ_values = [exp(dot(w_mean, f)) for f in features_test]
        plot!(p3, x_test, γ_values, label=name, lw=2, color=colors[i])
    end
    vline!(p3, [2π], color=:gray, ls=:dash, label="", lw=2)

    # Plot 4: Uncertainty comparison (dynamic vs static)
    p4 = plot(title="Uncertainty: Dynamic vs Static",
        xlabel="x", ylabel="Standard Deviation (σ)",
        legend=:topright
    )
    plot!(p4, x_test, dynamic_std, label="Dynamic σ", lw=2, color=:blue)
    plot!(p4, x_test, static_std, label="Static σ", lw=2, color=:red, ls=:dash)
    vline!(p4, [2π], color=:gray, ls=:dash, label="Train boundary", lw=2)

    # Plot 5: MSE comparison bar chart
    all_mses = vcat(individual_mses, [simple_avg_mse, static_mse, dynamic_mse])
    all_labels = vcat([f[1] for f in forecasters], ["Simple Avg", "Static", "Dynamic"])
    bar_colors = vcat(fill(:gray, n_forecasters), [:orange, :red, :blue])

    p5 = bar(1:length(all_mses), all_mses,
        title="MSE Comparison (Full Test Set 0-4π)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45
    )

    # Plot 6: Normalized weights over x
    p6 = plot(title="Normalized Dynamic Weights",
        xlabel="x", ylabel="Weight (normalized)",
        legend=:outerright
    )

    # Compute normalized weights for each test point
    normalized_weights = zeros(n_forecasters, n_test)
    for j in 1:n_test
        γ_values = zeros(n_forecasters)
        for i in 1:n_forecasters
            w_mean = mean(w_posteriors[i])
            z_ij = dot(w_mean, features_test[j])
            γ_values[i] = exp(z_ij)
        end
        normalized_weights[:, j] = γ_values ./ sum(γ_values)
    end

    for (i, (name, _)) in enumerate(forecasters)
        plot!(p6, x_test, normalized_weights[i, :], label=name, lw=2, color=colors[i])
    end
    vline!(p6, [2π], color=:gray, ls=:dash, label="", lw=2)

    # Combine plots (3x2 layout)
    plt = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 1200))

    savefig(plt, "dynamic_ensemble_demo.png")
    @info "Saved visualization" file="dynamic_ensemble_demo.png"

    return (
        w_posteriors = w_posteriors,
        τ_posteriors = τ_posteriors,
        γ_posteriors = γ_posteriors,
        forecasters = forecasters,
        dynamic_mean = dynamic_mean,
        dynamic_std = dynamic_std,
        static_mean = static_mean,
        static_std = static_std,
        dynamic_mse = dynamic_mse,
        static_mse = static_mse,
        individual_mses = individual_mses
    )
end

# True function: sin dominates in first half, cos in second half
function true_function(x)
    transition_point = 2π
    transition_width = π
    weight_cos = 0.5 * (1 + tanh((x - transition_point) / transition_width))
    weight_sin = 1 - weight_cos
    return weight_sin * sin(x) + weight_cos * cos(x)
end

function main()
    demo_dynamic_ensemble(true_function)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
