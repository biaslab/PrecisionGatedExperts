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
    q(w, z, γ, τ) = q(w)q(z, γ)q(τ)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ) :: ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

@initialization function dynamic_ensemble_init(n_features)
    q(w) = MvNormalMeanPrecision(zeros(n_features), diagm(ones(n_features)))
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
        return [Float64[1.0, xi, sin(xi), cos(xi)] for xi in x]
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

    n_train = 500
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
        ("constant_1", _ -> 1.0),
        # ("constant_10", _ -> 10.0),
        ("constant_100", _ -> 100.0),
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
    τ_priors_init = [ GammaShapeScale(1.0, 1e12)  for _ in 1:n_forecasters ]

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
        initialization = dynamic_ensemble_init(n_features),
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
        w_rounded = round.(w_mean, digits=4)
        @info "  Forecaster $i" name w=w_rounded
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
        initialization = dynamic_ensemble_init(n_features),
        iterations = 10
    )

    # Extract dynamic ensemble predictions
    dynamic_predictions = dynamic_predict.predictions[:y][end]
    dynamic_mean = map(mean, dynamic_predictions)
    dynamic_std = map(std, dynamic_predictions)

    # Extract γ posteriors for weight uncertainty visualization
    γ_dynamic_posteriors = dynamic_predict.posteriors[:γ][end]  # [n_forecasters, n_test] matrix of distributions

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
        iterations = 10
    )
    static_predictions = static_predict.predictions[:y][end]
    static_mean = map(mean, static_predictions)
    static_std = map(std, static_predictions)

    # Extract static γ posteriors (these are the same for all x since static model)
    γ_static_posteriors = γ_posteriors  # Vector of distributions, one per forecaster

    @info "Static Prediction Statistics" mean_std=round(mean(dynamic_std), digits=4)

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

    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan, :magenta, :yellow]

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

    # Plot 3: Dynamic weights over x with 95% CI from γ posteriors
    p3 = plot(title="Dynamic Precision Weights γ(x) (95% CI)",
        xlabel="x", ylabel="Precision γ",
        legend=:topright
    )

    for (i, (name, _)) in enumerate(forecasters)
        # Extract median and 95% CI using quantiles (proper for Gamma distributions)
        γ_medians = [quantile(γ_dynamic_posteriors[i, j], 0.5) for j in 1:n_test]
        γ_lower = [quantile(γ_dynamic_posteriors[i, j], 0.025) for j in 1:n_test]
        γ_upper = [quantile(γ_dynamic_posteriors[i, j], 0.975) for j in 1:n_test]

        plot!(p3, x_test, γ_medians,
            ribbon=(γ_medians .- γ_lower, γ_upper .- γ_medians),
            label=name, lw=2, color=colors[i], fillalpha=0.2)
    end
    vline!(p3, [2π], color=:gray, ls=:dash, label="", lw=2)

    # Plot 4: Uncertainty comparison (dynamic vs static) with 95% CI
    # Monte Carlo sampling to get uncertainty of the uncertainty
    n_mc_samples = 500

    # Dynamic: sample γ posteriors and compute prediction std for each sample
    dynamic_std_samples = zeros(n_mc_samples, n_test)
    for s in 1:n_mc_samples
        for j in 1:n_test
            # Sample γ for each forecaster and compute combined precision
            γ_samples = [rand(γ_dynamic_posteriors[i, j]) for i in 1:n_forecasters]
            # Prediction variance = weighted sum of individual variances
            # For precision-weighted combination: var = 1 / sum(γ)
            total_precision = sum(γ_samples)
            dynamic_std_samples[s, j] = 1.0 / sqrt(total_precision)
        end
    end
    # Use quantiles for 95% CI
    dynamic_std_median = [quantile(dynamic_std_samples[:, j], 0.5) for j in 1:n_test]
    dynamic_std_lower = [quantile(dynamic_std_samples[:, j], 0.025) for j in 1:n_test]
    dynamic_std_upper = [quantile(dynamic_std_samples[:, j], 0.975) for j in 1:n_test]

    # Static: sample γ posteriors (same for all x) and compute prediction std
    static_std_samples = zeros(n_mc_samples, n_test)
    for s in 1:n_mc_samples
        # Sample γ for each forecaster (constant across x)
        γ_samples = [rand(γ_static_posteriors[i]) for i in 1:n_forecasters]
        total_precision = sum(γ_samples)
        static_std_samples[s, :] .= 1.0 / sqrt(total_precision)
    end
    # Use quantiles for 95% CI
    static_std_median = [quantile(static_std_samples[:, j], 0.5) for j in 1:n_test]
    static_std_lower = [quantile(static_std_samples[:, j], 0.025) for j in 1:n_test]
    static_std_upper = [quantile(static_std_samples[:, j], 0.975) for j in 1:n_test]

    p4 = plot(title="Uncertainty: Dynamic vs Static (95% CI)",
        xlabel="x", ylabel="Standard Deviation (σ)",
        legend=:topright
    )
    plot!(p4, x_test, dynamic_std_median,
        ribbon=(dynamic_std_median .- dynamic_std_lower, dynamic_std_upper .- dynamic_std_median),
        label="Dynamic σ (95% CI)", lw=2, color=:blue, fillalpha=0.3)
    plot!(p4, x_test, static_std_median,
        ribbon=(static_std_median .- static_std_lower, static_std_upper .- static_std_median),
        label="Static σ (95% CI)", lw=2, color=:red, fillalpha=0.2)
    vline!(p4, [2π], color=:gray, ls=:dash, label="Train boundary", lw=2)

    # Plot 5: MSE comparison bar chart (only key methods)
    key_mses = [best_mse, simple_avg_mse, static_mse, dynamic_mse]
    key_labels = ["Best ($best_name)", "Simple Avg", "Static", "Dynamic"]
    key_colors = [:gray, :orange, :red, :blue]

    p5 = bar(1:length(key_mses), key_mses,
        title="MSE Comparison (Full Test Set 0-4π)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(key_mses), key_labels),
        legend=false, color=key_colors,
        xrotation=45
    )

    # Plot 6: Normalized weights over x with 95% CI (via Monte Carlo from γ posteriors)
    p6 = plot(title="Normalized Dynamic Weights (95% CI)",
        xlabel="x", ylabel="Weight (normalized)",
        legend=:outerright
    )

    # Monte Carlo sampling from γ posteriors to compute normalized weight uncertainty
    n_samples = 500
    normalized_weights_samples = zeros(n_samples, n_forecasters, n_test)

    for s in 1:n_samples
        for j in 1:n_test
            # Sample γ for each forecaster from their posteriors
            γ_samples = [rand(γ_dynamic_posteriors[i, j]) for i in 1:n_forecasters]
            normalized_weights_samples[s, :, j] = γ_samples ./ sum(γ_samples)
        end
    end

    # Compute median and 95% CI of normalized weights
    for (i, (name, _)) in enumerate(forecasters)
        weight_median = [quantile(normalized_weights_samples[:, i, j], 0.5) for j in 1:n_test]
        weight_lower = [quantile(normalized_weights_samples[:, i, j], 0.025) for j in 1:n_test]
        weight_upper = [quantile(normalized_weights_samples[:, i, j], 0.975) for j in 1:n_test]

        plot!(p6, x_test, weight_median,
            ribbon=(weight_median .- weight_lower, weight_upper .- weight_median),
            label=name, lw=2, color=colors[i], fillalpha=0.2
        )
    end
    vline!(p6, [2π], color=:gray, ls=:dash, label="", lw=2)

    # Combine plots (3x2 layout)
    plt = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 1200))

    savefig(plt, "dynamic_ensemble_demo.png")
    @info "Saved visualization" file="dynamic_ensemble_demo.png"

    # ==========================================================================
    # Plot 7: Variance Decomposition (Aleatoric vs Epistemic) - Separate figure
    # ==========================================================================
    # Law of total variance: Var[y|x] = E[Var[y|γ]] + Var[E[y|γ]]
    # Aleatoric: E[Var[y|γ]] = E[1/Σγ] - expected noise variance given γ
    # Epistemic: Var[E[y|γ]] = Var[μ(x,γ)] - how prediction shifts with γ uncertainty
    # where μ(x,γ) = Σ(γᵢfᵢ(x)) / Σγᵢ (precision-weighted mean)

    # Dynamic ensemble decomposition
    dynamic_aleatoric = zeros(n_test)
    dynamic_epistemic = zeros(n_test)
    dynamic_total_var = zeros(n_test)

    for j in 1:n_test
        # Sample predictions and variances
        mu_samples = zeros(n_mc_samples)
        var_samples = zeros(n_mc_samples)

        for s in 1:n_mc_samples
            γ_samples = [rand(γ_dynamic_posteriors[i, j]) for i in 1:n_forecasters]
            total_precision = sum(γ_samples)

            # Predictive mean: precision-weighted average of forecaster predictions
            mu_samples[s] = sum(γ_samples[i] * predictions_test[i, j] for i in 1:n_forecasters) / total_precision

            # Predictive variance given this γ
            var_samples[s] = 1.0 / total_precision
        end

        # Aleatoric: E[Var[y|γ]] - expected variance given γ
        dynamic_aleatoric[j] = mean(var_samples)

        # Epistemic: Var[E[y|γ]] - variance of the mean prediction
        dynamic_epistemic[j] = var(mu_samples)

        # Total: Aleatoric + Epistemic (law of total variance)
        dynamic_total_var[j] = dynamic_aleatoric[j] + dynamic_epistemic[j]
    end

    # Static ensemble decomposition
    static_aleatoric = zeros(n_test)
    static_epistemic = zeros(n_test)
    static_total_var = zeros(n_test)

    for j in 1:n_test
        mu_samples = zeros(n_mc_samples)
        var_samples = zeros(n_mc_samples)

        for s in 1:n_mc_samples
            # Static: γ is constant across x, sample once per forecaster
            γ_samples = [rand(γ_static_posteriors[i]) for i in 1:n_forecasters]
            total_precision = sum(γ_samples)

            # Predictive mean: precision-weighted average
            mu_samples[s] = sum(γ_samples[i] * predictions_test[i, j] for i in 1:n_forecasters) / total_precision

            # Predictive variance given this γ
            var_samples[s] = 1.0 / total_precision
        end

        # Aleatoric: E[Var[y|γ]]
        static_aleatoric[j] = mean(var_samples)

        # Epistemic: Var[E[y|γ]]
        static_epistemic[j] = var(mu_samples)

        # Total
        static_total_var[j] = static_aleatoric[j] + static_epistemic[j]
    end

    # Create variance decomposition plot
    p7 = plot(layout=(1, 2), size=(1200, 400),
        title=["Dynamic Ensemble" "Static Ensemble"])

    # Dynamic subplot - stacked area
    plot!(p7[1], x_test, dynamic_aleatoric,
        fillrange=0, fillalpha=0.6, color=:blue,
        label="Aleatoric", lw=0)
    plot!(p7[1], x_test, dynamic_aleatoric .+ dynamic_epistemic,
        fillrange=dynamic_aleatoric, fillalpha=0.4, color=:orange,
        label="Epistemic", lw=0)
    plot!(p7[1], x_test, dynamic_total_var,
        label="Total Var", lw=2, color=:black, ls=:dash)
    vline!(p7[1], [2π], color=:gray, ls=:dash, label="Train boundary", lw=2)
    xlabel!(p7[1], "x")
    ylabel!(p7[1], "Variance")

    # Static subplot - stacked area
    plot!(p7[2], x_test, static_aleatoric,
        fillrange=0, fillalpha=0.6, color=:blue,
        label="Aleatoric", lw=0)
    plot!(p7[2], x_test, static_aleatoric .+ static_epistemic,
        fillrange=static_aleatoric, fillalpha=0.4, color=:orange,
        label="Epistemic", lw=0)
    plot!(p7[2], x_test, static_total_var,
        label="Total Var", lw=2, color=:black, ls=:dash)
    vline!(p7[2], [2π], color=:gray, ls=:dash, label="Train boundary", lw=2)
    xlabel!(p7[2], "x")
    ylabel!(p7[2], "Variance")

    savefig(p7, "variance_decomposition.png")
    @info "Saved variance decomposition" file="variance_decomposition.png"

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
    return weight_sin * sin(x) + weight_cos * cos(x) + 0.1 * x^2
end

function main()
    demo_dynamic_ensemble(true_function)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
