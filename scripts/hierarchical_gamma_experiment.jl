#!/usr/bin/env julia

"""
Minimal experiment: Hierarchical Gamma model

Instead of γ = exp(z) directly (current model), we add a hierarchy:

    z[i,j] ~ softdot(features[j], w[i], τ[i])     # gating (Gaussian)
    β[i,j] ~ Log⁻¹(z[i,j])                         # β = exp(z) via Log node
    γ[i,j] ~ GammaShapeRate(α_fixed, β[i,j])        # actual precision
    y[j]   ~ NormalMeanPrecision(pred[i,j], γ[i,j])  # likelihood

Key insight:
  - Likelihood × GammaShapeRate is conjugate → q(γ) is exact Gamma, no projection needed
  - The only non-conjugate part is the Log node connecting z ↔ β (existing code)
  - The Gamma layer smooths out instabilities from exp(z)

Comparison:
  1. "direct"      — current model: γ = exp(z), projected q(γ)
  2. "hierarchical" — new model: β = exp(z), γ ~ Gamma(α, β), exact q(γ)
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
using ProbabilisticEnsembling

# =============================================================================
# Model 1: Current (direct) model — γ = exp(z)
# =============================================================================

@model function direct_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors, β_priors)
    local w, z, γ, τ, β

    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
        β[i] ~ β_priors[i]
    end

    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j])
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function direct_constraints()
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(γ) :: ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
end

@initialization function direct_init(n_features)
    q(w) = MvNormalMeanPrecision(zeros(n_features), diagm(ones(n_features)))
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
    q(β) = GammaShapeRate(1.0, 1.0)
end

# =============================================================================
# Model 2: Hierarchical model — β = exp(z), γ ~ Gamma(α, β)
# =============================================================================

@model function hierarchical_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors, ρ_priors, α_fixed)
    local w, z, β, γ, τ, ρ

    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
        ρ[i] ~ ρ_priors[i]           # per-expert rate hyperprior for β
    end

    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])

            # Hierarchy: exp(z) is the rate of a Gamma, not the precision itself
            β[i, j] ~ GammaShapeRate(1.0, ρ[i])      # learned rate shared across obs
            z[i, j] ~ Log(β[i, j])                     # z = log(β) ↔ β = exp(z)

            γ[i, j] ~ GammaShapeRate(α_fixed, β[i, j])  # actual precision
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function hierarchical_constraints()
    q(w, z, β, γ, τ, ρ) = q(w)q(z, β)q(γ)q(τ)q(ρ)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(β) :: ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
end

@initialization function hierarchical_init(n_features)
    q(w) = MvNormalMeanPrecision(zeros(n_features), diagm(ones(n_features)))
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(β) = GammaShapeScale(1.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
    q(ρ) = GammaShapeRate(1.0, 1.0)
end

# =============================================================================
# Helpers
# =============================================================================

function create_features(x)
    return [Float64[1.0, xi] for xi in x]
end

# =============================================================================
# Main experiment
# =============================================================================

function main()
    Random.seed!(7)

    n_train = 60
    n_iterations = 300
    n_forecasters = 2

    # Simple synthetic data
    x = collect(range(0, 2π, length = n_train))
    y_true = sin.(x)
    y = y_true .+ 0.05 .* randn(n_train)

    predictions = zeros(n_forecasters, n_train)
    predictions[1, :] = y          # cheater: perfect predictions
    predictions[2, :] .= 0.0      # baseline: all zeros

    features = create_features(x)
    n_features = length(features[1])

    w_priors = [MvNormalMeanPrecision(zeros(n_features), 0.1 * diagm(ones(n_features))) for _ in 1:n_forecasters]

    # -----------------------------------------------------------------
    # Run direct (current) model
    # -----------------------------------------------------------------
    println("=" ^ 60)
    println("Running DIRECT model (current: γ = exp(z))")
    println("=" ^ 60)

    τ_priors_direct = [GammaShapeScale(2.0, 0.2) for _ in 1:n_forecasters]
    β_priors_direct = [GammaShapeRate(20.0, 1.0) for _ in 1:n_forecasters]

    direct_result = infer(
        model = direct_model(
            n_forecasters = n_forecasters,
            n_obs = n_train,
            w_priors = w_priors,
            τ_priors = τ_priors_direct,
            β_priors = β_priors_direct
        ),
        data = (y = y, features = features, predictions = predictions),
        constraints = direct_constraints(),
        initialization = direct_init(n_features),
        iterations = n_iterations,
        free_energy = true,
        showprogress = true
    )

    # -----------------------------------------------------------------
    # Run hierarchical model with different α_fixed values
    # -----------------------------------------------------------------
    α_values = [1.0, 2.0, 5.0]
    hierarchical_results = Dict{Float64, Any}()

    for α in α_values
        println("\n" * "=" ^ 60)
        println("Running HIERARCHICAL model (α_fixed = $α)")
        println("=" ^ 60)

        τ_priors_hier = [GammaShapeScale(2.0, 0.2) for _ in 1:n_forecasters]
        ρ_priors_hier = [GammaShapeRate(1.0, 1.0) for _ in 1:n_forecasters]

        result = infer(
            model = hierarchical_model(
                n_forecasters = n_forecasters,
                n_obs = n_train,
                w_priors = w_priors,
                τ_priors = τ_priors_hier,
                ρ_priors = ρ_priors_hier,
                α_fixed = α
            ),
            data = (y = y, features = features, predictions = predictions),
            constraints = hierarchical_constraints(),
            initialization = hierarchical_init(n_features),
            iterations = n_iterations,
            free_energy = true,
            showprogress = true
        )

        hierarchical_results[α] = result
    end

    # -----------------------------------------------------------------
    # Compare free energy convergence
    # -----------------------------------------------------------------
    println("\n" * "=" ^ 60)
    println("FREE ENERGY COMPARISON")
    println("=" ^ 60)

    direct_fe = direct_result.free_energy
    println("\nDirect model:")
    println("  FE first 5: ", round.(direct_fe[1:min(5, end)], digits = 3))
    println("  FE last 5:  ", round.(direct_fe[max(1, end - 4):end], digits = 3))
    println("  FE monotonic decreasing: ", all(diff(direct_fe) .<= 0))
    println("  FE final change: ", round(direct_fe[end] - direct_fe[end-1], digits = 6))

    for α in α_values
        fe = hierarchical_results[α].free_energy
        println("\nHierarchical (α=$α):")
        println("  FE first 5: ", round.(fe[1:min(5, end)], digits = 3))
        println("  FE last 5:  ", round.(fe[max(1, end - 4):end], digits = 3))
        println("  FE monotonic decreasing: ", all(diff(fe) .<= 0))
        println("  FE final change: ", round(fe[end] - fe[end-1], digits = 6))
    end

    # -----------------------------------------------------------------
    # Compare learned posteriors
    # -----------------------------------------------------------------
    println("\n" * "=" ^ 60)
    println("POSTERIOR COMPARISON")
    println("=" ^ 60)

    println("\nDirect model — w posteriors:")
    w_post_direct = direct_result.posteriors[:w][end]
    for i in 1:n_forecasters
        println("  Forecaster $i: w = ", round.(mean(w_post_direct[i]), digits = 4))
    end

    for α in α_values
        println("\nHierarchical (α=$α) — w posteriors:")
        w_post = hierarchical_results[α].posteriors[:w][end]
        for i in 1:n_forecasters
            println("  Forecaster $i: w = ", round.(mean(w_post[i]), digits = 4))
        end

        println("Hierarchical (α=$α) — ρ posteriors (learned rate hyperprior for β):")
        ρ_post = hierarchical_results[α].posteriors[:ρ][end]
        for i in 1:n_forecasters
            println("  Forecaster $i: ρ mean=$(round(mean(ρ_post[i]), digits=4)), var=$(round(var(ρ_post[i]), digits=6))")
        end

        println("Hierarchical (α=$α) — γ posteriors (first 5 obs, forecaster 1):")
        γ_post = hierarchical_results[α].posteriors[:γ][end]
        for j in 1:min(5, n_train)
            g = γ_post[1, j]
            println("  γ[1,$j]: mean=$(round(mean(g), digits=4)), var=$(round(var(g), digits=6))")
        end
    end

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------
    colors = [:blue, :green, :orange, :purple]

    p = plot(1:length(direct_fe), direct_fe,
        xlabel = "Iteration", ylabel = "Free Energy",
        title = "Free Energy: Direct vs Hierarchical",
        label = "Direct (current)", lw = 2, color = :red,
        marker = :circle, markersize = 2, legend = :topright
    )

    for (idx, α) in enumerate(α_values)
        fe = hierarchical_results[α].free_energy
        plot!(p, 1:length(fe), fe,
            label = "Hierarchical (α=$α)", lw = 2, color = colors[idx],
            marker = :diamond, markersize = 2
        )
    end

    savefig(p, "free_energy_hierarchical_comparison.png")
    println("\nSaved: free_energy_hierarchical_comparison.png")

    # Also plot the last 20 iterations zoomed in
    p2 = plot(
        xlabel = "Iteration", ylabel = "Free Energy",
        title = "Free Energy (last 20 iterations, zoomed)",
        legend = :topright
    )

    start_iter = max(1, n_iterations - 19)
    plot!(p2, start_iter:length(direct_fe), direct_fe[start_iter:end],
        label = "Direct (current)", lw = 2, color = :red, marker = :circle, markersize = 3
    )

    for (idx, α) in enumerate(α_values)
        fe = hierarchical_results[α].free_energy
        plot!(p2, start_iter:length(fe), fe[start_iter:end],
            label = "Hierarchical (α=$α)", lw = 2, color = colors[idx],
            marker = :diamond, markersize = 3
        )
    end

    savefig(p2, "free_energy_hierarchical_zoomed.png")
    println("Saved: free_energy_hierarchical_zoomed.png")

    # -----------------------------------------------------------------
    # Prediction quality: use trained posteriors on test data
    # -----------------------------------------------------------------
    println("\n" * "=" ^ 60)
    println("PREDICTION QUALITY")
    println("=" ^ 60)

    # Generate test data (different x range for extrapolation + interpolation)
    n_test = 100
    x_test = collect(range(0, 2π, length = n_test))
    y_test = sin.(x_test) .+ 0.05 .* randn(n_test)

    predictions_test = zeros(n_forecasters, n_test)
    predictions_test[1, :] = sin.(x_test)   # no longer cheating — uses true function, not y
    predictions_test[2, :] .= 0.0

    features_test = create_features(x_test)
    y_missing = [missing for _ in 1:n_test]

    # --- Direct model predictions ---
    println("\nDirect model predictions:")
    w_post_d = direct_result.posteriors[:w][end]
    τ_post_d = direct_result.posteriors[:τ][end]
    β_post_d = direct_result.posteriors[:β][end]

    direct_pred = infer(
        model = direct_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            w_priors = w_post_d,
            τ_priors = τ_post_d,
            β_priors = β_post_d
        ),
        data = (y = y_missing, features = features_test, predictions = predictions_test),
        constraints = direct_constraints(),
        initialization = direct_init(n_features),
        iterations = 10,
        free_energy = false,
        showprogress = false
    )

    # Compute ensemble prediction for direct model
    γ_direct = direct_pred.posteriors[:γ][end]
    y_pred_direct = zeros(n_test)
    y_std_direct = zeros(n_test)
    for j in 1:n_test
        total_prec = 0.0
        weighted_mean = 0.0
        for i in 1:n_forecasters
            g = mean(γ_direct[i, j])
            total_prec += g
            weighted_mean += g * predictions_test[i, j]
        end
        y_pred_direct[j] = weighted_mean / total_prec
        y_std_direct[j] = 1.0 / sqrt(total_prec)
    end

    rmse_direct = sqrt(mean((y_pred_direct .- y_test) .^ 2))
    mae_direct = mean(abs.(y_pred_direct .- y_test))
    println("  RMSE: ", round(rmse_direct, digits = 6))
    println("  MAE:  ", round(mae_direct, digits = 6))

    # --- Hierarchical model predictions ---
    hierarchical_preds = Dict{Float64, Any}()
    hierarchical_y_pred = Dict{Float64, Vector{Float64}}()
    hierarchical_y_std = Dict{Float64, Vector{Float64}}()

    for α in α_values
        println("\nHierarchical (α=$α) predictions:")
        w_post_h = hierarchical_results[α].posteriors[:w][end]
        τ_post_h = hierarchical_results[α].posteriors[:τ][end]
        ρ_post_h = hierarchical_results[α].posteriors[:ρ][end]

        hier_pred = infer(
            model = hierarchical_model(
                n_forecasters = n_forecasters,
                n_obs = n_test,
                w_priors = w_post_h,
                τ_priors = τ_post_h,
                ρ_priors = ρ_post_h,
                α_fixed = α
            ),
            data = (y = y_missing, features = features_test, predictions = predictions_test),
            constraints = hierarchical_constraints(),
            initialization = hierarchical_init(n_features),
            iterations = 10,
            free_energy = false,
            showprogress = false
        )

        hierarchical_preds[α] = hier_pred

        γ_hier = hier_pred.posteriors[:γ][end]
        y_pred_h = zeros(n_test)
        y_std_h = zeros(n_test)
        for j in 1:n_test
            total_prec = 0.0
            weighted_mean = 0.0
            for i in 1:n_forecasters
                g = mean(γ_hier[i, j])
                total_prec += g
                weighted_mean += g * predictions_test[i, j]
            end
            y_pred_h[j] = weighted_mean / total_prec
            y_std_h[j] = 1.0 / sqrt(total_prec)
        end

        hierarchical_y_pred[α] = y_pred_h
        hierarchical_y_std[α] = y_std_h

        rmse_h = sqrt(mean((y_pred_h .- y_test) .^ 2))
        mae_h = mean(abs.(y_pred_h .- y_test))
        println("  RMSE: ", round(rmse_h, digits = 6))
        println("  MAE:  ", round(mae_h, digits = 6))
    end

    # --- Individual forecaster baselines ---
    println("\nBaseline (individual forecasters):")
    for i in 1:n_forecasters
        rmse_i = sqrt(mean((predictions_test[i, :] .- y_test) .^ 2))
        mae_i = mean(abs.(predictions_test[i, :] .- y_test))
        println("  Forecaster $i — RMSE: ", round(rmse_i, digits = 6), ", MAE: ", round(mae_i, digits = 6))
    end

    # -----------------------------------------------------------------
    # Prediction plot
    # -----------------------------------------------------------------
    p3 = plot(x_test, y_test,
        seriestype = :scatter, markersize = 2, alpha = 0.4, color = :gray,
        label = "Test data", xlabel = "x", ylabel = "y",
        title = "Ensemble Predictions", legend = :bottomleft
    )

    # Individual forecasters
    for i in 1:n_forecasters
        plot!(p3, x_test, predictions_test[i, :],
            label = "Forecaster $i", lw = 1, ls = :dash, alpha = 0.5
        )
    end

    # Direct model
    plot!(p3, x_test, y_pred_direct,
        label = "Direct", lw = 2, color = :red
    )
    plot!(p3, x_test, y_pred_direct .+ 2 .* y_std_direct,
        fillrange = y_pred_direct .- 2 .* y_std_direct,
        alpha = 0.15, color = :red, label = nothing
    )

    # Hierarchical models
    for (idx, α) in enumerate(α_values)
        plot!(p3, x_test, hierarchical_y_pred[α],
            label = "Hier (α=$α)", lw = 2, color = colors[idx]
        )
        plot!(p3, x_test, hierarchical_y_pred[α] .+ 2 .* hierarchical_y_std[α],
            fillrange = hierarchical_y_pred[α] .- 2 .* hierarchical_y_std[α],
            alpha = 0.1, color = colors[idx], label = nothing
        )
    end

    savefig(p3, "predictions_hierarchical_comparison.png")
    println("\nSaved: predictions_hierarchical_comparison.png")

    # -----------------------------------------------------------------
    # Precision weights plot: how much weight each forecaster gets
    # -----------------------------------------------------------------
    p4 = plot(layout = (length(α_values) + 1, 1), size = (800, 300 * (length(α_values) + 1)),
        xlabel = "x", ylabel = "Precision weight"
    )

    # Direct model weights
    γ_direct_train = direct_result.posteriors[:γ][end]
    for i in 1:n_forecasters
        γ_means = [mean(γ_direct_train[i, j]) for j in 1:n_train]
        total = sum([mean(γ_direct_train[k, j]) for k in 1:n_forecasters] for j in 1:n_train)
        weights = [mean(γ_direct_train[i, j]) / sum(mean(γ_direct_train[k, j]) for k in 1:n_forecasters) for j in 1:n_train]
        plot!(p4[1], x, weights, label = "F$i", lw = 2, title = "Direct model weights")
    end

    # Hierarchical model weights
    for (idx, α) in enumerate(α_values)
        γ_hier_train = hierarchical_results[α].posteriors[:γ][end]
        for i in 1:n_forecasters
            weights = [mean(γ_hier_train[i, j]) / sum(mean(γ_hier_train[k, j]) for k in 1:n_forecasters) for j in 1:n_train]
            plot!(p4[idx + 1], x, weights, label = "F$i", lw = 2, title = "Hierarchical (α=$α) weights")
        end
    end

    savefig(p4, "weights_hierarchical_comparison.png")
    println("Saved: weights_hierarchical_comparison.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
