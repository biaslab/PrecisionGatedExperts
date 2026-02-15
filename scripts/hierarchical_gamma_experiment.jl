#!/usr/bin/env julia

"""
Hierarchical Gamma experiment — stress-testing convergence and prediction quality.

Three scenarios of increasing difficulty:
  A) "hard_switch"   — y = sin(x) for x<π, y = cos(x) for x≥π
                        F1 = sin(x), F2 = cos(x) — model must learn a sharp switch
  B) "both_useful"   — y = 0.5·sin(x) + 0.5·cos(x) everywhere
                        F1 = sin(x), F2 = cos(x) — both always useful, roughly equal
  C) "many_experts"  — y = sin(x) for x<π, y = cos(x) for x≥π (same as A)
                        but 6 forecasters, only 2 useful, rest are noise/constants

Each scenario runs:
  1. Direct model (current: γ = exp(z))
  2. Hierarchical model with α ∈ {1.0, 2.0, 5.0}

Convergence criterion: |ΔFE/FE| < rtol for `patience` consecutive iterations.
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
# Models
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

@model function hierarchical_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors, ρ_priors, α_fixed)
    local w, z, β, γ, τ, ρ
    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
        ρ[i] ~ ρ_priors[i]
    end
    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])
            β[i, j] ~ GammaShapeRate(1.0, ρ[i])
            z[i, j] ~ Log(β[i, j])
            γ[i, j] ~ GammaShapeRate(α_fixed, β[i, j])
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

"""
Check convergence: |ΔFE / |FE|| < rtol for `patience` consecutive iterations.
Returns (converged::Bool, iteration_converged::Int or -1, relative_changes::Vector)
"""
function check_convergence(fe::Vector{<:Real}; rtol = 1e-4, patience = 5)
    rel_changes = Float64[]
    for i in 2:length(fe)
        denom = max(abs(fe[i]), 1.0)   # avoid division by tiny FE
        push!(rel_changes, abs(fe[i] - fe[i-1]) / denom)
    end

    streak = 0
    for i in eachindex(rel_changes)
        if rel_changes[i] < rtol
            streak += 1
            if streak >= patience
                return (true, i - patience + 2, rel_changes)   # +2: offset from diff indexing
            end
        else
            streak = 0
        end
    end
    return (false, -1, rel_changes)
end

"""
Compute ensemble predictions from γ posteriors.
Returns (y_pred, y_std).
"""
function ensemble_predict(γ_post, predictions, n_forecasters, n_obs)
    y_pred = zeros(n_obs)
    y_std = zeros(n_obs)
    for j in 1:n_obs
        total_prec = 0.0
        weighted_mean = 0.0
        for i in 1:n_forecasters
            g = mean(γ_post[i, j])
            total_prec += g
            weighted_mean += g * predictions[i, j]
        end
        y_pred[j] = weighted_mean / max(total_prec, 1e-12)
        y_std[j] = 1.0 / sqrt(max(total_prec, 1e-12))
    end
    return y_pred, y_std
end

# =============================================================================
# Scenario definitions
# =============================================================================

function make_scenario(name::Symbol, n_train, noise_level, rng)
    x = collect(range(0, 2π, length = n_train))

    if name == :hard_switch
        # True function: sin for x<π, cos for x≥π — sharp transition
        y_true = [xi < π ? sin(xi) : cos(xi) for xi in x]
        y = y_true .+ noise_level .* randn(rng, n_train)

        forecaster_names = ["sin(x)", "cos(x)"]
        predictions = zeros(2, n_train)
        predictions[1, :] = sin.(x)
        predictions[2, :] = cos.(x)

        # Test: same structure, fresh noise
        x_test = collect(range(0, 2π, length = 120))
        y_test = [xi < π ? sin(xi) : cos(xi) for xi in x_test] .+ noise_level .* randn(rng, 120)
        pred_test = zeros(2, 120)
        pred_test[1, :] = sin.(x_test)
        pred_test[2, :] = cos.(x_test)

        return (; x, y, predictions, forecaster_names, x_test, y_test, pred_test,
                  description = "Hard switch: sin(x) for x<π, cos(x) for x≥π")

    elseif name == :both_useful
        # True function: equal mix — both forecasters always useful
        y_true = 0.5 .* sin.(x) .+ 0.5 .* cos.(x)
        y = y_true .+ noise_level .* randn(rng, n_train)

        forecaster_names = ["sin(x)", "cos(x)"]
        predictions = zeros(2, n_train)
        predictions[1, :] = sin.(x)
        predictions[2, :] = cos.(x)

        x_test = collect(range(0, 2π, length = 120))
        y_test = 0.5 .* sin.(x_test) .+ 0.5 .* cos.(x_test) .+ noise_level .* randn(rng, 120)
        pred_test = zeros(2, 120)
        pred_test[1, :] = sin.(x_test)
        pred_test[2, :] = cos.(x_test)

        return (; x, y, predictions, forecaster_names, x_test, y_test, pred_test,
                  description = "Both useful: y = 0.5·sin + 0.5·cos")

    elseif name == :many_experts
        # Same hard switch target, but 6 forecasters: sin, cos + 4 distractors
        y_true = [xi < π ? sin(xi) : cos(xi) for xi in x]
        y = y_true .+ noise_level .* randn(rng, n_train)

        forecaster_names = ["sin(x)", "cos(x)", "0.5x", "constant_1", "noise", "-sin(x)"]
        predictions = zeros(6, n_train)
        predictions[1, :] = sin.(x)
        predictions[2, :] = cos.(x)
        predictions[3, :] = 0.5 .* x
        predictions[4, :] .= 1.0
        predictions[5, :] = 0.3 .* randn(rng, n_train)
        predictions[6, :] = -sin.(x)

        x_test = collect(range(0, 2π, length = 120))
        y_test = [xi < π ? sin(xi) : cos(xi) for xi in x_test] .+ noise_level .* randn(rng, 120)
        pred_test = zeros(6, 120)
        pred_test[1, :] = sin.(x_test)
        pred_test[2, :] = cos.(x_test)
        pred_test[3, :] = 0.5 .* x_test
        pred_test[4, :] .= 1.0
        pred_test[5, :] = 0.3 .* randn(rng, 120)
        pred_test[6, :] = -sin.(x_test)

        return (; x, y, predictions, forecaster_names, x_test, y_test, pred_test,
                  description = "Many experts: 6 forecasters, only sin+cos useful, hard switch")
    else
        error("Unknown scenario: $name")
    end
end

# =============================================================================
# Run one scenario
# =============================================================================

function run_scenario(scenario_name::Symbol; n_train = 100, noise_level = 0.1,
                      n_iterations = 500, α_values = [1.0, 2.0, 5.0],
                      conv_rtol = 1e-4, conv_patience = 10, rng = Random.default_rng())

    sc = make_scenario(scenario_name, n_train, noise_level, rng)
    n_forecasters = size(sc.predictions, 1)
    features = create_features(sc.x)
    n_features = length(features[1])

    println("\n" * "=" ^ 70)
    println("SCENARIO: $(scenario_name)  —  $(sc.description)")
    println("  n_train=$n_train, n_forecasters=$n_forecasters, noise=$noise_level, max_iter=$n_iterations")
    println("  convergence: |ΔFE/|FE|| < $conv_rtol for $conv_patience consecutive iters")
    println("=" ^ 70)

    w_priors = [MvNormalMeanPrecision(zeros(n_features), 0.1 * diagm(ones(n_features))) for _ in 1:n_forecasters]

    # --- Direct model ---
    println("\n  Running DIRECT model...")
    τ_d = [GammaShapeScale(2.0, 0.2) for _ in 1:n_forecasters]
    β_d = [GammaShapeRate(20.0, 1.0) for _ in 1:n_forecasters]

    direct_result = infer(
        model = direct_model(n_forecasters = n_forecasters, n_obs = n_train,
                             w_priors = w_priors, τ_priors = τ_d, β_priors = β_d),
        data = (y = sc.y, features = features, predictions = sc.predictions),
        constraints = direct_constraints(),
        initialization = direct_init(n_features),
        iterations = n_iterations, free_energy = true, showprogress = true
    )

    # --- Hierarchical models ---
    hier_results = Dict{Float64, Any}()
    for α in α_values
        println("\n  Running HIERARCHICAL model (α=$α)...")
        τ_h = [GammaShapeScale(2.0, 0.2) for _ in 1:n_forecasters]
        ρ_h = [GammaShapeRate(1.0, 1.0) for _ in 1:n_forecasters]

        result = infer(
            model = hierarchical_model(n_forecasters = n_forecasters, n_obs = n_train,
                                       w_priors = w_priors, τ_priors = τ_h,
                                       ρ_priors = ρ_h, α_fixed = α),
            data = (y = sc.y, features = features, predictions = sc.predictions),
            constraints = hierarchical_constraints(),
            initialization = hierarchical_init(n_features),
            iterations = n_iterations, free_energy = true, showprogress = true
        )
        hier_results[α] = result
    end

    # =====================================================================
    # Convergence analysis
    # =====================================================================
    println("\n  " * "-" ^ 60)
    println("  CONVERGENCE ANALYSIS")
    println("  " * "-" ^ 60)

    direct_fe = direct_result.free_energy
    conv_d, iter_d, rc_d = check_convergence(direct_fe; rtol = conv_rtol, patience = conv_patience)
    println("\n  Direct:")
    println("    monotonic: ", all(diff(direct_fe) .<= 0))
    println("    converged: $conv_d", conv_d ? " at iteration $iter_d" : " (NOT converged)")
    println("    final |ΔFE/|FE||: ", round(rc_d[end], sigdigits = 4))

    for α in α_values
        fe = hier_results[α].free_energy
        conv, iter, rc = check_convergence(fe; rtol = conv_rtol, patience = conv_patience)
        println("\n  Hierarchical (α=$α):")
        println("    monotonic: ", all(diff(fe) .<= 0))
        println("    converged: $conv", conv ? " at iteration $iter" : " (NOT converged)")
        println("    final |ΔFE/|FE||: ", round(rc[end], sigdigits = 4))
    end

    # =====================================================================
    # Predictions
    # =====================================================================
    println("\n  " * "-" ^ 60)
    println("  PREDICTION QUALITY")
    println("  " * "-" ^ 60)

    n_test = length(sc.x_test)
    features_test = create_features(sc.x_test)
    y_missing = [missing for _ in 1:n_test]

    # Direct predictions
    direct_pred = infer(
        model = direct_model(n_forecasters = n_forecasters, n_obs = n_test,
                             w_priors = direct_result.posteriors[:w][end],
                             τ_priors = direct_result.posteriors[:τ][end],
                             β_priors = direct_result.posteriors[:β][end]),
        data = (y = y_missing, features = features_test, predictions = sc.pred_test),
        constraints = direct_constraints(), initialization = direct_init(n_features),
        iterations = 10, free_energy = false, showprogress = false
    )
    y_pred_d, y_std_d = ensemble_predict(direct_pred.posteriors[:γ][end], sc.pred_test, n_forecasters, n_test)
    rmse_d = sqrt(mean((y_pred_d .- sc.y_test) .^ 2))
    mae_d = mean(abs.(y_pred_d .- sc.y_test))
    println("\n  Direct:  RMSE=$(round(rmse_d, digits=6)), MAE=$(round(mae_d, digits=6))")

    hier_y_preds = Dict{Float64, Vector{Float64}}()
    hier_y_stds = Dict{Float64, Vector{Float64}}()

    for α in α_values
        hier_pred = infer(
            model = hierarchical_model(n_forecasters = n_forecasters, n_obs = n_test,
                                       w_priors = hier_results[α].posteriors[:w][end],
                                       τ_priors = hier_results[α].posteriors[:τ][end],
                                       ρ_priors = hier_results[α].posteriors[:ρ][end],
                                       α_fixed = α),
            data = (y = y_missing, features = features_test, predictions = sc.pred_test),
            constraints = hierarchical_constraints(), initialization = hierarchical_init(n_features),
            iterations = 10, free_energy = false, showprogress = false
        )
        yp, ys = ensemble_predict(hier_pred.posteriors[:γ][end], sc.pred_test, n_forecasters, n_test)
        hier_y_preds[α] = yp
        hier_y_stds[α] = ys
        rmse_h = sqrt(mean((yp .- sc.y_test) .^ 2))
        mae_h = mean(abs.(yp .- sc.y_test))
        println("  Hier α=$α: RMSE=$(round(rmse_h, digits=6)), MAE=$(round(mae_h, digits=6))")
    end

    println("\n  Baselines (individual forecasters):")
    for i in 1:n_forecasters
        rmse_i = sqrt(mean((sc.pred_test[i, :] .- sc.y_test) .^ 2))
        mae_i = mean(abs.(sc.pred_test[i, :] .- sc.y_test))
        println("    $(sc.forecaster_names[i]): RMSE=$(round(rmse_i, digits=6)), MAE=$(round(mae_i, digits=6))")
    end

    # =====================================================================
    # Plots
    # =====================================================================
    colors = [:blue, :green, :orange, :purple]
    prefix = string(scenario_name)

    # --- Free energy convergence ---
    p1 = plot(direct_fe, label = "Direct", lw = 2, color = :red,
              xlabel = "Iteration", ylabel = "Free Energy",
              title = "$prefix — Free Energy", legend = :topright)
    for (idx, α) in enumerate(α_values)
        plot!(p1, hier_results[α].free_energy, label = "Hier α=$α", lw = 2, color = colors[idx])
    end
    savefig(p1, "$(prefix)_free_energy.png")

    # --- Relative convergence ---
    p1b = plot(rc_d, label = "Direct", lw = 2, color = :red, yscale = :log10,
               xlabel = "Iteration", ylabel = "|ΔFE/|FE||",
               title = "$prefix — Relative FE Change", legend = :topright)
    hline!(p1b, [conv_rtol], ls = :dash, color = :black, label = "rtol=$conv_rtol")
    for (idx, α) in enumerate(α_values)
        _, _, rc_h = check_convergence(hier_results[α].free_energy; rtol = conv_rtol, patience = conv_patience)
        plot!(p1b, rc_h, label = "Hier α=$α", lw = 2, color = colors[idx])
    end
    savefig(p1b, "$(prefix)_convergence.png")

    # --- Predictions ---
    p2 = plot(sc.x_test, sc.y_test, st = :scatter, ms = 2, alpha = 0.3, color = :gray,
              label = "Test data", xlabel = "x", ylabel = "y",
              title = "$prefix — Predictions", legend = :outertopright, size = (900, 400))
    for i in 1:n_forecasters
        plot!(p2, sc.x_test, sc.pred_test[i, :], ls = :dash, alpha = 0.4, label = sc.forecaster_names[i])
    end
    plot!(p2, sc.x_test, y_pred_d, lw = 2, color = :red, label = "Direct")
    plot!(p2, sc.x_test, y_pred_d .+ 2 .* y_std_d, fillrange = y_pred_d .- 2 .* y_std_d,
          alpha = 0.12, color = :red, label = nothing)
    for (idx, α) in enumerate(α_values)
        plot!(p2, sc.x_test, hier_y_preds[α], lw = 2, color = colors[idx], label = "Hier α=$α")
        plot!(p2, sc.x_test, hier_y_preds[α] .+ 2 .* hier_y_stds[α],
              fillrange = hier_y_preds[α] .- 2 .* hier_y_stds[α],
              alpha = 0.08, color = colors[idx], label = nothing)
    end
    savefig(p2, "$(prefix)_predictions.png")

    # --- Weights ---
    n_rows = 1 + length(α_values)
    p3 = plot(layout = (n_rows, 1), size = (800, 250 * n_rows))

    γ_d_train = direct_result.posteriors[:γ][end]
    for i in 1:n_forecasters
        wts = [mean(γ_d_train[i, j]) / sum(mean(γ_d_train[k, j]) for k in 1:n_forecasters) for j in 1:n_train]
        plot!(p3[1], sc.x, wts, label = sc.forecaster_names[i], lw = 2, title = "Direct weights", ylims = (0, 1))
    end
    for (idx, α) in enumerate(α_values)
        γ_h_train = hier_results[α].posteriors[:γ][end]
        for i in 1:n_forecasters
            wts = [mean(γ_h_train[i, j]) / sum(mean(γ_h_train[k, j]) for k in 1:n_forecasters) for j in 1:n_train]
            plot!(p3[idx + 1], sc.x, wts, label = sc.forecaster_names[i], lw = 2,
                  title = "Hier α=$α weights", ylims = (0, 1))
        end
    end
    savefig(p3, "$(prefix)_weights.png")

    println("\n  Saved: $(prefix)_free_energy.png, $(prefix)_convergence.png, $(prefix)_predictions.png, $(prefix)_weights.png")
end

# =============================================================================
# Main
# =============================================================================

function main()
    Random.seed!(42)

    scenarios = [:hard_switch, :both_useful, :many_experts]

    for sc in scenarios
        run_scenario(sc;
            n_train = 100,
            noise_level = 0.1,
            n_iterations = 500,
            α_values = [1.0, 2.0, 5.0],
            conv_rtol = 1e-4,
            conv_patience = 10
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
