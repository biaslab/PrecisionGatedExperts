#!/usr/bin/env julia

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Statistics
using CairoMakie

# Fixed weights for the XOR demo (not learned).
const V_SPLIT_VIZ = [[14.0, 0.0, -7.0], [-14.0, 0.0, 7.0]]
const W_LEFT_VIZ  = [0.0, 10.0, 0.0]
const W_RIGHT_VIZ = [0.0, -10.0, 10.0]

# --- Model with tau as a parameter ---
@model function deep_model_xor_tau(n_obs, n_forecasters, tau_softdot, features, y, predictors)
    local h, right_switch, left_switch, z, kappa, γ
    for i in 1:n_forecasters
        for j = 1:n_obs
            h[j, i] ~ softdot(features[j], V_SPLIT_VIZ[i], tau_softdot)
            right_switch[j, i] ~ softdot(h[j, i], 1.0, tau_softdot)
            left_switch[j, i]  ~ softdot(h[j, i], -1.0, tau_softdot)

            kappa[j, 1, i] ~ GammaShapeRate(1.0, 1.0)
            kappa[j, 2, i] ~ GammaShapeRate(1.0, 1.0)
            right_switch[j, i] ~ Log(kappa[j, 1, i])
            left_switch[j, i]  ~ Log(kappa[j, 2, i])

            z[j, 1, i] ~ softdot(features[j], W_LEFT_VIZ, tau_softdot)
            z[j, 2, i] ~ softdot(features[j], W_RIGHT_VIZ, tau_softdot)

            m[j, i] ~ NormalMeanPrecision(z[j, 1, i], kappa[j, 1, i])
            m[j, i] ~ NormalMeanPrecision(z[j, 2, i], kappa[j, 2, i])
            γ[j, i] ~ GammaShapeRate(1.0, 1.0)
            m[j, i] ~ Log(γ[j, i])
            y[j] ~ NormalMeanPrecision(predictors[i, j], γ[j, i])
        end
    end
end

@constraints function viz_constraints()
    q(h, right_switch, left_switch, z, kappa, m, γ) = q(h)q(right_switch)q(left_switch)q(z)q(kappa)q(m, γ)
    q(h)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(right_switch)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(left_switch)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(z)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(kappa)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(m)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
end

@initialization function viz_init()
    q(h) = NormalMeanVariance(0.0, 1.0)
    q(right_switch) = NormalMeanVariance(0.0, 1.0)
    q(left_switch) = NormalMeanVariance(0.0, 1.0)
    q(z) = NormalMeanVariance(0.5, 1.0)
    q(kappa) = GammaShapeScale(2.0, 1.0)
    q(m) = NormalMeanVariance(0.5, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
end

# --- Grid setup ---
const GRID_RES = 25
const N_ITERS  = 15
const TAUS     = [5000.0, 2000.0, 500.0, 100.0, 10.0]

# XOR ground truth points
const XOR_X1     = [0.0, 0.0, 1.0, 1.0]
const XOR_X2     = [0.0, 1.0, 0.0, 1.0]
const XOR_TARGET = [0, 1, 1, 0]

function run_grid_inference(tau::Float64)
    xs = range(-0.05, 1.05, length = GRID_RES)
    features = [Float64[x1, x2, 1.0] for x1 in xs for x2 in xs]
    n_obs = length(features)
    predictors = vcat(zeros(1, n_obs), ones(1, n_obs))

    println("Running inference for τ = $tau  ($(n_obs) grid points, $(N_ITERS) iterations)…")
    result = infer(
        model          = deep_model_xor_tau(n_obs = n_obs, n_forecasters = 2, tau_softdot = tau),
        data           = (features = features, y = fill(missing, n_obs), predictors = predictors),
        constraints    = viz_constraints(),
        initialization = viz_init(),
        iterations     = N_ITERS,
        free_energy    = false,
        showprogress   = true,
    )
    return result, xs
end

function extract_iteration_data(result, grid_res)
    # result.predictions[:y] is a vector of length N_ITERS,
    # each element is a vector of distributions over the grid points.
    n_iters = length(result.predictions[:y])
    means_per_iter = Matrix{Float64}[]
    vars_per_iter  = Matrix{Float64}[]

    # γ posteriors: result.posteriors[:γ][iter] is an array indexed [j, i]
    gamma_mean_per_iter = Matrix{Float64}[]
    gamma_var_per_iter  = Matrix{Float64}[]

    for it in 1:n_iters
        y_dists = result.predictions[:y][it]
        y_m = mean.(y_dists)
        y_v = var.(y_dists)
        push!(means_per_iter, reshape(y_m, grid_res, grid_res))
        push!(vars_per_iter,  reshape(y_v, grid_res, grid_res))

        # Sum γ mean across forecasters as total precision → certainty
        γ_dists = result.posteriors[:γ][it]  # [j, i]
        γ_m = map(mean, γ_dists)  # [j, i]
        γ_v = map(var, γ_dists)
        # Sum across forecasters (axis 2) for aggregate precision
        push!(gamma_mean_per_iter, reshape(sum(γ_m, dims = 2)[:, 1], grid_res, grid_res))
        push!(gamma_var_per_iter,  reshape(sum(γ_v, dims = 2)[:, 1], grid_res, grid_res))
    end
    return means_per_iter, vars_per_iter, gamma_mean_per_iter, gamma_var_per_iter
end

function make_gif()
    # Run inference for all taus
    all_results = Dict{Float64, Any}()
    all_xs = nothing
    for tau in TAUS
        result, xs = run_grid_inference(tau)
        all_results[tau] = extract_iteration_data(result, GRID_RES)
        all_xs = xs
    end
    xs = all_xs

    n_tau = length(TAUS)

    # Create figure: 2 rows × n_tau columns
    # Row 1: mean prediction (classification region)
    # Row 2: precision certainty (γ variance → low = certain, high = uncertain)
    fig = Figure(size = (380 * n_tau, 750), fontsize = 14)

    # Pre-compute global ranges for consistent color scales
    all_pred_min = 0.0
    all_pred_max = 1.0

    # Find global gamma variance range for consistent scale
    all_gamma_var_max = 0.0
    for tau in TAUS
        _, _, _, gv = all_results[tau]
        for mat in gv
            m = maximum(mat)
            if m > all_gamma_var_max
                all_gamma_var_max = m
            end
        end
    end

    # Create axes and observables
    iter_obs = Observable(1)

    axes_pred = Axis[]
    axes_cert = Axis[]

    for (col, tau) in enumerate(TAUS)
        means, vars, gamma_means, gamma_vars = all_results[tau]

        # Row 1: classification prediction
        ax1 = Axis(fig[1, col],
            title = "τ = $(Int(tau))",
            xlabel = col == 3 ? "x₁" : "",
            ylabel = col == 1 ? "P(class = 1)" : "",
            aspect = 1,
        )
        pred_data = @lift(means[$iter_obs]')
        hm1 = heatmap!(ax1, collect(xs), collect(xs), pred_data,
            colormap = Reverse(:RdBu), colorrange = (0.0, 1.0))
        # XOR training points
        marker_colors = [t == 1 ? :lime : :black for t in XOR_TARGET]
        scatter!(ax1, XOR_X1, XOR_X2, color = marker_colors,
            markersize = 14, strokewidth = 2, strokecolor = :white)
        push!(axes_pred, ax1)

        # Row 2: precision uncertainty (variance of γ) — lower = more certain
        ax2 = Axis(fig[2, col],
            xlabel = "x₁",
            ylabel = col == 1 ? "Var(γ) — uncertainty" : "",
            aspect = 1,
        )
        cert_data = @lift(gamma_vars[$iter_obs]')
        heatmap!(ax2, collect(xs), collect(xs), cert_data,
            colormap = :inferno, colorrange = (0.0, max(all_gamma_var_max, 1e-3)))
        scatter!(ax2, XOR_X1, XOR_X2, color = marker_colors,
            markersize = 14, strokewidth = 2, strokecolor = :white)
        push!(axes_cert, ax2)
    end

    # Colorbars
    Colorbar(fig[1, n_tau + 1], colormap = Reverse(:RdBu), limits = (0.0, 1.0), label = "P(y)")
    Colorbar(fig[2, n_tau + 1], colormap = :inferno, limits = (0.0, max(all_gamma_var_max, 1e-3)), label = "Var(γ)")

    # Supertitle with iteration counter
    iter_label = @lift("XOR Classification Regions — Iteration $($iter_obs) / $(N_ITERS)")
    Label(fig[0, :], iter_label, fontsize = 18, font = :bold)

    outpath = joinpath(@__DIR__, "xor_classification_regions.gif")
    println("Recording GIF to $outpath …")
    record(fig, outpath, 1:N_ITERS; framerate = 2) do it
        iter_obs[] = it
    end
    println("Done! Saved to $outpath")
end

make_gif()
