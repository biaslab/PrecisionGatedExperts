#!/usr/bin/env julia

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Statistics
using CairoMakie

# Fixed weights for the XOR demo (not learned).
const V_SPLIT_FIG = [[14.0, 0.0, -7.0], [-14.0, 0.0, 7.0]]
const W_LEFT_FIG  = [0.0, 10.0, 0.0]
const W_RIGHT_FIG = [0.0, -10.0, 10.0]

# --- Model with tau as a parameter ---
@model function deep_model_xor_tau_fig(n_obs, n_forecasters, tau_softdot, features, y, predictors)
    local h, right_switch, left_switch, z, kappa, γ
    for i in 1:n_forecasters
        for j = 1:n_obs
            h[j, i] ~ softdot(features[j], V_SPLIT_FIG[i], tau_softdot)
            right_switch[j, i] ~ softdot(h[j, i], 1.0, tau_softdot)
            left_switch[j, i]  ~ softdot(h[j, i], -1.0, tau_softdot)

            kappa[j, 1, i] ~ GammaShapeRate(1.0, 1.0)
            kappa[j, 2, i] ~ GammaShapeRate(1.0, 1.0)
            right_switch[j, i] ~ Log(kappa[j, 1, i])
            left_switch[j, i]  ~ Log(kappa[j, 2, i])

            z[j, 1, i] ~ softdot(features[j], W_LEFT_FIG, tau_softdot)
            z[j, 2, i] ~ softdot(features[j], W_RIGHT_FIG, tau_softdot)

            m[j, i] ~ NormalMeanPrecision(z[j, 1, i], kappa[j, 1, i])
            m[j, i] ~ NormalMeanPrecision(z[j, 2, i], kappa[j, 2, i])
            γ[j, i] ~ GammaShapeRate(1.0, 1.0)
            m[j, i] ~ Log(γ[j, i])
            y[j] ~ NormalMeanPrecision(predictors[i, j], γ[j, i])
        end
    end
end

@constraints function fig_constraints()
    q(h, right_switch, left_switch, z, kappa, m, γ) = q(h)q(right_switch)q(left_switch)q(z)q(kappa)q(m, γ)
    q(h)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(right_switch)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(left_switch)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(z)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(kappa)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(m)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
end

@initialization function fig_init()
    q(h) = NormalMeanVariance(0.0, 1.0)
    q(right_switch) = NormalMeanVariance(0.0, 1.0)
    q(left_switch) = NormalMeanVariance(0.0, 1.0)
    q(z) = NormalMeanVariance(0.5, 1.0)
    q(kappa) = GammaShapeScale(2.0, 1.0)
    q(m) = NormalMeanVariance(0.5, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
end

# --- Grid and inference ---
const GRID_RES = 50
const N_ITERS  = 15
const TAUS     = [10.0, 500.0]

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
        model          = deep_model_xor_tau_fig(n_obs = n_obs, n_forecasters = 2, tau_softdot = tau),
        data           = (features = features, y = fill(missing, n_obs), predictors = predictors),
        constraints    = fig_constraints(),
        initialization = fig_init(),
        iterations     = N_ITERS,
        free_energy    = false,
        showprogress   = true,
    )
    return result, xs
end

function extract_final(result, grid_res)
    y_dists = result.predictions[:y][end]
    y_m = reshape(mean.(y_dists), grid_res, grid_res)
    y_s = reshape(sqrt.(var.(y_dists)), grid_res, grid_res)
    return y_m, y_s
end

function make_figure()
    # Run inference for both tau values
    results = Dict{Float64, Tuple{Matrix{Float64}, Matrix{Float64}}}()
    xs = nothing
    for tau in TAUS
        result, xs_local = run_grid_inference(tau)
        results[tau] = extract_final(result, GRID_RES)
        xs = xs_local
    end

    # Find global std range for consistent colorbar
    std_max = maximum(maximum(results[tau][2]) for tau in TAUS)

    # Create figure: 2 rows × 2 columns, large enough for readable labels
    fig = Figure(size = (600, 550), fontsize = 14)

    marker_shapes = [t == 1 ? :circle : :diamond for t in XOR_TARGET]
    marker_colors = [t == 1 ? :lime : :black for t in XOR_TARGET]

    tau_labels = [L"\tau = 10", L"\tau = 500"]

    for (col, tau) in enumerate(TAUS)
        y_mean, y_std = results[tau]

        # Row 1: posterior mean
        ax1 = Axis(fig[1, col],
            title = tau_labels[col],
            titlesize = 16,
            ylabel = col == 1 ? L"\mathbb{E}_q[y](\mathbf{x})" : "",
            aspect = 1,
            xticklabelsvisible = false,
            ylabelsize = 14,
        )
        heatmap!(ax1, collect(xs), collect(xs), y_mean',
            colormap = Reverse(:RdBu), colorrange = (0.0, 1.0))
        scatter!(ax1, XOR_X1, XOR_X2, color = marker_colors, marker = marker_shapes,
            markersize = 12, strokewidth = 2, strokecolor = :white)

        # Row 2: posterior std
        ax2 = Axis(fig[2, col],
            xlabel = L"x_1",
            ylabel = col == 1 ? L"\mathrm{Std}_q[y](\mathbf{x})" : "",
            aspect = 1,
            xlabelsize = 14,
            ylabelsize = 14,
        )
        heatmap!(ax2, collect(xs), collect(xs), y_std',
            colormap = :inferno, colorrange = (0.0, std_max))
        scatter!(ax2, XOR_X1, XOR_X2, color = marker_colors, marker = marker_shapes,
            markersize = 12, strokewidth = 2, strokecolor = :white)
    end

    # Colorbars with readable labels
    Colorbar(fig[1, 3], colormap = Reverse(:RdBu), limits = (0.0, 1.0),
        label = "Mean", labelsize = 13, width = 14, ticklabelsize = 11)
    Colorbar(fig[2, 3], colormap = :inferno, limits = (0.0, std_max),
        label = "Std", labelsize = 13, width = 14, ticklabelsize = 11)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)

    # Save at high DPI
    outdir = joinpath(@__DIR__, "..", "Lukashchuk-2026-Learn-Experts-Infer-Gates", "figures")
    mkpath(outdir)
    save(joinpath(outdir, "xor_uncertainty.pdf"), fig; pt_per_unit = 1)
    save(joinpath(outdir, "xor_uncertainty.png"), fig; px_per_unit = 6)  # 600*6 = 3600px wide
    println("Saved to $(outdir)/xor_uncertainty.{pdf,png}")
end

make_figure()
