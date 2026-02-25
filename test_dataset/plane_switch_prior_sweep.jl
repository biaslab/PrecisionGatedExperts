#!/usr/bin/env julia

include("generate_plane_switch_dataset.jl")

using ProbabilisticEnsembling
using DataFrames
using CSV
using ExponentialFamily
using Plots
using Statistics

# ── Helpers ───────────────────────────────────────────────────────────────────

function normalized_gamma_probs(γ)
    n_obs = size(γ, 2)
    probs_a = zeros(n_obs)
    for j in 1:n_obs
        m1 = mean(γ[1, j])
        m2 = mean(γ[2, j])
        probs_a[j] = m1 / (m1 + m2)
    end
    return probs_a
end

function fit_model(df; scale = 1e-3, beta_rate = 1e6, iterations = 50)
    priors = Dict{Symbol,Any}(
        :w => [
            MvNormalMeanScalePrecision(zeros(3), scale),
            MvNormalMeanScalePrecision(zeros(3), scale),
        ],
        :τ => [GammaShapeRate(1.0, 1.0), GammaShapeRate(1.0, 1.0)],
        :β => [GammaShapeRate(1.0, beta_rate), GammaShapeRate(1.0, beta_rate)],
    )

    model = ProbabilisticEnsembling.univariate_dynamic_ensemble(
        n_forecasters = 2,
        n_obs = nrow(df),
        priors = priors,
    )

    features = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]
    predictions = permutedims(hcat(df.pred_a, df.pred_b))
    data = (y = df.OT, features = features, predictions = predictions)

    spec = (
        prediction_type = ProbabilisticEnsembling.Univariate(),
        model_type = ProbabilisticEnsembling.Dynamic(),
        inference_iterations = iterations,
        subsample_size = nothing,
        subsample_percentage = nothing,
    )

    constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
    init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
    result = ProbabilisticEnsembling.run_training_rxinfer(
        spec, model, data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    return Dict{Symbol,Any}(
        :w => result.posteriors[:w][end],
        :τ => result.posteriors[:τ][end],
        :β => result.posteriors[:β][end],
    )
end

function probe_grid(posteriors; x1_range, x2_range, n_grid = 50, iterations = 50)
    x1s = range(x1_range...; length = n_grid)
    x2s = range(x2_range...; length = n_grid)
    n_probe = n_grid^2

    feats = Vector{Vector{Float64}}(undef, n_probe)
    k = 0
    for x2 in x2s
        for x1 in x1s
            k += 1
            feats[k] = [1.0, x1, x2]
        end
    end

    infer_probe = ProbabilisticEnsembling.predict_with_model(
        ProbabilisticEnsembling.Univariate(),
        ProbabilisticEnsembling.Dynamic(),
        posteriors;
        n_forecasters = 2,
        n_steps = n_probe,
        prediction_array = [missing for _ in 1:n_probe],
        predictions_test = zeros(2, n_probe),
        features_test = feats,
        prediction_iterations = iterations,
    )

    pA = normalized_gamma_probs(infer_probe.posteriors[:γ][end])
    return x1s, x2s, reshape(pA, n_grid, n_grid)
end

# ── Main ──────────────────────────────────────────────────────────────────────

function main()
    mkpath("test_dataset/viz")

    scale = 1e-3
    beta_rate = 1e6
    iterations = 50
    n_grid = 50
    margin = 0.3

    # ── 1. Linear boundary ────────────────────────────────────────────────────
    df_lin = CSV.read("test_dataset/plane_switch_dataset.csv", DataFrame)
    @info "Fitting plane_switch" n = nrow(df_lin) scale beta_rate

    post_lin = fit_model(df_lin; scale, beta_rate, iterations)
    x1r = (minimum(df_lin.x1) - margin, maximum(df_lin.x1) + margin)
    x2r = (minimum(df_lin.x2) - margin, maximum(df_lin.x2) + margin)
    x1s, x2s, pA = probe_grid(post_lin; x1_range = x1r, x2_range = x2r, n_grid, iterations)

    # True boundary: a*x1 + b*x2 + c = 0
    a, b, c = 1.0, -0.7, 0.1
    xs_bnd = range(x1r...; length = 300)
    ys_bnd = (-a .* xs_bnd .- c) ./ b

    p1 = contourf(x1s, x2s, pA';
        color = :RdBu, clims = (0, 1), levels = 20,
        xlabel = "x₁", ylabel = "x₂",
        title = "Linear boundary (learned)",
        colorbar_title = "P(expert A)",
    )
    contour!(p1, x1s, x2s, pA'; levels = [0.5],
        linewidth = 2.5, linecolor = :green, label = "learned")
    plot!(p1, xs_bnd, ys_bnd;
        linewidth = 2.5, color = :black, linestyle = :dash, label = "true")

    # ── 2. XOR boundary ──────────────────────────────────────────────────────
    df_xor = CSV.read("test_dataset/xor_dataset.csv", DataFrame)
    @info "Fitting XOR" n = nrow(df_xor) scale beta_rate

    post_xor = fit_model(df_xor; scale, beta_rate, iterations)
    x1r_x = (minimum(df_xor.x1) - margin, maximum(df_xor.x1) + margin)
    x2r_x = (minimum(df_xor.x2) - margin, maximum(df_xor.x2) + margin)
    x1s_x, x2s_x, pA_x = probe_grid(post_xor; x1_range = x1r_x, x2_range = x2r_x, n_grid, iterations)

    xs_xor = range(x1r_x...; length = 300)

    p2 = contourf(x1s_x, x2s_x, pA_x';
        color = :RdBu, clims = (0, 1), levels = 20,
        xlabel = "x₁", ylabel = "x₂",
        title = "XOR boundary (learned)",
        colorbar_title = "P(expert A)",
    )
    contour!(p2, x1s_x, x2s_x, pA_x'; levels = [0.5],
        linewidth = 2.5, linecolor = :green, label = "learned")
    plot!(p2, xs_xor, zero.(xs_xor);
        linewidth = 2.5, color = :black, linestyle = :dash, label = "")
    vline!(p2, [0.0];
        linewidth = 2.5, color = :black, linestyle = :dash, label = "true")

    # ── Combined ──────────────────────────────────────────────────────────────
    p_all = plot(p1, p2; layout = (1, 2), size = (1200, 500), margin = 5Plots.mm)
    savefig(p_all, "test_dataset/viz/learned_decision_boundaries.png")
    @info "Saved" path = "test_dataset/viz/learned_decision_boundaries.png"
end

main()
