#!/usr/bin/env julia

include("generate_plane_switch_dataset.jl")

using ProbabilisticEnsembling
using DataFrames
using CSV
using StableRNGs
using ExponentialFamily
using Plots
using Statistics
using Random: randperm

"""
2D hard-switch synthetic setup with noisy regime boundary:
    regime = sign(a*x1 + b*x2 + c + ε)

Runs a sweep over:
  - w prior scale (MvNormalMeanScalePrecision)
  - β prior rate (GammaShapeRate(1, rate))

For each setting, it reports:
  - free-energy trajectory
  - selector sharpness on training data (top-share)
  - selector regime accuracy on training data
  - selector probability of expert A vs signed distance to switching line

Outputs:
  - test_dataset/viz/plane_switch_data.png
  - test_dataset/viz/plane_switch_free_energy_grid.png
  - test_dataset/viz/plane_switch_selector_vs_distance.png
  - test_dataset/plane_switch_prior_sweep_summary.csv
"""

function build_probe_features(line::NTuple{3,Float64}; distances = range(-2.5, 2.5; length = 161))
    a, b, c = line
    nrm = sqrt(a^2 + b^2)
    nvec = [a, b] ./ nrm
    x0 = -c .* [a, b] ./ (a^2 + b^2) # one point on the boundary

    feats = Vector{Vector{Float64}}(undef, length(distances))
    for (k, d) in enumerate(distances)
        x = x0 .+ d .* nvec
        feats[k] = [1.0, x[1], x[2]]
    end
    return collect(distances), feats
end

function normalized_gamma_probs(γ)
    n_obs = size(γ, 2)
    probs_a = zeros(n_obs)
    top_share = zeros(n_obs)
    for j in 1:n_obs
        m1 = mean(γ[1, j])
        m2 = mean(γ[2, j])
        s = m1 + m2
        p1 = m1 / s
        p2 = m2 / s
        probs_a[j] = p1
        top_share[j] = max(p1, p2)
    end
    return probs_a, top_share
end

function run_single_setting(df_train, scale, beta_rate, inference_iterations)
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
        n_obs = nrow(df_train),
        priors = priors,
    )

    features = [[1.0, df_train.x1[i], df_train.x2[i]] for i in 1:nrow(df_train)]
    predictions = permutedims(hcat(df_train.pred_a, df_train.pred_b))
    data = (y = df_train.OT, features = features, predictions = predictions)

    spec = (
        prediction_type = ProbabilisticEnsembling.Univariate(),
        model_type = ProbabilisticEnsembling.Dynamic(),
        inference_iterations = inference_iterations,
        subsample_size = nothing,
        subsample_percentage = nothing,
    )

    constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
    init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
    result = ProbabilisticEnsembling.run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = false,
    )

    γ_train = result.posteriors[:γ][end]
    pA_train, top_share_train = normalized_gamma_probs(γ_train)
    yhat_train = pA_train .* df_train.pred_a .+ (1 .- pA_train) .* df_train.pred_b
    mse_train = mean((yhat_train .- df_train.OT) .^ 2)
    pred_regime = ifelse.(pA_train .>= 0.5, 1, 2)
    selector_acc = mean(pred_regime .== df_train.regime)

    posterior_priors = Dict{Symbol,Any}(
        :w => result.posteriors[:w][end],
        :τ => result.posteriors[:τ][end],
        :β => result.posteriors[:β][end],
    )

    return (
        result = result,
        spec = spec,
        posterior_priors = posterior_priors,
        pA_train = pA_train,
        top_share_train = top_share_train,
        mse_train = mse_train,
        selector_acc = selector_acc,
    )
end

function main()
    rng = StableRNG(2026)
    mkpath("test_dataset/viz")
    sweep_mode = lowercase(get(ENV, "SWEEP_MODE", "quick"))

    # Dataset and line setup.
    line = (1.0, -0.7, 0.1)
    df = generate_plane_switch_data(
        rng;
        n = sweep_mode == "full" ? 2200 : 1400,
        line = line,
        regime_noise = 0.35,
        obs_noise = 0.08,
    )

    # Shuffle and split train set used for fitting the selector.
    perm = randperm(rng, nrow(df))
    df = df[perm, :]
    n_train = Int(round(0.7 * nrow(df)))
    df_train = df[1:n_train, :]

    scales = sweep_mode == "full" ? [1e-1, 1e-2, 1e-3] : [1e-2, 1e-3]
    beta_rates = [1.0, 1e3, 1e6]
    inference_iterations = sweep_mode == "full" ? 80 : 35
    @info "Sweep mode" sweep_mode n_total = nrow(df) n_train scales beta_rates inference_iterations

    # Plot data and boundary.
    a, b, c = line
    xline = range(minimum(df.x1), maximum(df.x1); length = 250)
    yline = (-a .* xline .- c) ./ b
    p_data = scatter(
        df.x1,
        df.x2;
        marker_z = df.regime,
        xlabel = "x1",
        ylabel = "x2",
        markersize = 3,
        alpha = 0.65,
        legend = false,
        title = "2D Regimes with Noisy Linear Boundary",
        colorbar = true,
    )
    plot!(p_data, xline, yline; linewidth = 3, color = :black, label = "")
    savefig(p_data, "test_dataset/viz/plane_switch_data.png")

    fe_plots = Vector{Any}()
    p_dist = plot(
        xlabel = "Signed distance to boundary",
        ylabel = "P(expert A)",
        title = "Selector Probability vs Distance",
        legend = :outerright,
        linewidth = 2,
    )

    distances, probe_features = build_probe_features(line; distances = range(-2.5, 2.5; length = 161))
    summary_rows = DataFrame(
        scale = Float64[],
        beta_rate = Float64[],
        final_free_energy = Float64[],
        free_energy_delta = Float64[],
        train_mse = Float64[],
        selector_accuracy = Float64[],
        mean_top_share = Float64[],
        boundary_top_share = Float64[],
        pA_at_neg_1 = Float64[],
        pA_at_0 = Float64[],
        pA_at_pos_1 = Float64[],
    )

    for s in scales
        for br in beta_rates
            @info "Running setting" scale = s beta_rate = br
            run = run_single_setting(df_train, s, br, inference_iterations)
            fe = run.result.free_energy

            # Probe selector away/near boundary using only learned gating parameters.
            n_probe = length(probe_features)
            infer_probe = ProbabilisticEnsembling.predict_with_model(
                run.spec.prediction_type,
                run.spec.model_type,
                run.posterior_priors;
                n_forecasters = 2,
                n_steps = n_probe,
                prediction_array = [missing for _ in 1:n_probe],
                predictions_test = zeros(2, n_probe),
                features_test = probe_features,
                prediction_iterations = run.spec.inference_iterations,
            )
            γ_probe = infer_probe.posteriors[:γ][end]
            pA_probe, _ = normalized_gamma_probs(γ_probe)

            label = "scale=$(s), β_rate=$(br)"
            plot!(p_dist, distances, pA_probe; label = label)
            p_fe = plot(
                fe;
                xlabel = "Iteration",
                ylabel = "Free energy",
                title = label,
                legend = false,
                linewidth = 2,
            )
            push!(fe_plots, p_fe)

            boundary_mask = abs.(df_train.dist_to_line) .< 0.25
            boundary_share = any(boundary_mask) ? mean(run.top_share_train[boundary_mask]) : NaN

            pA_neg_1 = pA_probe[argmin(abs.(distances .+ 1.0))]
            pA_0 = pA_probe[argmin(abs.(distances))]
            pA_pos_1 = pA_probe[argmin(abs.(distances .- 1.0))]

            push!(summary_rows, (
                s,
                br,
                fe[end],
                fe[1] - fe[end],
                run.mse_train,
                run.selector_acc,
                mean(run.top_share_train),
                boundary_share,
                pA_neg_1,
                pA_0,
                pA_pos_1,
            ))

            @info "Completed setting" scale = s beta_rate = br free_energy_end = fe[end] selector_acc = run.selector_acc mean_top_share = mean(run.top_share_train)
        end
    end

    p_fe_grid = plot(
        fe_plots...;
        layout = (length(scales), length(beta_rates)),
        size = (1400, 950),
        margin = 6Plots.mm,
    )

    savefig(p_fe_grid, "test_dataset/viz/plane_switch_free_energy_grid.png")
    savefig(p_dist, "test_dataset/viz/plane_switch_selector_vs_distance.png")
    CSV.write("test_dataset/plane_switch_prior_sweep_summary.csv", summary_rows)

    sort!(summary_rows, [:selector_accuracy, :mean_top_share], rev = true)
    println("\nTop settings by selector accuracy/top-share:")
    show(summary_rows[1:min(5, nrow(summary_rows)), :], allrows = true, allcols = true)
end

main()
