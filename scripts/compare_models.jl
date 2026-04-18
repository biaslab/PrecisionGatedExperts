#!/usr/bin/env julia
#
# Compare dynamic, static, and neural ensemble (MoE) predictions from paper/results/
#
# Usage:
#   julia --project scripts/compare_models.jl <dataset> <horizon> [--dim <d>]
#
# Examples:
#   julia --project scripts/compare_models.jl ETTh1 96
#   julia --project scripts/compare_models.jl exchange_rate 192 --dim 3
#   julia --project scripts/compare_models.jl electricity 96 --dim 1
#   julia --project scripts/compare_models.jl exchange_rate 192 --dim 3 --show-val
#
# Available datasets (based on paper/results/ contents):
#   Univariate:    ETTh1, ETTh2
#   Multivariate:  exchange_rate, electricity, traffic
#
# Horizons: 96, 192, 336, 720

using ProbabilisticEnsembling
using ExponentialFamily
using JLD2
using Plots
using BayesBase: cov
using Distributions
using StableRNGs
using LinearAlgebra: diag
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results_vae")
const PREDICTION_ITERATIONS = 3
const COMPARE_LEGEND_FONTSIZE = 16
const COMPARE_TITLE_FONTSIZE = 20
const COMPARE_GUIDE_FONTSIZE = 16
const COMPARE_TICK_FONTSIZE = 14
const SUBPLOT_TOP_BOTTOM_MARGIN = 10Plots.mm
const COMBINED_PLOT_MARGIN = 10Plots.mm
const RESULT_MODEL_SPECS = [
    ("Static", "static"),
    ("Noisy Diagonal", "noisy_experts_diagonal"),
]
const RESULT_MODEL_DIRS = Dict(
    "noisy_experts_diagonal" => "noisy_diagonal",
)

# ── CLI parsing ──────────────────────────────────────────────────────────────

function parse_args()
    if length(ARGS) < 2
        println(stderr, """
        Usage: julia --project scripts/compare_models.jl <dataset> <horizon> [--dim <d>] [--show-val]

        Datasets: ETTh1, ETTh2, exchange_rate, electricity, traffic
        Horizons: 96, 192, 336, 720

        Options:
          --dim <d>   Dimension to plot for multivariate datasets (default: 1)
          --show-val  Prepend validation ground truth with a vertical boundary line
        """)
        exit(1)
    end

    dataset = ARGS[1]
    horizon = parse(Int, ARGS[2])

    dim = 1
    show_val = false
    i = 3
    while i <= length(ARGS)
        if ARGS[i] == "--dim" && i + 1 <= length(ARGS)
            dim = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--show-val"
            show_val = true
            i += 1
        else
            println(stderr, "Unknown or incomplete argument: $(ARGS[i])")
            exit(1)
        end
    end

    return (; dataset, horizon, dim, show_val)
end

# ── File discovery ───────────────────────────────────────────────────────────

function find_result_file(dataset::String, horizon::Int, model_type::String)
    dir_name = get(RESULT_MODEL_DIRS, model_type, model_type)
    dir = joinpath(RESULTS_DIR, dir_name)
    isdir(dir) || return nothing

    for f in readdir(dir)
        endswith(f, ".jld2") || continue
        stem = replace(f, ".jld2" => "")
        stem_parts = split(stem, "_")
        model_parts = split(model_type, "_")
        has_exact_suffix =
            length(stem_parts) >= length(model_parts) &&
            stem_parts[(end - length(model_parts) + 1):end] == model_parts
        has_hashed_suffix =
            length(stem_parts) >= length(model_parts) + 1 &&
            stem_parts[(end - length(model_parts)):(end - 1)] == model_parts

        if startswith(f, "$(dataset)_h$(horizon)_") && (has_exact_suffix || has_hashed_suffix)
            return joinpath(dir, f)
        end
    end
    return nothing
end

function find_neural_result_file(dataset::String, horizon::Int)
    dir = joinpath(RESULTS_DIR, "neural_ensemble")
    isdir(dir) || return nothing

    for f in readdir(dir)
        endswith(f, ".jld2") || continue
        if startswith(f, "$(dataset)_h$(horizon)_neural_ensemble_")
            return joinpath(dir, f)
        end
    end
    return nothing
end

# ── Run prediction from a saved .jld2 ───────────────────────────────────────

function run_prediction(path::String)
    saved = JLD2.load(path)
    spec_saved = saved["spec"]

    prediction_type = ProbabilisticEnsembling._parse_prediction_type(string(spec_saved.prediction_type))
    model_type = ProbabilisticEnsembling._parse_model_type(string(spec_saved.model_type))

    spec_for_data = ProbabilisticEnsembling._spec_for_prediction_from_saved(saved, PREDICTION_ITERATIONS)
    experts = spec_for_data.experts
    selected_quantiles = spec_for_data.selected_quantiles

    y_val_all, y_test_all, _, predictions_test_all, _, features_test_all =
        ProbabilisticEnsembling.before_rxinfer(spec_for_data)
    n_steps = length(y_test_all)

    y_test = ProbabilisticEnsembling.prepare_y_test(prediction_type, y_test_all, n_steps)
    Y_val_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_val_all)
    predictions_test = predictions_test_all
    features_test = features_test_all

    n_forecasters = size(predictions_test, 1)

    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved)
    ProbabilisticEnsembling.prepare_priors!(prediction_type, model_type, priors, predictions_test)
    @info "Running prediction for $(basename(path))..." prediction_mode =
        ProbabilisticEnsembling.prediction_mode_name(spec_for_data.prediction_mode)
    ensemble_preds, test_posteriors = ProbabilisticEnsembling._predict_test(
        spec_for_data.prediction_mode,
        prediction_type,
        model_type,
        priors;
        n_forecasters = n_forecasters,
        n_test = n_steps,
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = PREDICTION_ITERATIONS,
    )

    influence = test_posteriors[:γ]

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test)

    _, _, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics,
    )

    return (;
        ensemble_preds,
        ensemble_metrics,
        influence,
        Y_for_metrics,
        Y_val_for_metrics,
        prediction_type,
        model_type,
        experts,
        selected_quantiles,
        n_forecasters,
    )
end

# ── Run neural ensemble prediction from a saved .jld2 ─────────────────────

function run_neural_prediction(path::String)
    @info "Running neural ensemble prediction for $(basename(path))..."
    return predict_from_trained_neural_ensemble(path)
end

# ── Plotting helpers ─────────────────────────────────────────────────────────

function marginal_mean_std_multivariate(preds, dim)
    μ = [mean(p)[dim] for p in preds]
    σ = [sqrt(cov(p)[dim, dim]) for p in preds]
    return μ, σ
end

function marginal_mean_std_univariate(preds)
    μ = [mean(p) for p in preds]
    σ = [std(p) for p in preds]
    return μ, σ
end

function is_multivariate(prediction_type)
    return prediction_type isa ProbabilisticEnsembling.Multivariate
end

function get_y_true(Y_for_metrics, prediction_type, dim)
    if is_multivariate(prediction_type)
        return Y_for_metrics[dim, :]
    else
        return Y_for_metrics
    end
end

function get_mean_std(preds, prediction_type, dim)
    if is_multivariate(prediction_type)
        return marginal_mean_std_multivariate(preds, dim)
    else
        return marginal_mean_std_univariate(preds)
    end
end

function shorten_expert_name(path::String)
    name = replace(basename(path), ".jld2" => "")
    # Extract architecture from patterns like ETTh1_h96_s96_CNN_enzyme
    m = match(r"_s\d+_(\w+?)_", name)
    if !isnothing(m)
        model_name = m.captures[1]
        return model_name == "MLP" ? "NLinear" : model_name
    end
    # Fallback: last meaningful part
    parts = split(name, "_")
    model_name = length(parts) >= 2 ? parts[end-1] : name
    return model_name == "MLP" ? "NLinear" : model_name
end

function build_forecaster_labels(experts, selected_quantiles, n_forecasters)
    expert_labels = [shorten_expert_name(e) for e in experts]
    quantile_labels = ["q$(Int(round(q)))" for q in selected_quantiles]
    labels = vcat(expert_labels, quantile_labels)
    if length(labels) < n_forecasters
        for k in (length(labels)+1):n_forecasters
            push!(labels, "E$k")
        end
    end
    return labels[1:n_forecasters]
end

# ── Influence plots (dynamic models have time-varying γ, static have constant) ──

function plot_influence_dynamic!(p, influence, forecaster_labels, n_f, n_t, expert_colors)
    normalized_samples = normalized_influence_dist(StableRNG(42), influence, 100)

    norm_mean = dropdims(mean(normalized_samples, dims=1), dims=1)
    norm_lo = zeros(n_f, n_t)
    norm_hi = zeros(n_f, n_t)
    for i in 1:n_f, j in 1:n_t
        samples_ij = normalized_samples[:, i, j]
        norm_lo[i, j] = quantile(samples_ij, 0.025)
        norm_hi[i, j] = quantile(samples_ij, 0.975)
    end

    for i in 1:n_f
        plot!(p, 1:n_t, norm_mean[i, :],
            label=forecaster_labels[i], color=expert_colors[i], linewidth=1.5)
        plot!(p, 1:n_t, norm_hi[i, :],
            fillrange=norm_lo[i, :],
            fillalpha=0.15, linealpha=0, label="", color=expert_colors[i])
    end

    top_share = vec(maximum(norm_mean; dims=1))
    return top_share
end

function plot_influence_static!(p, influence, forecaster_labels, n_f, expert_colors)
    # Static γ posteriors are a vector of distributions (one per forecaster), not time-varying
    n_mc = 1000
    rng = StableRNG(42)
    raw_samples = [rand(rng, influence[i], n_mc) for i in 1:n_f]
    totals = sum(hcat(raw_samples...), dims=2)
    norm_means = [mean(raw_samples[i] ./ vec(totals)) for i in 1:n_f]

    bar!(p, forecaster_labels, norm_means,
        color=[expert_colors[i] for i in 1:n_f],
        label="", ylabel="Normalized γ", xlabel="Expert")

    return maximum(norm_means)
end

function is_time_varying_influence(influence)
    return ndims(influence) > 1
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    args = parse_args()
    dataset = args.dataset
    horizon = args.horizon
    dim = args.dim
    show_val = args.show_val

    neural_path = find_neural_result_file(dataset, horizon)

    bayes_paths = Dict{String,Union{Nothing,String}}(
        label => find_result_file(dataset, horizon, model_type)
        for (label, model_type) in RESULT_MODEL_SPECS
    )

    if all(isnothing, values(bayes_paths)) && isnothing(neural_path)
        println(stderr, "No result files found for dataset=$dataset horizon=$horizon")
        println(stderr, "Available files in paper/results/:")
        for d in ["static", "noisy_diagonal", "neural_ensemble"]
            dir = joinpath(RESULTS_DIR, d)
            isdir(dir) && for f in readdir(dir)
                println(stderr, "  $d/$f")
            end
        end
        exit(1)
    end

    # Run predictions for Bayesian ensemble models (RxInfer-based)
    results = Dict{String,Any}()
    for (label, _) in RESULT_MODEL_SPECS
        path = bayes_paths[label]
        isnothing(path) && continue
        @info "Loading $(label) model: $(basename(path))"
        results[label] = run_prediction(path)
    end

    # Run neural ensemble prediction (separate because data shape differs)
    neural_result = nothing
    if !isnothing(neural_path)
        @info "Loading neural ensemble: $(basename(neural_path))"
        neural_result = run_neural_prediction(neural_path)
    end

    # Determine prediction type and ground truth
    if !isempty(results)
        first_result = first(values(results))
        prediction_type = first_result.prediction_type
        Y_for_metrics = first_result.Y_for_metrics
    elseif !isnothing(neural_result)
        prediction_type = neural_result.prediction_type
        if is_multivariate(prediction_type)
            Y_for_metrics = neural_result.y_test_mat
        else
            ci = neural_result.col_idx
            Y_for_metrics = vec(neural_result.y_test_mat[ci, :])
        end
    else
        println(stderr, "No results available")
        exit(1)
    end

    is_mv = is_multivariate(prediction_type)
    if is_mv
        n_dims = size(Y_for_metrics, 1)
        if dim > n_dims
            println(stderr, "Dimension $dim out of range (max $n_dims)")
            exit(1)
        end
        dim_label = " (dim=$dim)"
    else
        dim = 1
        dim_label = ""
    end

    y_true = get_y_true(Y_for_metrics, prediction_type, dim)
    n_test = length(y_true)

    if show_val
        # Validation ground truth is only available from dynamic/static results
        val_result = nothing
        for (_, res) in results
            if hasproperty(res, :Y_val_for_metrics) && res.Y_val_for_metrics !== nothing
                val_result = res
                break
            end
        end
        if !isnothing(val_result)
            y_val = get_y_true(val_result.Y_val_for_metrics, prediction_type, dim)
            n_val = length(y_val)
            y_plot = vcat(y_val, y_true)
            t_test = (n_val+1):(n_val+n_test)
        else
            @warn "--show-val requested but validation ground truth not available; disabling"
            show_val = false
            y_plot = y_true
            n_val = 0
            t_test = 1:n_test
        end
    else
        y_plot = y_true
        n_val = 0
        t_test = 1:n_test
    end

    subplot_order = [label for (label, _) in RESULT_MODEL_SPECS if haskey(results, label)]

    # ── Top plot: predictions comparison ─────────────────────────────────
    p1 = plot(1:length(y_plot), y_plot, label="Ground Truth", color=:black, linewidth=2,
        legend=:topright, ylabel="Value",
        title="$(dataset) h=$(horizon)$(dim_label)",
        legendfontsize=COMPARE_LEGEND_FONTSIZE,
        titlefontsize=COMPARE_TITLE_FONTSIZE,
        guidefontsize=COMPARE_GUIDE_FONTSIZE,
        tickfontsize=COMPARE_TICK_FONTSIZE,
        top_margin=SUBPLOT_TOP_BOTTOM_MARGIN,
        bottom_margin=SUBPLOT_TOP_BOTTOM_MARGIN)

    if show_val
        vline!(p1, [n_val + 0.5], color=:gray, linestyle=:dash, linewidth=2,
            label="Val / Test boundary")
    end

    model_colors = Dict(
        "Static" => :red,
        "Noisy Diagonal" => :goldenrod,
    )
    for name in subplot_order
        res = results[name]
        μ, σ = get_mean_std(res.ensemble_preds, prediction_type, dim)
        c = model_colors[name]
        m = res.ensemble_metrics
        plot!(p1, t_test, μ,
            label="$name (MSE=$(round(m.mse, digits=4)), MAE=$(round(m.mae, digits=4)))",
            color=c, linewidth=1.5)
        plot!(p1, t_test, μ .+ 1.96 .* σ, fillrange=μ .- 1.96 .* σ,
            fillalpha=0.15, linealpha=0, label="", color=c)
    end

    # Add neural ensemble (MoE) predictions using Bayesian posterior normal_predictions
    if !isnothing(neural_result)
        μ_n, σ_n = get_mean_std(neural_result.normal_predictions, neural_result.prediction_type, dim)
        m = neural_result.ensemble_metrics
        plot!(p1, t_test, μ_n,
            label="MoE (MSE=$(round(m.mse, digits=4)), MAE=$(round(m.mae, digits=4)))",
            color=:green, linewidth=1.5)
        plot!(p1, t_test, μ_n .+ 1.96 .* σ_n, fillrange=μ_n .- 1.96 .* σ_n,
            fillalpha=0.15, linealpha=0, label="", color=:green)
    end

    # ── Bottom plots: influence per model ────────────────────────────────
    subplots = [p1]

    for name in subplot_order
        res = results[name]
        n_f = res.n_forecasters
        labels = build_forecaster_labels(res.experts, res.selected_quantiles, n_f)
        expert_colors = distinguishable_colors(n_f, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

        if is_time_varying_influence(res.influence)
            n_t = size(res.influence, 2)
            p_inf = plot(title="$(name) — Gammas",
                ylabel="Normalized influence", xlabel="Time Step",
                legend=:topright,
                legendfontsize=COMPARE_LEGEND_FONTSIZE,
                titlefontsize=COMPARE_TITLE_FONTSIZE,
                guidefontsize=COMPARE_GUIDE_FONTSIZE,
                tickfontsize=COMPARE_TICK_FONTSIZE,
                top_margin=SUBPLOT_TOP_BOTTOM_MARGIN,
                bottom_margin=SUBPLOT_TOP_BOTTOM_MARGIN)
            top_share = plot_influence_dynamic!(p_inf, res.influence, labels, n_f, n_t, expert_colors)
            push!(subplots, p_inf)

            p_ts = plot(1:n_t, top_share,
                title="$(name) — TopShare",
                xlabel="Time Step", ylabel="Top-1 normalized γ",
                color=:darkblue, linewidth=2, ylims=(0.0, 1.0),
                label="max γᵢ / Σγ", legend=:topright,
                legendfontsize=COMPARE_LEGEND_FONTSIZE,
                titlefontsize=COMPARE_TITLE_FONTSIZE,
                guidefontsize=COMPARE_GUIDE_FONTSIZE,
                tickfontsize=COMPARE_TICK_FONTSIZE,
                top_margin=SUBPLOT_TOP_BOTTOM_MARGIN,
                bottom_margin=SUBPLOT_TOP_BOTTOM_MARGIN)
            push!(subplots, p_ts)
        else
            p_inf = plot(title="Static — Gammas",
                ylabel="Normalized γ", xlabel="Expert",
                legend=false, xrotation=45,
                titlefontsize=COMPARE_TITLE_FONTSIZE,
                guidefontsize=COMPARE_GUIDE_FONTSIZE,
                tickfontsize=COMPARE_TICK_FONTSIZE,
                top_margin=SUBPLOT_TOP_BOTTOM_MARGIN,
                bottom_margin=SUBPLOT_TOP_BOTTOM_MARGIN)
            plot_influence_static!(p_inf, res.influence, labels, n_f, expert_colors)
            push!(subplots, p_inf)
        end
    end

    # Add neural ensemble gating weights subplot
    if !isnothing(neural_result)
        gw = neural_result.gating_weights  # n_experts × n_test (already softmax-normalized)
        n_f = neural_result.n_forecasters
        n_t = size(gw, 2)
        labels = build_forecaster_labels(neural_result.experts, neural_result.selected_quantiles, n_f)
        expert_colors = distinguishable_colors(n_f, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

        p_gating = plot(title="MoE — Gating Weights",
            ylabel="Softmax prob.", xlabel="Time Step",
            legend=:topright,
            legendfontsize=COMPARE_LEGEND_FONTSIZE,
            titlefontsize=COMPARE_TITLE_FONTSIZE,
            guidefontsize=COMPARE_GUIDE_FONTSIZE,
            tickfontsize=COMPARE_TICK_FONTSIZE,
            top_margin=SUBPLOT_TOP_BOTTOM_MARGIN,
            bottom_margin=SUBPLOT_TOP_BOTTOM_MARGIN)
        for i in 1:n_f
            plot!(p_gating, 1:n_t, gw[i, :],
                label=labels[i], color=expert_colors[i], linewidth=1.5)
        end
        push!(subplots, p_gating)
    end

    # ── Layout ───────────────────────────────────────────────────────────
    n_sub = length(subplots)
    if n_sub == 1
        p = plot(subplots[1], size=(2400, 800), dpi=600, margin=COMBINED_PLOT_MARGIN)
    elseif n_sub == 2
        l = @layout [a; b]
        p = plot(subplots..., layout=l, size=(2400, 1400), margin=COMBINED_PLOT_MARGIN, dpi=600)
    elseif n_sub == 3
        l = @layout [a; b; c]
        p = plot(subplots..., layout=l, size=(2400, 1800), margin=COMBINED_PLOT_MARGIN, dpi=600)
    elseif n_sub == 4
        l = @layout [a; b; c d]
        p = plot(subplots..., layout=l, size=(2400, 1800), margin=COMBINED_PLOT_MARGIN, dpi=600)
    elseif n_sub == 5
        l = @layout [a; b; c d; e]
        p = plot(subplots..., layout=l, size=(2400, 2200), margin=COMBINED_PLOT_MARGIN, dpi=600)
    else
        p = plot(subplots..., layout=(n_sub, 1), size=(2400, 600 * n_sub), margin=COMBINED_PLOT_MARGIN, dpi=600)
    end

    outstem = "compare_$(dataset)_h$(horizon)_dim$(dim)"
    outfile_png = "$(outstem).png"
    outfile_pdf = "$(outstem).pdf"
    savefig(p, outfile_png)
    savefig(p, outfile_pdf)
    @info "Saved plots" png = outfile_png pdf = outfile_pdf
    display(p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
