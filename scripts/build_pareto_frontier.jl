#!/usr/bin/env julia

using JLD2
using Printf
using CairoMakie

const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results_vae")
const FIGURES_DIR = joinpath(@__DIR__, "..", "paper", "figures")
const HORIZONS = [96, 192, 336, 720]
const VALUE_CAP = 1e40

const DATASETS_DEFAULT = [
    ("ETTh1",         "ETTh1",         :univariate),
    ("ETTh2",         "ETTh2",         :univariate),
    ("exchange_rate", "Exchange Rate", :multivariate),
    ("electricity",   "Electricity",   :multivariate),
    ("traffic",       "Traffic",       :multivariate),
]

const ENSEMBLE_MODEL_TYPES = [
    "static",
    "dynamic",
    "dynamic_diagonal",
    "noisy_experts",
    "noisy_experts_diagonal",
    "neural_ensemble",
    "neural_ensemble_big",
]

const ENSEMBLE_MODEL_LABELS = Dict(
    "static" => "Static",
    "dynamic" => "Dyn.",
    "dynamic_diagonal" => "Dyn. Diag.",
    "noisy_experts" => "Noisy",
    "noisy_experts_diagonal" => "Noisy Diag.",
    "neural_ensemble" => "MoE",
    "neural_ensemble_big" => "MoE Big",
)

const MODEL_DIR = Dict(
    "noisy_experts_diagonal" => "noisy_diagonal",
    "noisy_experts" => "noisy_experts",
)

const MODEL_COLORS = Dict(
    "static" => :royalblue,
    "dynamic" => :orangered,
    "dynamic_diagonal" => :forestgreen,
    "noisy_experts" => :darkorchid,
    "noisy_experts_diagonal" => :goldenrod,
    "neural_ensemble" => :deeppink,
    "neural_ensemble_big" => :saddlebrown,
)

# Update this table if you want to change the x-axis counts.
const MODEL_PARAM_COUNTS = Dict(
    "static" => 14,
    "dynamic" => 15_274,
    "dynamic_diagonal" => 938,
    "noisy_experts" => 15_288,
    "noisy_experts_diagonal" => 952,
    "neural_ensemble" => 462,
    "neural_ensemble_big" => 14_600,
)

function result_filename(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    if startswith(model_type, "neural_ensemble")
        return "$(dataset)_h$(horizon)_neural_ensemble.jld2"
    elseif pred_type == :univariate
        return "$(dataset)_h$(horizon)_OT_$(model_type).jld2"
    else
        return "$(dataset)_h$(horizon)_multivariate_$(model_type).jld2"
    end
end

function load_metrics(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    fname = result_filename(dataset, horizon, pred_type, model_type)
    dir_name = get(MODEL_DIR, model_type, model_type)
    fpath = joinpath(RESULTS_DIR, dir_name, fname)
    if !isfile(fpath)
        dir = joinpath(RESULTS_DIR, dir_name)
        !isdir(dir) && return nothing
        prefix = replace(fname, ".jld2" => "")
        matches = filter(f -> startswith(f, prefix) && endswith(f, ".jld2"), readdir(dir))
        isempty(matches) && return nothing
        fpath = joinpath(dir, first(matches))
    end

    metrics = JLD2.load(fpath)["ensemble_metrics"]
    return (
        mse = metrics.mse,
        # build_radar_charts.jl flips the sign before scoring, so keep that here too
        nll = -metrics.nll,
    )
end

clean_val(v, cap) = (v === nothing || !isfinite(v)) ? nothing : min(abs(v), cap)
clean_val_log(v, cap) = (v === nothing || !isfinite(v)) ? nothing : min(log(abs(v)), log(cap))

function radar_polygon_area(radii, angles)
    n = length(radii)
    area = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        area += radii[i] * radii[j] * sin(angles[j] - angles[i])
    end
    return abs(area) / 2.0
end

function score_models(metric::Symbol;
                      datasets = DATASETS_DEFAULT,
                      models = ENSEMBLE_MODEL_TYPES,
                      cap = VALUE_CAP,
                      log_metric = true)
    n_axes = length(datasets)
    raw_data = Dict{String, Vector{Union{Nothing,Float64}}}()

    for model_type in models
        vals = Union{Nothing,Float64}[]
        for (dataset, _, pred_type) in datasets
            horizon_vals = Float64[]
            for horizon in HORIZONS
                metrics = load_metrics(dataset, horizon, pred_type, model_type)
                metrics === nothing && continue
                value = getfield(metrics, metric)
                cleaned = log_metric ? clean_val_log(value, cap) : clean_val(value, cap)
                cleaned !== nothing && push!(horizon_vals, cleaned)
            end
            push!(vals, isempty(horizon_vals) ? nothing : sum(horizon_vals) / length(horizon_vals))
        end
        raw_data[model_type] = vals
    end

    axis_best = fill(Inf, n_axes)
    axis_worst = fill(-Inf, n_axes)
    for axis_idx in 1:n_axes
        for model_type in models
            value = raw_data[model_type][axis_idx]
            value === nothing && continue
            axis_best[axis_idx] = min(axis_best[axis_idx], value)
            axis_worst[axis_idx] = max(axis_worst[axis_idx], value)
        end
    end

    for axis_idx in 1:n_axes
        span = axis_worst[axis_idx] - axis_best[axis_idx]
        if span ≈ 0.0
            axis_worst[axis_idx] = axis_best[axis_idx] + 1.0
        else
            axis_worst[axis_idx] += 0.10 * span
        end
    end

    normalize_val(v, i) = v === nothing ? 0.0 : clamp((v - axis_worst[i]) / (axis_best[i] - axis_worst[i]), 0.0, 1.0)

    θ = collect(LinRange(0, 2π, n_axes + 1))[1:n_axes]
    max_area = radar_polygon_area(ones(n_axes), θ)

    scores = Dict{String, Float64}()
    for model_type in models
        radii = [normalize_val(raw_data[model_type][i], i) for i in 1:n_axes]
        scores[model_type] = 100 * radar_polygon_area(radii, θ) / max_area
    end

    return scores
end

function pareto_frontier(points)
    frontier = typeof(points[1])[]
    for point in points
        dominated = any(other !== point &&
                        other.params <= point.params &&
                        other.score >= point.score &&
                        (other.params < point.params || other.score > point.score)
                        for other in points)
        dominated || push!(frontier, point)
    end
    sort!(frontier, by = point -> point.params)
    return frontier
end

function plot_metric_frontier!(ax, metric_label::String, points)
    frontier = pareto_frontier(points)
    frontier_x = [point.params for point in frontier]
    frontier_y = [point.score for point in frontier]

    scatter!(
        ax,
        [point.params for point in points],
        [point.score for point in points];
        color = [MODEL_COLORS[point.model_type] for point in points],
        markersize = 18,
        strokecolor = :black,
        strokewidth = 1,
    )
    lines!(ax, frontier_x, frontier_y; color = :black, linewidth = 2, linestyle = :dash)

    for point in points
        text!(
            ax,
            point.params,
            point.score;
            text = point.label,
            fontsize = 12,
            align = (:left, :bottom),
            offset = (8, 6),
        )
    end

    ax.title = "$metric_label Pareto Frontier"
    ax.xlabel = "Parameter count"
    ax.ylabel = "Radar area (% of max)"
    ax.xscale = log10
end

function build_pareto_frontier()
    mse_scores = score_models(:mse)
    nll_scores = score_models(:nll)

    mse_points = [
        (
            model_type = model_type,
            label = ENSEMBLE_MODEL_LABELS[model_type],
            params = MODEL_PARAM_COUNTS[model_type],
            score = mse_scores[model_type],
        )
        for model_type in ENSEMBLE_MODEL_TYPES
    ]
    nll_points = [
        (
            model_type = model_type,
            label = ENSEMBLE_MODEL_LABELS[model_type],
            params = MODEL_PARAM_COUNTS[model_type],
            score = nll_scores[model_type],
        )
        for model_type in ENSEMBLE_MODEL_TYPES
    ]

    println("\n--- Pareto Scores ---")
    @printf("%-16s %10s %12s %12s\n", "Model", "Params", "MSE area %", "NLL area %")
    println("-"^56)
    for model_type in ENSEMBLE_MODEL_TYPES
        @printf(
            "%-16s %10d %11.2f%% %11.2f%%\n",
            ENSEMBLE_MODEL_LABELS[model_type],
            MODEL_PARAM_COUNTS[model_type],
            mse_scores[model_type],
            nll_scores[model_type],
        )
    end

    fig = Figure(size = (1500, 700), fontsize = 14)
    ax_mse = Axis(fig[1, 1])
    ax_nll = Axis(fig[1, 2])

    plot_metric_frontier!(ax_mse, "MSE", mse_points)
    plot_metric_frontier!(ax_nll, "NLL", nll_points)

    Label(
        fig[0, 1:2],
        "Pareto frontier: model size vs radar-area score";
        fontsize = 20,
        font = :bold,
        padding = (0, 0, 10, 0),
    )

    mkpath(FIGURES_DIR)
    png_path = joinpath(FIGURES_DIR, "pareto_frontier_models.png")
    pdf_path = joinpath(FIGURES_DIR, "pareto_frontier_models.pdf")
    save(png_path, fig; px_per_unit = 2)
    save(pdf_path, fig)

    println("\nSaved: $png_path")
    println("Saved: $pdf_path")
end

build_pareto_frontier()
