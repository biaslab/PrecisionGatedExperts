#!/usr/bin/env julia
# Plot dependence of forecasting metrics on final Free Energy.
#
# Usage:
#   julia --project=. scripts/plot_fe_vs_metrics.jl
#
# Reads all JLD2 results from paper/results/dependence_on_iteration/
# and produces scatter plots: NLL vs FE, MSE vs FE, MAE vs FE,
# plus CI95 coverage vs FE and CI95 width vs FE.
# Points are coloured by dataset and shaped by horizon.

using JLD2
using CairoMakie

# ── Load data ─────────────────────────────────────────────────────────────────

const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results", "dependence_on_iteration")

struct RunResult
    dataset::String
    horizon::Int
    iterations::Int
    fe_final::Float64
    mse::Float64
    mae::Float64
    nll::Float64
    ci95_overlap::Float64
    ci95_width::Float64
    ci95_iscore::Float64
end

function load_all_runs(dir)
    runs = RunResult[]
    for f in readdir(dir)
        endswith(f, ".jld2") || continue
        data = JLD2.load(joinpath(dir, f))
        rs   = data["raw_spec"]["params"]
        m    = data["ensemble_metrics"]
        fe   = data["free_energy"]
        push!(runs, RunResult(
            rs["dataset"],
            rs["horizon"],
            rs["inference_iterations"],
            Float64(fe[end]),
            m.mse, m.mae, m.nll,
            m.ci95_target_overlap,
            m.ci95_avg_width,
            m.ci95_interval_score,
        ))
    end
    return runs
end

runs = load_all_runs(RESULTS_DIR)
@info "Loaded $(length(runs)) runs"

# ── Organise by dataset & horizon ─────────────────────────────────────────────

datasets = sort(unique(r.dataset for r in runs))
horizons = sort(unique(r.horizon for r in runs))

dataset_colors = Dict(
    "ETTh1" => :dodgerblue,
    "ETTh2" => :orangered,
)
# Fallback for any extra datasets
for (i, ds) in enumerate(datasets)
    haskey(dataset_colors, ds) || (dataset_colors[ds] = Makie.wong_colors()[i])
end

horizon_markers = Dict(
    96  => :circle,
    192 => :utriangle,
    336 => :rect,
    720 => :diamond,
)
for (i, h) in enumerate(horizons)
    haskey(horizon_markers, h) || (horizon_markers[h] = :star5)
end

# ── Iteration label annotations ──────────────────────────────────────────────

iter_to_label = Dict(1 => "1", 5 => "5", 10 => "10", 15 => "15")

# ── Helper: scatter one metric vs FE ─────────────────────────────────────────

function plot_metric_vs_fe!(ax, runs, metric_fn, ylabel;
                            annotate_iters=true)
    for ds in datasets
        for h in horizons
            subset = filter(r -> r.dataset == ds && r.horizon == h, runs)
            isempty(subset) && continue

            xs = [r.fe_final for r in subset]
            ys = [metric_fn(r) for r in subset]

            scatter!(ax, xs, ys;
                color  = dataset_colors[ds],
                marker = horizon_markers[h],
                markersize = 14,
                strokewidth = 0.5,
                strokecolor = :black,
            )

            # connect points within same dataset+horizon with a thin line
            order = sortperm(xs)
            lines!(ax, xs[order], ys[order];
                color     = (dataset_colors[ds], 0.3),
                linewidth = 1.0,
            )

            # annotate iteration count next to each point
            if annotate_iters
                for r in subset
                    text!(ax, r.fe_final, metric_fn(r);
                        text   = iter_to_label[r.iterations],
                        fontsize = 8,
                        color  = :gray40,
                        offset = (5, 5),
                    )
                end
            end
        end
    end
    ax.xlabel = "Free Energy"
    ax.ylabel = ylabel
end

# ── Build figure ──────────────────────────────────────────────────────────────

fig = Figure(size = (1400, 1000), fontsize = 13)

ax_nll  = Axis(fig[1, 1]; title = "NLL vs Free Energy",
               yscale = log10)
ax_mse  = Axis(fig[1, 2]; title = "MSE vs Free Energy")
ax_mae  = Axis(fig[2, 1]; title = "MAE vs Free Energy")
ax_ci   = Axis(fig[2, 2]; title = "CI95 score vs Free Energy")

# NLL was saved with wrong sign — flip it
plot_metric_vs_fe!(ax_nll, runs, r -> -r.nll,       "NLL")
plot_metric_vs_fe!(ax_mse, runs, r -> r.mse,        "MSE")
plot_metric_vs_fe!(ax_mae, runs, r -> r.mae,        "MAE")
plot_metric_vs_fe!(ax_ci,  runs, r -> r.ci95_iscore, "CI95 score")

# ── Legend ────────────────────────────────────────────────────────────────────

# Dataset legend
dataset_elements = [MarkerElement(; color = dataset_colors[ds], marker = :circle,
                                   markersize = 12) for ds in datasets]
# Horizon legend
horizon_elements = [MarkerElement(; color = :gray50, marker = horizon_markers[h],
                                   markersize = 12) for h in horizons]

Legend(fig[3, 1:2],
    [dataset_elements..., horizon_elements...],
    [datasets..., ["h=$h" for h in horizons]...];
    orientation = :horizontal,
    tellheight  = true,
    tellwidth   = false,
    framevisible = false,
    nbanks = 1,
    labelsize = 12,
)

# ── Supertitle ────────────────────────────────────────────────────────────────

Label(fig[0, 1:2],
    "Does Free Energy measure forecasting quality?";
    fontsize  = 18,
    font      = :bold,
    halign    = :center,
    padding   = (0, 0, 0, 10),
)

outpath = joinpath(@__DIR__, "..", "fe_vs_metrics.png")
save(outpath, fig; px_per_unit = 2)
@info "Saved" outpath
