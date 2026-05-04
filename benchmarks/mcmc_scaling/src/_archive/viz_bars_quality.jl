# Grouped bar chart: (N, D) groups × 4 methods per group.
# Two panels: rel-L2 error (top) and wall-clock time (bottom, log scale).

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

mcmc = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv"), DataFrame)
vmp  = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_vmp.csv"),  DataFrame)
vmp_conv = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_vmp_converged.csv"), DataFrame)

for df in (mcmc, vmp, vmp_conv)
    if !("status" in names(df)); df.status = fill("OK", nrow(df)); end
end

cols = [:method, :N, :D, :rel_err_total, :total_time_s, :cov95, :status]
all = vcat(mcmc[:, cols], vmp[:, cols], vmp_conv[:, cols])
filter!(r -> r.status == "OK", all)

METHODS = ["Gibbs", "NUTS", "VMP-dynamic", "VMP-converged"]
method_label = Dict(
    "Gibbs"         => "Gibbs",
    "NUTS"          => "NUTS",
    "VMP-dynamic"   => "VMP-10",
    "VMP-converged" => "VMP-conv",
)
method_color = Dict(
    "Gibbs"         => :crimson,
    "NUTS"          => :seagreen,
    "VMP-dynamic"   => :darkorange,
    "VMP-converged" => :purple,
)

N_VALUES = sort(unique(all.N))
D_VALUES = sort(unique(all.D))
# Group order: primary = N, secondary = D
group_configs = [(N, D) for N in N_VALUES for D in D_VALUES]
nGroups  = length(group_configs)
nMethods = length(METHODS)
nTotal   = nGroups * nMethods

# x-positions: each group has nMethods bars packed together, with a gap between groups
bar_width = 0.8
group_gap = 1.2

function build_bars(group_configs, METHODS, all, bar_width, group_gap, method_color, method_label)
    positions = Float64[]
    colors    = Symbol[]
    labels    = String[]
    err_vals  = Float64[]
    time_vals = Float64[]
    cov_vals  = Float64[]
    group_centers = Float64[]
    x0 = 0.0
    for (N, D) in group_configs
        group_start = x0
        for m in METHODS
            row = filter(r -> r.N == N && r.D == D && r.method == m, all)
            ev = nrow(row) == 1 ? row[1, :rel_err_total]  : NaN
            tv = nrow(row) == 1 ? row[1, :total_time_s]   : NaN
            cv = nrow(row) == 1 ? row[1, :cov95]          : NaN
            push!(positions, x0)
            push!(colors, method_color[m])
            push!(labels, method_label[m])
            push!(err_vals, ev)
            push!(time_vals, tv)
            push!(cov_vals, cv)
            x0 += bar_width
        end
        push!(group_centers, (group_start + (x0 - bar_width)) / 2)
        x0 += group_gap
    end
    return (positions, colors, labels, err_vals, time_vals, cov_vals, group_centers)
end

positions, colors, labels, err_vals, time_vals, cov_vals, group_centers =
    build_bars(group_configs, METHODS, all, bar_width, group_gap, method_color, method_label)

fig = Figure(size=(2400, 700), fontsize=14)

Label(fig[0, 1:3],
      "Quality & cost per config — bars grouped by (N, D)",
      fontsize=17, font=:bold)

xticks_labels = (group_centers, ["N=$N\nD=$D" for (N, D) in group_configs])

# ---------- Panel 1: rel-L2 error ----------
ax_err = Axis(fig[1, 1];
              title = "rel-L2 error (lower = better)",
              ylabel = "rel-L2 error",
              xticks = xticks_labels, xticklabelsize = 10)
barplot!(ax_err, positions, err_vals; color=colors, width=bar_width)
hlines!(ax_err, [1.0]; color=:gray, linestyle=:dash, linewidth=0.8)

# ---------- Panel 2: wall-clock time (log) ----------
ax_time = Axis(fig[1, 2];
               title = "wall-clock time (s)  [log]",
               ylabel = "time (s)",
               xticks = xticks_labels, xticklabelsize = 10, yscale = log10)
barplot!(ax_time, positions, time_vals; color=colors, width=bar_width)

# ---------- Panel 3: 95% CI coverage ----------
ax_cov = Axis(fig[1, 3];
              title = "95% CI coverage (higher = better, target 0.95)",
              ylabel = "coverage frac",
              xticks = xticks_labels, xticklabelsize = 10)
barplot!(ax_cov, positions, cov_vals; color=colors, width=bar_width)
hlines!(ax_cov, [0.95]; color=:gray, linestyle=:dash, linewidth=0.8)
ylims!(ax_cov, 0, 1.05)

# Legend (common, below all panels)
leg_elems = [PolyElement(color=method_color[m], strokecolor=:black, strokewidth=0.3)
             for m in METHODS]
leg_labels = [method_label[m] for m in METHODS]
Legend(fig[2, 1:3], leg_elems, leg_labels;
       framevisible=true, orientation=:horizontal, tellwidth=false, tellheight=true)

save(joinpath(RESULTS_DIR, "bars_quality.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "bars_quality.pdf"), fig)
println("Saved bars_quality.{png,pdf}")
