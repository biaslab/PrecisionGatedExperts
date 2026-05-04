# Updated figure: NUTS vs VMP-FE-converged vs VMP-10, full (N, D) grid.
# 2 rows × 2 columns:
#   row 1 = wall-clock vs D
#   row 2 = w-only WIS vs D
# columns = N = 50, N = 250
# NUTS: red circles. VMP-FE-converged (rtol=1e-4): green diamonds.
# VMP-10 (fixed 10 iters): orange squares.
# Cells where VMP-FE-converged produced numerically unstable predictive
# (CRPS > 10, indicating posterior-predictive blow-up) are marked with
# an open marker.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CairoMakie
using CSV, DataFrames
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

# ----- NUTS cells -------
v2      = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_mcmc.csv"),     DataFrame)
focus   = CSV.read(joinpath(RESULTS_DIR, "ess_vs_D_focused.csv"),    DataFrame)
pareto  = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv"), DataFrame)
stretch = CSV.read(joinpath(RESULTS_DIR, "nuts_wis_stretch.csv"),    DataFrame)
nfill   = CSV.read(joinpath(RESULTS_DIR, "nuts_fill_n250.csv"),      DataFrame)
n50fill = CSV.read(joinpath(RESULTS_DIR, "nuts_fill_n50.csv"),       DataFrame)
n250d10 = CSV.read(joinpath(RESULTS_DIR, "nuts_wis_n250_d10.csv"),   DataFrame)

# Combine NUTS times — include new n50fill cells (more accurate WIS-measurement times)
common = [:method, :N, :D, :total_time_s]
nuts_time = vcat(v2[:, common], focus[:, common], pareto[:, common],
                  stretch[:, common], nfill[:, common], n50fill[:, common])
filter!(r -> r.method == "NUTS", nuts_time)
unique!(nuts_time, [:N, :D]); sort!(nuts_time, [:N, :D])

# Combine NUTS WIS_w (only have it where we measured it)
nuts_wis_rows = vcat(
    DataFrame(method=["NUTS"], N=[250], D=[10], wis_w=[n250d10.wis_w[1]]),
    select(stretch, :method, :N, :D, :wis_w),
    select(nfill,   :method, :N, :D, :wis_w),
    select(n50fill, :method, :N, :D, :wis_w),
)
filter!(r -> r.method == "NUTS", nuts_wis_rows)
unique!(nuts_wis_rows, [:N, :D])

# ----- VMP-FE-converged -------
vmpfe = CSV.read(joinpath(RESULTS_DIR, "vmp_fe_converged_sweep.csv"), DataFrame)

# ----- VMP-10 (from pareto_wis + bigD sweeps) -------
p1 = CSV.read(joinpath(RESULTS_DIR, "pareto_wis.csv"),      DataFrame)
p2 = CSV.read(joinpath(RESULTS_DIR, "pareto_wis_bigD.csv"), DataFrame)
vmp10 = vcat(p1, p2; cols=:union)
filter!(r -> r.method == "VMP-dynamic" && r.status == "OK", vmp10)
unique!(vmp10, [:N, :D]); sort!(vmp10, [:N, :D])

Nticks = [2, 5, 10, 25, 65, 100]

fig = Figure(size = (1800, 1100), fontsize = 14)
Label(fig[0, 1:2],
      "NUTS vs VMP-FE-converged vs VMP-10 across D  (N = 50 left, N = 250 right)",
      fontsize=17, font=:bold)

function timepanel!(gp, N::Int, title::String)
    ax = Axis(gp;
              title  = title, xlabel = "D (log)", ylabel = "wall-clock (s, log)",
              xscale = log10, yscale = log10,
              xticks = (Nticks, string.(Nticks)),
              yticks = ([1, 10, 100, 1000, 10000, 100000],
                        ["1 s", "10 s", "100 s", "1000 s", "10⁴ s", "10⁵ s"]))
    limits!(ax, (1.6, 130), (3, 2e5))
    # NUTS
    subN = sort(filter(r -> r.N == N, nuts_time), :D)
    lines!(ax, subN.D, subN.total_time_s; color=:firebrick, linewidth=3, label="NUTS")
    scatter!(ax, subN.D, subN.total_time_s; color=:firebrick, markersize=17, strokecolor=:black, strokewidth=1)
    # VMP-FE-converged
    subV = sort(filter(r -> r.N == N, vmpfe), :D)
    lines!(ax, subV.D, subV.total_time_s; color=:seagreen, linewidth=3, label="VMP-FE-conv")
    scatter!(ax, subV.D, subV.total_time_s; color=:seagreen, marker=:diamond, markersize=16,
             strokecolor=:black, strokewidth=1)
    # VMP-10 fixed
    subS = sort(filter(r -> r.N == N, vmp10), :D)
    lines!(ax, subS.D, subS.total_time_s; color=:darkorange, linewidth=2.6, linestyle=:dash,
           label="VMP-10 (fixed 10 iters)")
    scatter!(ax, subS.D, subS.total_time_s; color=:darkorange, marker=:rect, markersize=13,
             strokecolor=:black, strokewidth=1)
    axislegend(ax; position=:lt, labelsize=11, backgroundcolor=(:white, 0.9))
end

function wispanel!(gp, N::Int, title::String)
    ax = Axis(gp;
              title  = title, xlabel = "D (log)", ylabel = "WIS_w (log, lower = better)",
              xscale = log10, yscale = log10,
              xticks = (Nticks, string.(Nticks)),
              yticks = ([0.03, 0.1, 0.3, 1.0, 3.0], ["0.03", "0.1", "0.3", "1.0", "3.0"]))
    limits!(ax, (1.6, 130), (0.03, 4.0))
    # NUTS WIS (only where measured)
    subN = sort(filter(r -> r.N == N, nuts_wis_rows), :D)
    if !isempty(subN)
        lines!(ax, subN.D, subN.wis_w; color=:firebrick, linewidth=3, label="NUTS")
        scatter!(ax, subN.D, subN.wis_w; color=:firebrick, markersize=17, strokecolor=:black, strokewidth=1)
    end
    # VMP-FE-converged WIS_w; open markers where predictive blew up
    subV = sort(filter(r -> r.N == N, vmpfe), :D)
    if !isempty(subV)
        stable = subV.crps .< 10
        lines!(ax, subV.D, subV.wis_w; color=:seagreen, linewidth=3, label="VMP-FE-conv")
        scatter!(ax, subV.D[stable],   subV.wis_w[stable];
                 color=:seagreen, marker=:diamond, markersize=16,
                 strokecolor=:black, strokewidth=1)
        # unstable markers: open (white-filled) diamonds
        unstable = .!stable
        scatter!(ax, subV.D[unstable], subV.wis_w[unstable];
                 color=:white, marker=:diamond, markersize=16,
                 strokecolor=:seagreen, strokewidth=2.5)
    end
    # VMP-10 WIS
    subS = sort(filter(r -> r.N == N, vmp10), :D)
    if !isempty(subS)
        lines!(ax, subS.D, subS.wis_w; color=:darkorange, linewidth=2.6, linestyle=:dash,
               label="VMP-10")
        scatter!(ax, subS.D, subS.wis_w; color=:darkorange, marker=:rect, markersize=13,
                 strokecolor=:black, strokewidth=1)
    end
    axislegend(ax; position=:rt, labelsize=11, backgroundcolor=(:white, 0.9))
end

# Row 1: time
timepanel!(fig[1, 1], 50,  "N = 50   —  wall-clock")
timepanel!(fig[1, 2], 250, "N = 250  —  wall-clock")
# Row 2: WIS_w
wispanel!( fig[2, 1], 50,  "N = 50   —  w-only WIS")
wispanel!( fig[2, 2], 250, "N = 250  —  w-only WIS")

save(joinpath(RESULTS_DIR, "nuts_vs_vmp_full.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "nuts_vs_vmp_full.pdf"), fig)
println("Saved nuts_vs_vmp_full.{png,pdf}")
