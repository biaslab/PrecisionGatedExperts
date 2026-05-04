# Focused 2-panel figure: NUTS vs VMP-converged wall-clock at N=50 and N=250.
# Both methods run to a fixed budget — NUTS: 200 warm-up + 1000 samples
# (single chain); VMP-converged: per-N iter budget (N=50 → 50 iters,
# N=250 → 100 iters).  VMP-converged is the conservative VMP comparison:
# VMP-10 (the default we advocate in the paper) is uniformly faster than
# VMP-converged and at least as good on predictive NLL.
# The purpose of this figure is to show the gap is not sensitive to whether
# we give VMP "the best of itself" or "the most conservative self".

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

# ---- NUTS times (consolidated from all sweeps) -----------------------------
v2     = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_mcmc.csv"),     DataFrame)
focus  = CSV.read(joinpath(RESULTS_DIR, "ess_vs_D_focused.csv"),    DataFrame)
pareto = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv"), DataFrame)
stretch = CSV.read(joinpath(RESULTS_DIR, "nuts_wis_stretch.csv"),   DataFrame)
common = [:method, :N, :D, :total_time_s]
nuts = vcat(v2[:, common], focus[:, common], pareto[:, common], stretch[:, common])
filter!(r -> r.method == "NUTS", nuts)
unique!(nuts, [:N, :D]); sort!(nuts, [:N, :D])

# ---- VMP-converged times ---------------------------------------------------
vmpconv = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_vmp_converged.csv"), DataFrame)
filter!(r -> r.status == "OK", vmpconv)

nuts_50   = sort(filter(r -> r.N == 50,   nuts), :D)
nuts_250  = sort(filter(r -> r.N == 250,  nuts), :D)
vmpc_50   = sort(filter(r -> r.N == 50,   vmpconv), :D)
vmpc_250  = sort(filter(r -> r.N == 250,  vmpconv), :D)

# ---- figure layout ---------------------------------------------------------
fig = Figure(size = (1600, 800), fontsize = 15)
Label(fig[0, 1:2],
      "NUTS vs VMP-converged wall-clock time versus feature dimension D",
      fontsize=18, font=:bold)

function axis!(gp, title)
    ax = Axis(gp;
              title   = title,
              xlabel  = "D (feature dimension)  — log scale",
              ylabel  = "wall-clock per run (s)  — log scale",
              xscale  = log10, yscale = log10,
              xticks  = ([2, 5, 10, 25, 65, 120],
                         ["2", "5", "10", "25", "65", "120"]),
              yticks  = ([1, 10, 100, 1000, 10000, 100000],
                         ["1 s", "10 s", "100 s", "1 000 s", "10⁴ s", "10⁵ s"]))
    limits!(ax, (1.6, 220), (3, 2e5))
    return ax
end

# ---------- Panel 1: N = 50 ----------
ax1 = axis!(fig[1, 1], "N = 50")

# Reference slopes anchored at NUTS D=10 (so the reader can read the NUTS
# slope against D¹, D², D³).
D10_t = nuts_50.total_time_s[findfirst(==(10), nuts_50.D)]
Dref  = [2, 3, 5, 10, 25, 65, 90, 120]
lines!(ax1, Dref, D10_t .* (Dref ./ 10).^1; color=(:gray, 0.55), linestyle=:dot,     linewidth=1.5)
lines!(ax1, Dref, D10_t .* (Dref ./ 10).^2; color=(:gray, 0.55), linestyle=:dash,    linewidth=1.5)
lines!(ax1, Dref, D10_t .* (Dref ./ 10).^3; color=(:black, 0.7),  linestyle=:dashdot, linewidth=2.0)
for (exp_k, lbl, clr, fnt) in [(1, "D¹", :gray, :regular),
                               (2, "D²", :gray, :regular),
                               (3, "D³", :black, :bold)]
    text!(ax1, Dref[end], D10_t * (Dref[end]/10)^exp_k;
          text=" $lbl", align=(:left, :center),
          fontsize=13, color=clr, font=fnt)
end

# NUTS measured curve
lines!(ax1, nuts_50.D, nuts_50.total_time_s;
       color=:firebrick, linewidth=3.2, label="NUTS (200 + 1000 samples)")
scatter!(ax1, nuts_50.D, nuts_50.total_time_s;
         color=:firebrick, markersize=18, strokecolor=:black, strokewidth=1)

# Highlight the D=25 → 65 cubic segment
idx25 = findfirst(==(25), nuts_50.D); idx65 = findfirst(==(65), nuts_50.D)
lines!(ax1, nuts_50.D[idx25:idx65], nuts_50.total_time_s[idx25:idx65];
       color=:black, linewidth=5.0)

# VMP-converged measured curve
lines!(ax1, vmpc_50.D, vmpc_50.total_time_s;
       color=:seagreen, linewidth=3.2, label="VMP-converged (50 iters)")
scatter!(ax1, vmpc_50.D, vmpc_50.total_time_s;
         color=:seagreen, markersize=16, marker=:diamond,
         strokecolor=:black, strokewidth=1)

# Gap annotation at D=65 (visually the most dramatic cell)
t_nuts_65 = nuts_50.total_time_s[findfirst(==(65), nuts_50.D)]
t_vmp_65  = vmpc_50.total_time_s[findfirst(==(65), vmpc_50.D)]
ratio_65  = round(Int, t_nuts_65 / t_vmp_65)
text!(ax1, 65, sqrt(t_nuts_65 * t_vmp_65);
      text="$ratio_65×  gap",
      fontsize=14, color=:darkorange, font=:bold,
      align=(:left, :center), offset=(10, 0))

axislegend(ax1; position=:lt, labelsize=11, backgroundcolor=(:white, 0.9))

# ---------- Panel 2: N = 250 ----------
ax2 = axis!(fig[1, 2], "N = 250")

# NUTS measured
lines!(ax2, nuts_250.D, nuts_250.total_time_s;
       color=:firebrick, linewidth=3.2, label="NUTS (200 + 1000 samples)")
scatter!(ax2, nuts_250.D, nuts_250.total_time_s;
         color=:firebrick, markersize=18, strokecolor=:black, strokewidth=1)

# VMP-converged measured (only D ≤ 10 available)
lines!(ax2, vmpc_250.D, vmpc_250.total_time_s;
       color=:seagreen, linewidth=3.2, label="VMP-converged (100 iters, measured)")
scatter!(ax2, vmpc_250.D, vmpc_250.total_time_s;
         color=:seagreen, markersize=16, marker=:diamond,
         strokecolor=:black, strokewidth=1)

# VMP-converged extrapolation using t ≈ 0.079 · N^1.49 · D^0.08 (from fit)
Dext = 10.0 .^ range(log10(maximum(vmpc_250.D)), log10(120); length=40)
t_ext = 0.079 .* 250^1.49 .* Dext.^0.08
lines!(ax2, Dext, t_ext; color=(:seagreen, 0.8), linestyle=:dash, linewidth=2,
       label="VMP-converged (extrapolated: t ≈ N^{1.49}·D^{0.08})")

# Gap annotation at the D=100 NUTS point
if any(nuts_250.D .== 100)
    t_nuts_100 = nuts_250.total_time_s[findfirst(==(100), nuts_250.D)]
    t_vmp_100  = 0.079 * 250^1.49 * 100^0.08
    ratio_100  = round(Int, t_nuts_100 / t_vmp_100)
    text!(ax2, 100, sqrt(t_nuts_100 * t_vmp_100);
          text="$ratio_100×  gap",
          fontsize=14, color=:darkorange, font=:bold,
          align=(:left, :center), offset=(10, 0))
end

axislegend(ax2; position=:lt, labelsize=11, backgroundcolor=(:white, 0.9))

save(joinpath(RESULTS_DIR, "nuts_vs_vmp_converged.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "nuts_vs_vmp_converged.pdf"), fig)
println("Saved nuts_vs_vmp_converged.{png,pdf}")
