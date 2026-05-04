# NUTS rel-L2 posterior error contour over (N, D).
# Lower = posterior mean closer to ground truth.
# rel_err_total = ||E[θ|y] − θ*|| / ||θ*||  (dimensionless)
# Perfect recovery → 0.  Prior-only → ~1.  Over-fit mean → > 1.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics
using LinearAlgebra

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

df = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv"), DataFrame)
nuts = filter(r -> r.method == "NUTS" &&
                   !ismissing(r.rel_err_total) && !isnan(r.rel_err_total) &&
                   r.rel_err_total > 0, df)
sort!(nuts, [:N, :D])

println("NUTS rel_err_total data points:")
show(stdout, nuts[:, [:N, :D, :rel_err_total]]; allrows=true); println()

logN  = log10.(Float64.(nuts.N))
logD  = log10.(Float64.(nuts.D))
le    = log10.(Float64.(nuts.rel_err_total))
X     = hcat(ones(length(logN)), logN, logD)
coef  = X \ le
a, α, β = coef
R2    = 1 - sum((le .- X*coef).^2) / sum((le .- mean(le)).^2)
println("\nFit:  rel_err ≈ $(round(10^a, digits=3)) · N^$(round(α, digits=3)) · D^$(round(β, digits=3))   (R² on log = $(round(R2, digits=2)))")

Nmin, Nmax = 10, 250
Dmin, Dmax = 2, 65
N_grid = 10.0 .^ range(log10(Nmin) - 0.1, log10(Nmax) + 0.1; length=120)
D_grid = 10.0 .^ range(log10(Dmin) - 0.1, log10(Dmax) + 0.1; length=120)
E = [10.0 ^ (a + α*log10(N) + β*log10(D)) for N in N_grid, D in D_grid]

fig = Figure(size=(1250, 800), fontsize=15)
Label(fig[0, 1:2],
      "NUTS rel-L2 posterior error  ||E[θ|y]−θ*|| / ||θ*||  over (N, D)\n" *
      "fit: rel_err ≈ $(round(10^a, digits=2))·N^$(round(α, digits=2))·D^$(round(β, digits=2))   (R²=$(round(R2, digits=2)))" *
      "   — gold dashed = 1.0 (prior-only baseline)",
      fontsize=15, font=:bold)

Nticks = [10, 25, 50, 100, 250]
Dticks = [2, 5, 10, 25, 65]
ax = Axis(fig[1, 1];
          xlabel = "N (observations)  — log scale",
          ylabel = "D (feature dim)   — log scale",
          xscale = log10, yscale = log10,
          xticks = (Nticks, string.(Nticks)),
          yticks = (Dticks, string.(Dticks)))

# Viridis — low error = dark (good), high error = light (bad) would be confusing,
# so use :Reds (low=light, high=dark-red) which matches "more red = more wrong"
hm = heatmap!(ax, N_grid, D_grid, E;
              colormap = :Reds, alpha = 0.9)

iso_levels = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
contour!(ax, N_grid, D_grid, E;
         levels = iso_levels,
         color = :black, linewidth = 1.2)
for lvl in iso_levels
    D_anchor = 10.0
    N_label = 10.0 ^ ((log10(lvl) - a - β*log10(D_anchor)) / α)
    if N_grid[1] ≤ N_label ≤ N_grid[end]
        text!(ax, N_label, D_anchor; text = string(lvl),
              fontsize=11, color=:black, font=:bold,
              align=(:center, :center))
    end
end

# Prior-baseline contour highlighted
contour!(ax, N_grid, D_grid, E; levels = [1.0],
         color = :gold, linewidth = 3.0, linestyle = :dash)

# Data points with numeric labels; red rim if rel_err > 1 (worse than prior)
for r in eachrow(nuts)
    rim = r.rel_err_total > 1.0 ? :red : :forestgreen
    scatter!(ax, [r.N], [r.D];
             color=:white, strokecolor=rim, strokewidth=2.0,
             markersize=16)
    text!(ax, r.N, r.D;
          text = "  $(round(r.rel_err_total, digits=2))",
          fontsize=11, color=:black, font=:bold,
          align=(:left, :center), offset=(8, 0))
end

Colorbar(fig[1, 2], hm; label = "rel-L2 posterior error", width = 18)

save(joinpath(RESULTS_DIR, "nuts_rel_err_contour.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "nuts_rel_err_contour.pdf"), fig)
println("\nSaved nuts_rel_err_contour.{png,pdf}")
