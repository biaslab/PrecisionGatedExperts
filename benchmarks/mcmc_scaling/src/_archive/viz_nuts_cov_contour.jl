# NUTS 95% CI coverage contour over (N, D).
# Data from pareto_quality_mcmc.csv (NUTS rows, 13 cells).
# Because coverage is bounded in (0, 1), fit is done in logit-space:
#     logit(cov) = a + α log10 N + β log10 D
# The surface is converted back via σ(·). Target-coverage contour (0.95) is drawn in gold.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics
using LinearAlgebra

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

df = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv"), DataFrame)
nuts = filter(r -> r.method == "NUTS" &&
                  !ismissing(r.cov95) && !isnan(r.cov95), df)
sort!(nuts, [:N, :D])

println("NUTS cov95 data points:")
show(stdout, nuts[:, [:N, :D, :cov95]]; allrows=true); println()

# Logit-space joint fit — robust near 0/1 boundaries
clip(x) = clamp(x, 1e-3, 1 - 1e-3)
logit(p) = log(p / (1 - p))
sig(z)   = 1 / (1 + exp(-z))

logN  = log10.(Float64.(nuts.N))
logD  = log10.(Float64.(nuts.D))
lcov  = logit.(clip.(Float64.(nuts.cov95)))
X     = hcat(ones(length(logN)), logN, logD)
coef  = X \ lcov
a, α, β = coef
R2    = 1 - sum((lcov .- X*coef).^2) / sum((lcov .- mean(lcov)).^2)
println("\nLogit-fit: logit(cov95) = $(round(a, digits=2)) + $(round(α, digits=2))·log10(N) + $(round(β, digits=2))·log10(D)   (R² on logit = $(round(R2, digits=2)))")

Nmin, Nmax = 10, 250
Dmin, Dmax = 2, 65
N_grid = 10.0 .^ range(log10(Nmin) - 0.1, log10(Nmax) + 0.1; length=120)
D_grid = 10.0 .^ range(log10(Dmin) - 0.1, log10(Dmax) + 0.1; length=120)
C = [sig(a + α*log10(N) + β*log10(D)) for N in N_grid, D in D_grid]

fig = Figure(size=(1250, 800), fontsize=15)
Label(fig[0, 1:2],
      "NUTS 95% CI coverage over (N, D)  —  gold line is target coverage 0.95\n" *
      "(fit: logit(cov)=$(round(a, digits=2)) + $(round(α, digits=2))·log N + $(round(β, digits=2))·log D,  R²=$(round(R2, digits=2)))",
      fontsize=15, font=:bold)

Nticks = [10, 25, 50, 100, 250]
Dticks = [2, 5, 10, 25, 65]
ax = Axis(fig[1, 1];
          xlabel = "N (observations)  — log scale",
          ylabel = "D (feature dim)   — log scale",
          xscale = log10, yscale = log10,
          xticks = (Nticks, string.(Nticks)),
          yticks = (Dticks, string.(Dticks)))

# Diverging colormap centred at 0.95 — under-coverage (red) vs over-coverage (blue)
hm = heatmap!(ax, N_grid, D_grid, C;
              colormap = :RdYlBu, colorrange = (0.0, 1.0), alpha = 0.9)

# General level lines
levels = [0.2, 0.4, 0.6, 0.8, 0.9]
contour!(ax, N_grid, D_grid, C; levels = levels, color = :white, linewidth = 1.6)
for lvl in levels
    # Solve for N given D=10: log N = (logit(lvl) - a - β*log D) / α
    N_label = 10.0 ^ ((logit(lvl) - a - β*log10(10.0)) / α)
    if N_grid[1] ≤ N_label ≤ N_grid[end]
        text!(ax, N_label, 10.0; text = string(lvl),
              fontsize=11, color=:white, font=:bold,
              align=(:center, :center))
    end
end

# Highlighted target line at 0.95
contour!(ax, N_grid, D_grid, C; levels = [0.95],
         color = :gold, linewidth = 3.5)
N_lab95 = 10.0 ^ ((logit(0.95) - a - β*log10(10.0)) / α)
if N_grid[1] ≤ N_lab95 ≤ N_grid[end]
    text!(ax, N_lab95, 10.0; text="cov=0.95",
          fontsize=12, color=:black, font=:bold,
          align=(:center, :bottom), offset=(0, 6))
end

# Data points with numeric labels and rim color showing under/over coverage
for r in eachrow(nuts)
    rim = r.cov95 < 0.95 ? :red : :black
    scatter!(ax, [r.N], [r.D];
             color=:white, strokecolor=rim, strokewidth=2.0,
             markersize=16)
    text!(ax, r.N, r.D;
          text = "  $(round(r.cov95, digits=2))",
          fontsize=11, color=:black, font=:bold,
          align=(:left, :center), offset=(8, 0))
end

Colorbar(fig[1, 2], hm; label = "coverage (fraction of 95% CIs containing truth)", width = 18)

save(joinpath(RESULTS_DIR, "nuts_cov_contour.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "nuts_cov_contour.pdf"), fig)
println("\nSaved nuts_cov_contour.{png,pdf}")
