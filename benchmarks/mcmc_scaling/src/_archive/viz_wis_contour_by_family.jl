# 4-panel WIS contour (total, w-only, τ, β) × (Gibbs | VMP-dynamic).
# w-only is the intended paper metric; τ/β are diagnostic panels.
#
# Plus a companion single-panel figure for the paper: Gibbs & VMP w-only
# side-by-side.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics
using LinearAlgebra

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const CSV_PATH    = joinpath(RESULTS_DIR, "pareto_wis.csv")

df = CSV.read(CSV_PATH, DataFrame)
filter!(r -> r.status == "OK", df)

# Joint log-linear fit log10 y = a + α log10 N + β log10 D
function fit_surface(sub::DataFrame, col::Symbol)
    logN = log10.(Float64.(sub.N))
    logD = log10.(Float64.(sub.D))
    y    = log10.(max.(Float64.(getproperty(sub, col)), 1e-6))
    X    = hcat(ones(length(y)), logN, logD)
    coef = X \ y
    ŷ    = X * coef
    SS_res = sum((y .- ŷ).^2); SS_tot = sum((y .- mean(y)).^2)
    R2 = 1 - SS_res / max(SS_tot, 1e-12)
    return coef, R2
end

function make_grid(coef, (Nmin, Nmax, Dmin, Dmax)=(10, 250, 2, 65))
    a, α, β = coef
    N_grid = 10.0 .^ range(log10(Nmin) - 0.1, log10(Nmax) + 0.1; length=130)
    D_grid = 10.0 .^ range(log10(Dmin) - 0.1, log10(Dmax) + 0.1; length=130)
    M = [10.0 ^ (a + α*log10(N) + β*log10(D)) for N in N_grid, D in D_grid]
    return N_grid, D_grid, M
end

Nticks = [10, 25, 50, 100, 250]
Dticks = [2, 5, 10, 25, 65]

function draw_panel!(ax, sub::DataFrame, col::Symbol, coef, R2, clim, iso)
    N_grid, D_grid, M = make_grid(coef)
    hm = heatmap!(ax, N_grid, D_grid, log10.(max.(M, 1e-6));
                  colormap = :viridis, alpha = 0.9, colorrange = clim)
    contour!(ax, N_grid, D_grid, M; levels = iso,
             color = :white, linewidth = 1.4)
    a, α, β = coef
    for lvl in iso
        D_anchor = 10.0
        N_lab = 10.0 ^ ((log10(lvl) - a - β*log10(D_anchor)) / α)
        if N_grid[1] ≤ N_lab ≤ N_grid[end]
            text!(ax, N_lab, D_anchor; text=string(lvl),
                  fontsize=10, color=:white, font=:bold,
                  align=(:center, :center))
        end
    end
    scatter!(ax, sub.N, sub.D;
             color=:white, strokecolor=:black, strokewidth=1.5, markersize=14)
    for r in eachrow(sub)
        v = getproperty(r, col)
        txt = v < 0.01 ? "$(round(v, digits=3))" :
              v < 0.1  ? "$(round(v, digits=2))" :
                         "$(round(v, digits=2))"
        text!(ax, r.N, r.D; text="  $txt",
              fontsize=10, color=:black, font=:bold,
              align=(:left, :center), offset=(6, 0))
    end
    return hm
end

# ============================================================================
# Figure 1 — 4-panel grid: (total, w, τ, β) × (Gibbs, VMP-dynamic)
# ============================================================================
METHODS    = ["Gibbs", "VMP-dynamic"]
METRICS    = [(:wis, "total WIS"), (:wis_w, "w-only WIS"),
              (:wis_τ, "τ-only WIS"), (:wis_β, "β-only WIS")]
ISO_LEVELS = Dict(
    :wis   => [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.7],
    :wis_w => [0.05, 0.1, 0.2, 0.4, 0.8, 1.3, 1.8],
    :wis_τ => [0.4, 0.8, 1.2, 1.8, 2.5, 3.0],
    :wis_β => [0.001, 0.005, 0.02, 0.05, 0.1, 0.15],
)

fig1 = Figure(size = (2200, 1500), fontsize = 14)
Label(fig1[0, 1:length(METRICS)],
      "WIS broken down by parameter family  —  rows: method, cols: metric\n" *
      "w block is the intended forecasting metric;  τ and β blocks reveal prior–truth mismatch " *
      "(Gamma(1,1) τ prior has mean 1, truth ∼ U(0.5, 5.0))",
      fontsize=16, font=:bold)

# Compute a shared colour range per column so rows are comparable
clims = Dict{Symbol, Tuple{Float64, Float64}}()
for (col, _) in METRICS
    vals = log10.(max.(Float64.(df[!, col]), 1e-6))
    clims[col] = (minimum(vals), maximum(vals))
end

for (row, m) in enumerate(METHODS)
    sub_m = sort(filter(r -> r.method == m, df), [:N, :D])
    for (col_i, (metric, label)) in enumerate(METRICS)
        coef, R2 = fit_surface(sub_m, metric)
        a, α, β = coef
        ax = Axis(fig1[row, col_i];
                  title = "$m — $label\nfit ≈ $(round(10^a, digits=3))·N^$(round(α, digits=2))·D^$(round(β, digits=2))  (R²=$(round(R2, digits=2)))",
                  titlesize = 13,
                  xlabel = row == length(METHODS) ? "N (log)" : "",
                  ylabel = col_i == 1 ? "D (log)" : "",
                  xscale = log10, yscale = log10,
                  xticks = (Nticks, string.(Nticks)),
                  yticks = (Dticks, string.(Dticks)))
        draw_panel!(ax, sub_m, metric, coef, R2, clims[metric], ISO_LEVELS[metric])
    end
end

save(joinpath(RESULTS_DIR, "wis_contour_by_family.png"), fig1; px_per_unit=2)
save(joinpath(RESULTS_DIR, "wis_contour_by_family.pdf"), fig1)
println("Saved wis_contour_by_family.{png,pdf}")

# ============================================================================
# Figure 2 — clean 2-panel: w-only WIS, Gibbs | VMP-dynamic (PAPER figure)
# ============================================================================
fig2 = Figure(size = (1800, 750), fontsize = 15)
Label(fig2[0, 1:2],
      "Weighted Interval Score on forecasting weights w  over (N, D)  —  lower = better.\n" *
      "Proper scoring rule (≈ CRPS from 50/80/90/95 % intervals).  τ and β excluded — their behaviour is driven by prior–data mismatch, see diagnostic panel.",
      fontsize=15, font=:bold)

# Shared colour range for the two w-only panels
w_sub = filter(r -> r.method in METHODS, df)
w_clim = (minimum(log10.(max.(w_sub.wis_w, 1e-6))), maximum(log10.(max.(w_sub.wis_w, 1e-6))))
w_iso  = ISO_LEVELS[:wis_w]

local hm_last
for (col_i, m) in enumerate(METHODS)
    sub_m = sort(filter(r -> r.method == m, df), [:N, :D])
    coef, R2 = fit_surface(sub_m, :wis_w)
    a, α, β = coef
    ax = Axis(fig2[1, col_i];
              title = "$m   —   fit  WIS_w ≈ $(round(10^a, digits=3))·N^$(round(α, digits=2))·D^$(round(β, digits=2))   (R²=$(round(R2, digits=2)))",
              titlesize = 14,
              xlabel = "N (observations)  — log scale",
              ylabel = col_i == 1 ? "D (feature dim)  — log scale" : "",
              xscale = log10, yscale = log10,
              xticks = (Nticks, string.(Nticks)),
              yticks = (Dticks, string.(Dticks)))
    hm = draw_panel!(ax, sub_m, :wis_w, coef, R2, w_clim, w_iso)
    if col_i == length(METHODS)
        Colorbar(fig2[1, 3], hm; label = "log₁₀(w-only WIS)", width = 18)
    end
end

save(joinpath(RESULTS_DIR, "wis_w_contour.png"), fig2; px_per_unit=2)
save(joinpath(RESULTS_DIR, "wis_w_contour.pdf"), fig2)
println("Saved wis_w_contour.{png,pdf}")

# ============================================================================
# Print summary tables
# ============================================================================
println("\nper-family WIS table (Gibbs & VMP-dynamic):")
show(stdout, sort(df[:, [:method, :N, :D, :wis, :wis_w, :wis_τ, :wis_β]], [:method, :D, :N]);
     allrows=true, allcols=true)
println()
