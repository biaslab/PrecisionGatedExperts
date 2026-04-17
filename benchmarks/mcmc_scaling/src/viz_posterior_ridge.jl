using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using LinearAlgebra
using StableRNGs
using Statistics
using CSV
using DataFrames

include("data_generation.jl")
include("gibbs_handcoded.jl")

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

function posterior_slice_2d(w_grid, z_grid, data; expert=1, dim_k=1, obs_j=1)
    i = expert
    logp = zeros(length(w_grid), length(z_grid))
    for (wi, wval) in enumerate(w_grid)
        for (zi, zval) in enumerate(z_grid)
            ld = -0.01 / 2 * wval^2
            for j in 1:data.N
                wTphi = wval * data.features[j][dim_k]
                for k in 1:data.d
                    k == dim_k && continue
                    wTphi += data.w_true[i][k] * data.features[j][k]
                end
                z_ij = (j == obs_j) ? zval : data.z_true[i, j]
                ld += -data.τ_true[i] / 2 * (z_ij - wTphi)^2
            end
            ld += zval - data.β_true[i] * exp(zval)
            γ_val = exp(zval)
            ld += 0.5 * log(γ_val) - 0.5 * γ_val * (data.y[obs_j] - data.predictions[i, obs_j])^2
            logp[wi, zi] = ld
        end
    end
    return logp
end

function gibbs_traces(data; expert=1, dim_k=1, obs_j=1, n_warmup=200, n_samples=500)
    result = run_gibbs(
        data.y, data.features, data.predictions;
        n_experts=data.n_experts, d=data.d,
        n_warmup=n_warmup, n_samples=n_samples,
    )
    w_trace = [result.w_samples[s][expert, dim_k] for s in 1:n_samples]
    z_trace = [result.z_samples[s][expert, obs_j] for s in 1:n_samples]
    return (; w_trace, z_trace)
end

# ============================================================
# Load NUTS samples from CSV (pre-computed separately)
# ============================================================
nuts_csv = joinpath(RESULTS_DIR, "nuts_n10_samples.csv")
nuts_n10 = isfile(nuts_csv) ? CSV.read(nuts_csv, DataFrame) : nothing
if nuts_n10 !== nothing
    println("Loaded $(nrow(nuts_n10)) NUTS samples for N=10")
else
    println("WARNING: no NUTS samples found at $nuts_csv — run NUTS separately first")
end

# ============================================================
# Generate data and run Gibbs
# ============================================================
N_values = [10, 100, 1000]
datasets = Dict{Int,Any}()
gibbs_results = Dict{Int,Any}()

for N in N_values
    data = generate_synthetic_data(N=N, n_experts=7, d=65, rng=StableRNG(42))
    datasets[N] = data
    println("N=$N: running Gibbs (200+500)...")
    gibbs_results[N] = gibbs_traces(data; expert=1, dim_k=1, obs_j=1, n_warmup=200, n_samples=500)
end

# ============================================================
# Figure
# ============================================================
data10 = datasets[10]
w_true_k = data10.w_true[1][1]
z_true_ij = data10.z_true[1, 1]

w_range = range(w_true_k - 0.15, w_true_k + 0.15, length = 200)
z_range = range(z_true_ij - 0.5, z_true_ij + 0.5, length = 200)

fig = Figure(size = (1600, 900), fontsize = 13)

for (col, N) in enumerate(N_values)
    data = datasets[N]
    println("Computing 2D posterior for N=$N...")

    logp = posterior_slice_2d(w_range, z_range, data; expert=1, dim_k=1, obs_j=1)
    logp .-= maximum(logp)

    ax_2d = Axis(fig[1, col]; xlabel="w₁₁", ylabel="z₁₁", title="N = $N")

    # Oracle posterior (blue)
    contourf!(ax_2d, collect(w_range), collect(z_range), logp';
        levels=range(-8, 0, length=20), colormap=:blues)
    contour!(ax_2d, collect(w_range), collect(z_range), logp';
        levels=[-0.5, -2, -5], color=:steelblue, linewidth=0.8)

    # Gibbs samples (red)
    gr = gibbs_results[N]
    scatter!(ax_2d, gr.w_trace, gr.z_trace; color=(:red, 0.3), markersize=2)

    # NUTS samples (green) — only for N=10
    if N == 10 && nuts_n10 !== nothing
        scatter!(ax_2d, nuts_n10.w, nuts_n10.z;
            color=(:green, 0.7), markersize=6, marker=:diamond)
    elseif N > 10
        text!(ax_2d, 0.98, 0.02; text="NUTS: infeasible ($(round(Int, N * 7 + 469)) dims)",
            space=:relative, fontsize=10, color=:darkgreen,
            align=(:right, :bottom), font=:bold)
    end

    # Ground truth
    scatter!(ax_2d, [data.w_true[1][1]], [data.z_true[1, 1]];
        color=:white, markersize=12, marker=:star5,
        strokecolor=:black, strokewidth=2)

    # Legend on first panel
    if col == 1
        elems = [
            PolyElement(color=:steelblue, strokecolor=:transparent),
            MarkerElement(marker=:circle, color=(:red, 0.5), markersize=6),
            MarkerElement(marker=:diamond, color=(:green, 0.7), markersize=8),
            MarkerElement(marker=:star5, color=:white, markersize=10,
                strokecolor=:black, strokewidth=1.5),
        ]
        labels = ["Oracle posterior\n(rest = truth)", "Gibbs (700 sweeps)",
                  "NUTS (30 samples)", "Ground truth"]
        Legend(fig[1, col], elems, labels;
            tellwidth=false, tellheight=false,
            halign=:right, valign=:top,
            margin=(8, 8, 8, 8), framevisible=true,
            padding=(6, 6, 4, 4), labelsize=9)
    end

    # --- Bottom row: traces ---
    ax_trace = Axis(fig[2, col]; xlabel="iteration", ylabel="w₁₁",
        title="N = $N  —  trace of w₁₁")

    lines!(ax_trace, 1:length(gr.w_trace), gr.w_trace;
        color=(:red, 0.5), linewidth=0.5)

    if N == 10 && nuts_n10 !== nothing
        lines!(ax_trace, 1:nrow(nuts_n10), nuts_n10.w;
            color=(:green, 0.8), linewidth=1.5)
    end

    hlines!(ax_trace, [data.w_true[1][1]];
        color=:black, linestyle=:dash, linewidth=1)

    ess_gibbs = ess_simple(gr.w_trace)
    lbl = "Gibbs ESS=$(round(Int, ess_gibbs))"
    if N == 10 && nuts_n10 !== nothing
        ess_nuts = ess_simple(nuts_n10.w)
        lbl *= " | NUTS ESS=$(round(Int, ess_nuts))"
    elseif N > 10
        lbl *= " | NUTS: infeasible"
    end

    text!(ax_trace, 0.02, 0.95; text=lbl,
        space=:relative, fontsize=11, color=:black,
        align=(:left, :top))
end

Label(fig[0, :],
    "Oracle posterior (blue) vs MCMC samples: Gibbs (red, 700 sweeps) and NUTS (green, 30 samples)",
    fontsize=15, font=:bold)

save(joinpath(RESULTS_DIR, "posterior_ridge.png"), fig; px_per_unit=3)
save(joinpath(RESULTS_DIR, "posterior_ridge.pdf"), fig)
println("\nSaved to $(RESULTS_DIR)/posterior_ridge.{png,pdf}")
