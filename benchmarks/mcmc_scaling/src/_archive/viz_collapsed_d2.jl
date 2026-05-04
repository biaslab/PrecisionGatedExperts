using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using LinearAlgebra
using StableRNGs
using Statistics
using Turing
using MCMCChains
using ADTypes
using KernelDensity

import Enzyme

include("data_generation.jl")
include("gibbs_handcoded.jl")
include("turing_model_collapsed.jl")

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

# ==================================================================
# Setup
# ==================================================================
const N = 10
const N_EXPERTS = 7
const D = 2                      # ← key change: 2-dim features
const K_GH = 15

const GIBBS_WARMUP = 1000
const GIBBS_SAMPLES = 50000

const NUTS_WARMUP = 1000
const NUTS_SAMPLES = 2000
const NUTS_DELTA = 0.95

println("=" ^ 60)
println("d=2 marginal-posterior visualization")
println("N=$N, n_experts=$N_EXPERTS, d=$D")
println("=" ^ 60)

data = generate_synthetic_data(N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(42))

# ==================================================================
# Gibbs reference (long run on un-collapsed model, keep only globals)
# ==================================================================
println("\n[Gibbs] $(GIBBS_WARMUP) warmup + $(GIBBS_SAMPLES) samples...")
gibbs_result = run_gibbs(
    data.y, data.features, data.predictions;
    n_experts=N_EXPERTS, d=D,
    n_warmup=GIBBS_WARMUP, n_samples=GIBBS_SAMPLES,
    rng=StableRNG(1),
)
println("  Gibbs done in $(round(gibbs_result.elapsed_seconds, digits=1))s " *
        "($(round(gibbs_result.time_per_sweep * 1000, digits=2))ms/sweep)")

# Extract global chains: gibbs_w[sweep][expert, dim], etc.
# Reshape into trace-per-param arrays
gibbs_w = zeros(GIBBS_SAMPLES, N_EXPERTS, D)       # [sweep, expert, dim]
gibbs_τ = zeros(GIBBS_SAMPLES, N_EXPERTS)
gibbs_β = zeros(GIBBS_SAMPLES, N_EXPERTS)
for s in 1:GIBBS_SAMPLES
    gibbs_w[s, :, :] = gibbs_result.w_samples[s]     # n_experts × d
    gibbs_τ[s, :]    = gibbs_result.τ_samples[s]
    gibbs_β[s, :]    = gibbs_result.β_samples[s]
end

# ESS diagnostics for Gibbs
println("  Gibbs ESS (sample of params):")
for i in 1:3
    ess_w = ess_simple(gibbs_w[:, i, 1])
    ess_τ = ess_simple(gibbs_τ[:, i])
    println("    expert $i: ESS(w[1,$i])=$(round(Int,ess_w))  ESS(τ[$i])=$(round(Int,ess_τ))")
end

# ==================================================================
# NUTS on collapsed model
# ==================================================================
println("\n[NUTS] Enzyme.Reverse, $(NUTS_WARMUP) warmup + $(NUTS_SAMPLES) samples...")
gh_nodes, gh_weights = gausshermite_gw(K_GH)
model = pge_ensemble_collapsed(
    data.y, data.features, data.predictions,
    N_EXPERTS, D, gh_nodes, gh_weights,
)
adtype = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

print("  Pre-compiling...")
t_pre = @elapsed sample(model, NUTS(3, NUTS_DELTA; adtype=adtype), 3;
                         discard_initial=3, progress=false)
println(" $(round(t_pre, digits=1))s")

print("  Sampling...")
t0 = time()
chain = sample(
    model, NUTS(NUTS_WARMUP, NUTS_DELTA; adtype=adtype), NUTS_SAMPLES;
    discard_initial=NUTS_WARMUP, progress=false,
)
t_nuts = time() - t0
println(" $(round(t_nuts, digits=1))s  ($(round(t_nuts*1000/(NUTS_WARMUP+NUTS_SAMPLES), digits=1))ms/iter)")

# Extract NUTS global chains
# filldist(MvNormal(...), n_experts) gives w of shape (d, n_experts)
# Chain names: "w[k,i]" = dim k, expert i
nuts_w = zeros(NUTS_SAMPLES, N_EXPERTS, D)
nuts_τ = zeros(NUTS_SAMPLES, N_EXPERTS)
nuts_β = zeros(NUTS_SAMPLES, N_EXPERTS)
for i in 1:N_EXPERTS
    for k in 1:D
        nuts_w[:, i, k] = vec(Array(chain[:, Symbol("w[$k, $i]"), :]))
    end
    nuts_τ[:, i] = vec(Array(chain[:, Symbol("τ[$i]"), :]))
    nuts_β[:, i] = vec(Array(chain[:, Symbol("β[$i]"), :]))
end

# ESS for NUTS
ess_tbl = ess_rhat(chain)
min_ess_nuts = minimum(skipmissing(ess_tbl[:, :ess]))
max_rhat_nuts = maximum(skipmissing(ess_tbl[:, :rhat]))
println("  NUTS min ESS=$(round(min_ess_nuts, digits=1))/$NUTS_SAMPLES  max R-hat=$(round(max_rhat_nuts, digits=3))")

# ==================================================================
# Figure 1: 4×7 marginal ridge
# ==================================================================
println("\nRendering marginal ridge figure...")
fig = Figure(size=(1800, 1000), fontsize=11)

Label(fig[0, 1:N_EXPERTS],
      "Marginal posteriors at d=2: Gibbs reference (blue) vs collapsed NUTS (green)\n" *
      "N=$N, $N_EXPERTS experts — NUTS min ESS=$(round(Int, min_ess_nuts)), max R-hat=$(round(max_rhat_nuts, digits=2))",
      fontsize=14, font=:bold)

param_names = ["w[1,i]  (intercept)", "w[2,i]  (slope)", "τ[i]  (precision)", "β[i]  (γ-rate)"]

function plot_marginal!(ax, gibbs_samples, nuts_samples, truth; xlabel="")
    # KDE / histogram for Gibbs reference
    gibbs_k = kde(gibbs_samples)
    band!(ax, gibbs_k.x, zeros(length(gibbs_k.x)), gibbs_k.density;
          color=(:steelblue, 0.35))
    lines!(ax, gibbs_k.x, gibbs_k.density; color=:steelblue, linewidth=1.2)

    # Histogram for NUTS, rescaled to match density axis
    hist!(ax, nuts_samples; bins=30, normalization=:pdf,
          color=(:seagreen, 0.45), strokewidth=0.5, strokecolor=:seagreen)

    # Ground truth
    vlines!(ax, [truth]; color=:red, linestyle=:dash, linewidth=1.5)

    ax.xlabel = xlabel
    hideydecorations!(ax; grid=false)
end

for i in 1:N_EXPERTS
    for (row, (params_g, params_n, truth_val, label)) in enumerate([
        (view(gibbs_w, :, i, 1), view(nuts_w, :, i, 1), data.w_true[i][1], "w[1,$i]"),
        (view(gibbs_w, :, i, 2), view(nuts_w, :, i, 2), data.w_true[i][2], "w[2,$i]"),
        (view(gibbs_τ, :, i),    view(nuts_τ, :, i),    data.τ_true[i],     "τ[$i]"),
        (view(gibbs_β, :, i),    view(nuts_β, :, i),    data.β_true[i],     "β[$i]"),
    ])
        ax = Axis(fig[row, i]; xlabel=label, xlabelsize=10)
        plot_marginal!(ax, collect(params_g), collect(params_n), truth_val; xlabel=label)

        # ESS annotation per panel
        ess_n = ess_simple(collect(params_n))
        text!(ax, 0.02, 0.95; text="ESS=$(round(Int, ess_n))",
              space=:relative, fontsize=8, color=:seagreen,
              align=(:left, :top))

        if i == 1
            Label(fig[row, 0], param_names[row];
                  rotation=π/2, fontsize=11, font=:bold)
        end
        if row == 1
            Label(fig[row - 1, i, Top()], "Expert $i";
                  fontsize=10, font=:bold, padding=(0, 0, 2, 2))
        end
    end
end

# Legend row at bottom
leg_ax = Axis(fig[5, 1:N_EXPERTS])
hidedecorations!(leg_ax)
hidespines!(leg_ax)
leg_elems = [
    PolyElement(color=(:steelblue, 0.35), strokecolor=:steelblue),
    PolyElement(color=(:seagreen, 0.45), strokecolor=:seagreen),
    LineElement(color=:red, linestyle=:dash, linewidth=1.5),
]
leg_labels = [
    "Gibbs reference ($(GIBBS_SAMPLES) samples, un-collapsed)",
    "Collapsed NUTS / Enzyme ($(NUTS_SAMPLES) samples)",
    "Ground truth",
]
Legend(fig[5, 1:N_EXPERTS], leg_elems, leg_labels;
       orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false)

rowsize!(fig.layout, 5, Relative(0.04))

save(joinpath(RESULTS_DIR, "marginals_d2.png"), fig; px_per_unit=3)
save(joinpath(RESULTS_DIR, "marginals_d2.pdf"), fig)
println("  Saved marginals_d2.{png,pdf}")

# ==================================================================
# Figure 2: Expert-1 corner plot (4×4)
# ==================================================================
println("\nRendering expert-1 corner plot...")

params_names = ["w[1,1]", "w[2,1]", "τ[1]", "β[1]"]
gibbs_cols = [gibbs_w[:, 1, 1], gibbs_w[:, 1, 2], gibbs_τ[:, 1], gibbs_β[:, 1]]
nuts_cols  = [nuts_w[:, 1, 1],  nuts_w[:, 1, 2],  nuts_τ[:, 1],  nuts_β[:, 1]]
truth_vals = [data.w_true[1][1], data.w_true[1][2], data.τ_true[1], data.β_true[1]]

corner = Figure(size=(900, 900), fontsize=11)
Label(corner[0, 1:4],
      "Expert 1 joint posterior — Gibbs reference (blue contours) vs collapsed NUTS (green scatter)",
      fontsize=13, font=:bold)

for ri in 1:4, ci in 1:4
    if ci > ri
        continue  # upper triangle: leave empty
    end

    ax = Axis(corner[ri, ci])

    if ri == ci
        # Diagonal: marginal densities
        gk = kde(gibbs_cols[ri])
        band!(ax, gk.x, zeros(length(gk.x)), gk.density; color=(:steelblue, 0.35))
        lines!(ax, gk.x, gk.density; color=:steelblue, linewidth=1.2)
        hist!(ax, nuts_cols[ri]; bins=30, normalization=:pdf,
              color=(:seagreen, 0.45), strokewidth=0.5, strokecolor=:seagreen)
        vlines!(ax, [truth_vals[ri]]; color=:red, linestyle=:dash, linewidth=1.5)
        hideydecorations!(ax; grid=false)
    else
        # Lower triangle: 2D
        # Gibbs as hex density via hexbin approximation (scatter with transparency)
        scatter!(ax, gibbs_cols[ci], gibbs_cols[ri];
                 color=(:steelblue, 0.06), markersize=3)
        scatter!(ax, nuts_cols[ci], nuts_cols[ri];
                 color=(:seagreen, 0.15), markersize=2)
        scatter!(ax, [truth_vals[ci]], [truth_vals[ri]];
                 color=:red, markersize=14, marker=:star5,
                 strokecolor=:black, strokewidth=1.2)
    end

    # Labels
    if ri == 4
        ax.xlabel = params_names[ci]
    else
        hidexdecorations!(ax; grid=false)
    end
    if ci == 1 && ri > 1
        ax.ylabel = params_names[ri]
    elseif ri != ci
        hideydecorations!(ax; grid=false)
    end
end

save(joinpath(RESULTS_DIR, "corner_expert1_d2.png"), corner; px_per_unit=3)
save(joinpath(RESULTS_DIR, "corner_expert1_d2.pdf"), corner)
println("  Saved corner_expert1_d2.{png,pdf}")

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
println("  Gibbs: $(GIBBS_SAMPLES) samples, ESS(w[1,1])=$(round(Int, ess_simple(gibbs_w[:, 1, 1])))")
println("  NUTS : $(NUTS_SAMPLES) samples, ESS(w[1,1])=$(round(Int, ess_simple(nuts_w[:, 1, 1])))")
println("         per-iter=$(round(t_nuts*1000/(NUTS_WARMUP+NUTS_SAMPLES), digits=1))ms")
println("  Figures in $RESULTS_DIR")
