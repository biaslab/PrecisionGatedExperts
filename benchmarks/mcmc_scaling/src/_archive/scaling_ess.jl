using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using LinearAlgebra
using StableRNGs
using Statistics
using Turing
using MCMCChains
using ADTypes
using DataFrames
using CSV

import Enzyme

include("data_generation.jl")
include("gibbs_handcoded.jl")
include("turing_model_collapsed.jl")

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

# ==================================================================
# Experiment configuration
# ==================================================================
const N_EXPERTS = 7
const K_GH = 15

# Gibbs run length — short enough to be fast, long enough for ESS signal
const GIBBS_WARMUP = 500
const GIBBS_SAMPLES = 2000

# NUTS: short but enough for ESS signal
const NUTS_WARMUP = 500
const NUTS_SAMPLES = 1000
const NUTS_DELTA = 0.95

# Sweeps
const N_SWEEP = [10, 25, 50, 100, 250, 500, 1000]   # with D=2
const D_SWEEP = [2, 5, 10, 25, 65]                   # with N=10

# Pre-compute GH nodes
const GH_NODES, GH_WEIGHTS = gausshermite_gw(K_GH)

# ==================================================================
# Run helpers — return (min_ess, median_ess, t_per_iter_ms, max_rhat)
# ==================================================================
function run_gibbs_config(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed))
    result = run_gibbs(
        data.y, data.features, data.predictions;
        n_experts=N_EXPERTS, d=D,
        n_warmup=GIBBS_WARMUP, n_samples=GIBBS_SAMPLES,
        rng=StableRNG(seed + 1),
    )
    # Extract global params → matrix of shape (n_samples, n_params_global)
    n_params = N_EXPERTS * (D + 2)
    samples = zeros(GIBBS_SAMPLES, n_params)
    for s in 1:GIBBS_SAMPLES
        idx = 1
        for i in 1:N_EXPERTS, k in 1:D
            samples[s, idx] = result.w_samples[s][i, k]
            idx += 1
        end
        for i in 1:N_EXPERTS
            samples[s, idx] = result.τ_samples[s][i]; idx += 1
            samples[s, idx] = result.β_samples[s][i]; idx += 1
        end
    end
    ess_vals = [ess_simple(samples[:, p]) for p in 1:n_params]
    return (;
        min_ess = minimum(ess_vals),
        median_ess = median(ess_vals),
        t_per_iter_ms = result.time_per_sweep * 1000,
        max_rhat = NaN,   # single chain; skip
        total_time_s = result.elapsed_seconds,
    )
end

function run_nuts_config(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed))
    model = pge_ensemble_collapsed(
        data.y, data.features, data.predictions,
        N_EXPERTS, D, GH_NODES, GH_WEIGHTS,
    )
    adtype = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

    # Pre-compile on this shape (first call per shape triggers Enzyme codegen)
    try
        sample(model, NUTS(3, NUTS_DELTA; adtype=adtype), 3;
               discard_initial=3, progress=false)
    catch e
        return (; min_ess=NaN, median_ess=NaN, t_per_iter_ms=NaN,
                max_rhat=NaN, total_time_s=NaN, status="PRECOMPILE_FAIL: $(sprint(showerror, e))")
    end

    t0 = time()
    chain = sample(
        model, NUTS(NUTS_WARMUP, NUTS_DELTA; adtype=adtype), NUTS_SAMPLES;
        discard_initial=NUTS_WARMUP, progress=false,
    )
    elapsed = time() - t0

    ess_tbl = ess_rhat(chain)
    ess_vals = collect(skipmissing(ess_tbl[:, :ess]))
    rhat_vals = collect(skipmissing(ess_tbl[:, :rhat]))
    return (;
        min_ess = minimum(ess_vals),
        median_ess = median(ess_vals),
        t_per_iter_ms = elapsed * 1000 / (NUTS_WARMUP + NUTS_SAMPLES),
        max_rhat = maximum(rhat_vals),
        total_time_s = elapsed,
    )
end

# ==================================================================
# Run experiments
# ==================================================================
println("=" ^ 70)
println("ESS scaling: Gibbs (un-collapsed) vs Enzyme NUTS (collapsed)")
println("$N_EXPERTS experts, Gibbs=$(GIBBS_WARMUP)+$(GIBBS_SAMPLES), NUTS=$(NUTS_WARMUP)+$(NUTS_SAMPLES)")
println("=" ^ 70)

results = DataFrame(
    sweep=String[], method=String[], N=Int[], D=Int[],
    min_ess=Float64[], median_ess=Float64[],
    t_per_iter_ms=Float64[], max_rhat=Float64[], total_time_s=Float64[],
)

# ---------- N sweep (D = 2) ----------
println("\n[N sweep, D=2]")
for N in N_SWEEP
    print("  N=$N Gibbs...")
    r = run_gibbs_config(N, 2)
    println(" min ESS=$(round(Int, r.min_ess))  med=$(round(Int, r.median_ess))  " *
            "t/it=$(round(r.t_per_iter_ms, digits=2))ms")
    push!(results, ("N", "Gibbs", N, 2,
                    r.min_ess, r.median_ess, r.t_per_iter_ms, r.max_rhat, r.total_time_s))

    print("  N=$N NUTS ...")
    r = run_nuts_config(N, 2)
    println(" min ESS=$(round(Int, r.min_ess))  med=$(round(Int, r.median_ess))  " *
            "t/it=$(round(r.t_per_iter_ms, digits=1))ms  R-hat=$(round(r.max_rhat, digits=3))")
    push!(results, ("N", "NUTS", N, 2,
                    r.min_ess, r.median_ess, r.t_per_iter_ms, r.max_rhat, r.total_time_s))
end

# ---------- D sweep (N = 10) ----------
println("\n[D sweep, N=10]")
for D in D_SWEEP
    print("  D=$D Gibbs...")
    r = run_gibbs_config(10, D)
    println(" min ESS=$(round(Int, r.min_ess))  med=$(round(Int, r.median_ess))  " *
            "t/it=$(round(r.t_per_iter_ms, digits=2))ms")
    push!(results, ("D", "Gibbs", 10, D,
                    r.min_ess, r.median_ess, r.t_per_iter_ms, r.max_rhat, r.total_time_s))

    print("  D=$D NUTS ...")
    r = run_nuts_config(10, D)
    println(" min ESS=$(round(Int, r.min_ess))  med=$(round(Int, r.median_ess))  " *
            "t/it=$(round(r.t_per_iter_ms, digits=1))ms  R-hat=$(round(r.max_rhat, digits=3))")
    push!(results, ("D", "NUTS", 10, D,
                    r.min_ess, r.median_ess, r.t_per_iter_ms, r.max_rhat, r.total_time_s))
end

CSV.write(joinpath(RESULTS_DIR, "scaling_ess.csv"), results)
println("\nSaved $(joinpath(RESULTS_DIR, "scaling_ess.csv"))")

# ==================================================================
# Plot — 2×2 figure
# ==================================================================
println("\nRendering figure...")
fig = Figure(size=(1400, 900), fontsize=13)
Label(fig[0, 1:2],
      "ESS and cost scaling — Gibbs (un-collapsed) vs Enzyme NUTS (collapsed)\n" *
      "$N_EXPERTS experts, Gibbs=$(GIBBS_SAMPLES) samples, NUTS=$(NUTS_SAMPLES) samples",
      fontsize=15, font=:bold)

function line_with_markers!(ax, xs, ys, color, label; marker=:circle)
    lines!(ax, xs, ys; color=color, linewidth=2, label=label)
    scatter!(ax, xs, ys; color=color, marker=marker, markersize=10)
end

# Panel 1: min ESS vs N (D=2)
ax11 = Axis(fig[1, 1]; xlabel="N (observations)", ylabel="min ESS",
            title="min ESS vs N (D=2)", xscale=log10, yscale=log10)
df_N = filter(r -> r.sweep == "N", results)
line_with_markers!(ax11, df_N[df_N.method .== "Gibbs", :N], max.(df_N[df_N.method .== "Gibbs", :min_ess], 1.0),
                   :crimson, "Gibbs")
line_with_markers!(ax11, df_N[df_N.method .== "NUTS",  :N], max.(df_N[df_N.method .== "NUTS",  :min_ess], 1.0),
                   :seagreen, "NUTS"; marker=:diamond)
hlines!(ax11, [100]; color=:gray, linestyle=:dash)
text!(ax11, N_SWEEP[end], 110; text="ESS=100 target", color=:gray,
      align=(:right, :bottom), fontsize=10)
axislegend(ax11; position=:lt)

# Panel 2: min ESS vs D (N=10)
ax12 = Axis(fig[1, 2]; xlabel="D (feature dim)", ylabel="min ESS",
            title="min ESS vs D (N=10)", xscale=log10, yscale=log10)
df_D = filter(r -> r.sweep == "D", results)
line_with_markers!(ax12, df_D[df_D.method .== "Gibbs", :D], max.(df_D[df_D.method .== "Gibbs", :min_ess], 1.0),
                   :crimson, "Gibbs")
line_with_markers!(ax12, df_D[df_D.method .== "NUTS",  :D], max.(df_D[df_D.method .== "NUTS",  :min_ess], 1.0),
                   :seagreen, "NUTS"; marker=:diamond)
hlines!(ax12, [100]; color=:gray, linestyle=:dash)
axislegend(ax12; position=:lb)

# Panel 3: ms/iter vs N (D=2)
ax21 = Axis(fig[2, 1]; xlabel="N (observations)", ylabel="ms / iter",
            title="Per-iteration cost vs N (D=2)", xscale=log10, yscale=log10)
line_with_markers!(ax21, df_N[df_N.method .== "Gibbs", :N], df_N[df_N.method .== "Gibbs", :t_per_iter_ms],
                   :crimson, "Gibbs")
line_with_markers!(ax21, df_N[df_N.method .== "NUTS",  :N], df_N[df_N.method .== "NUTS",  :t_per_iter_ms],
                   :seagreen, "NUTS"; marker=:diamond)
axislegend(ax21; position=:lt)

# Panel 4: ms/iter vs D (N=10)
ax22 = Axis(fig[2, 2]; xlabel="D (feature dim)", ylabel="ms / iter",
            title="Per-iteration cost vs D (N=10)", xscale=log10, yscale=log10)
line_with_markers!(ax22, df_D[df_D.method .== "Gibbs", :D], df_D[df_D.method .== "Gibbs", :t_per_iter_ms],
                   :crimson, "Gibbs")
line_with_markers!(ax22, df_D[df_D.method .== "NUTS",  :D], df_D[df_D.method .== "NUTS",  :t_per_iter_ms],
                   :seagreen, "NUTS"; marker=:diamond)
axislegend(ax22; position=:lt)

save(joinpath(RESULTS_DIR, "scaling_ess.png"), fig; px_per_unit=3)
save(joinpath(RESULTS_DIR, "scaling_ess.pdf"), fig)
println("Saved scaling_ess.{png,pdf}")

println("\n" * "=" ^ 70)
println("Final table")
println("=" ^ 70)
show(stdout, results; allrows=true, allcols=true)
println()
