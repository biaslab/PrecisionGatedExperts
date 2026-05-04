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
# Small sweep — incremental CSV writes after every config
# ==================================================================
const N_EXPERTS = 7
const K_GH = 15

# Tight priors (same setting as d=2 marginals figure)
const PRIOR_PREC_W = 0.01
const PRIOR_RATE_τ = 1.0
const PRIOR_RATE_β = 1e3

# Gibbs: cheap, generous budget
const GIBBS_WARMUP = 500
const GIBBS_SAMPLES = 2000

# NUTS: small but enough for ESS signal
const NUTS_WARMUP = 200
const NUTS_SAMPLES = 500
const NUTS_DELTA = 0.95

# Primary sweep: N scaling at D=2 — start small so we see per-iter cost trend fast
const N_SWEEP_AT_D2 = [10, 20, 50, 100]

# Secondary: D sweep at a single moderate N
const D_SWEEP_AT_N50 = [2, 5, 10]
const D_SWEEP_N = 50

const GH_NODES, GH_WEIGHTS = gausshermite_gw(K_GH)

const CSV_PATH = joinpath(RESULTS_DIR, "scaling_ess_small.csv")

function empty_results()
    return DataFrame(
        sweep=String[], method=String[], N=Int[], D=Int[],
        min_ess=Float64[], median_ess=Float64[],
        t_per_iter_ms=Float64[], max_rhat=Float64[], total_time_s=Float64[],
    )
end

# initialize CSV
CSV.write(CSV_PATH, empty_results())
println("Incremental CSV: $CSV_PATH")

function append_row!(row)
    df = empty_results()
    push!(df, row)
    CSV.write(CSV_PATH, df; append=true)
end

# ==================================================================
# Per-config runners
# ==================================================================
function run_gibbs_config(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(
        N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed),
        τ_range=(0.5, 5.0),
    )
    result = run_gibbs(
        data.y, data.features, data.predictions;
        n_experts=N_EXPERTS, d=D,
        n_warmup=GIBBS_WARMUP, n_samples=GIBBS_SAMPLES,
        rng=StableRNG(seed + 1),
        prior_prec_w=PRIOR_PREC_W, prior_rate_τ=PRIOR_RATE_τ, prior_rate_β=PRIOR_RATE_β,
    )
    # Gather all global params into a matrix (n_samples, n_params)
    n_params = N_EXPERTS * (D + 2)
    samples = zeros(GIBBS_SAMPLES, n_params)
    for s in 1:GIBBS_SAMPLES
        idx = 1
        for i in 1:N_EXPERTS, k in 1:D
            samples[s, idx] = result.w_samples[s][i, k]; idx += 1
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
        max_rhat = NaN,
        total_time_s = result.elapsed_seconds,
    )
end

function run_nuts_config(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(
        N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed),
        τ_range=(0.5, 5.0),
    )
    model = pge_ensemble_collapsed(
        data.y, data.features, data.predictions,
        N_EXPERTS, D, GH_NODES, GH_WEIGHTS;
        prior_prec_w=PRIOR_PREC_W, prior_rate_τ=PRIOR_RATE_τ, prior_rate_β=PRIOR_RATE_β,
    )
    adtype = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

    # Touch-compile (3 iters)
    try
        sample(model, NUTS(3, NUTS_DELTA; adtype=adtype), 3;
               discard_initial=3, progress=false)
    catch e
        return (; min_ess=NaN, median_ess=NaN, t_per_iter_ms=NaN,
                max_rhat=NaN, total_time_s=NaN)
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
# Run the sweeps (incrementally saving)
# ==================================================================
println("=" ^ 60)
println("Sweep: N at D=2, then D at N=$D_SWEEP_N")
println("  Gibbs=$(GIBBS_SAMPLES), NUTS=$(NUTS_WARMUP)+$(NUTS_SAMPLES)")
println("=" ^ 60)

println("\n[N sweep, D=2]")
for N in N_SWEEP_AT_D2
    print("  N=$N, D=2  Gibbs...")
    rg = run_gibbs_config(N, 2)
    println(" ESS=$(round(Int, rg.min_ess)) / med=$(round(Int, rg.median_ess))  " *
            "t/it=$(round(rg.t_per_iter_ms, digits=2))ms")
    append_row!(("N@D2", "Gibbs", N, 2, rg.min_ess, rg.median_ess,
                 rg.t_per_iter_ms, rg.max_rhat, rg.total_time_s))

    print("  N=$N, D=2  NUTS ...")
    rn = run_nuts_config(N, 2)
    println(" ESS=$(round(Int, rn.min_ess)) / med=$(round(Int, rn.median_ess))  " *
            "t/it=$(round(rn.t_per_iter_ms, digits=1))ms  R-hat=$(round(rn.max_rhat, digits=3))")
    append_row!(("N@D2", "NUTS", N, 2, rn.min_ess, rn.median_ess,
                 rn.t_per_iter_ms, rn.max_rhat, rn.total_time_s))
end

println("\n[D sweep, N=$D_SWEEP_N]")
for D in D_SWEEP_AT_N50
    print("  N=$D_SWEEP_N, D=$D  Gibbs...")
    rg = run_gibbs_config(D_SWEEP_N, D)
    println(" ESS=$(round(Int, rg.min_ess)) / med=$(round(Int, rg.median_ess))  " *
            "t/it=$(round(rg.t_per_iter_ms, digits=2))ms")
    append_row!(("D@N$D_SWEEP_N", "Gibbs", D_SWEEP_N, D, rg.min_ess, rg.median_ess,
                 rg.t_per_iter_ms, rg.max_rhat, rg.total_time_s))

    print("  N=$D_SWEEP_N, D=$D  NUTS ...")
    rn = run_nuts_config(D_SWEEP_N, D)
    println(" ESS=$(round(Int, rn.min_ess)) / med=$(round(Int, rn.median_ess))  " *
            "t/it=$(round(rn.t_per_iter_ms, digits=1))ms  R-hat=$(round(rn.max_rhat, digits=3))")
    append_row!(("D@N$D_SWEEP_N", "NUTS", D_SWEEP_N, D, rn.min_ess, rn.median_ess,
                 rn.t_per_iter_ms, rn.max_rhat, rn.total_time_s))
end

# ==================================================================
# Plot
# ==================================================================
println("\nRendering figure...")
results = CSV.read(CSV_PATH, DataFrame)

fig = Figure(size=(1400, 900), fontsize=13)
Label(fig[0, 1:2],
      "ESS and cost scaling — Gibbs (un-collapsed) vs Enzyme NUTS (collapsed), d=2 priors\n" *
      "Gibbs $GIBBS_SAMPLES samples, NUTS $(NUTS_WARMUP)+$(NUTS_SAMPLES)",
      fontsize=14, font=:bold)

function line_with_markers!(ax, xs, ys, color, label; marker=:circle)
    lines!(ax, xs, ys; color=color, linewidth=2, label=label)
    scatter!(ax, xs, ys; color=color, marker=marker, markersize=10)
end

function safe_pos(xs)
    return [x <= 0 || isnan(x) ? 1.0 : x for x in xs]
end

# Panel 1: ESS vs N (D=2)
df_N = filter(r -> r.sweep == "N@D2", results)
ax11 = Axis(fig[1, 1]; xlabel="N (observations)", ylabel="min ESS",
            title="ESS vs N  (D=2)", xscale=log10, yscale=log10)
line_with_markers!(ax11, df_N[df_N.method .== "Gibbs", :N], safe_pos(df_N[df_N.method .== "Gibbs", :min_ess]), :crimson, "Gibbs")
line_with_markers!(ax11, df_N[df_N.method .== "NUTS",  :N], safe_pos(df_N[df_N.method .== "NUTS",  :min_ess]), :seagreen, "NUTS"; marker=:diamond)
hlines!(ax11, [100]; color=:gray, linestyle=:dash)
text!(ax11, df_N.N[end], 110; text="ESS=100 target", color=:gray, align=(:right, :bottom), fontsize=10)
axislegend(ax11; position=:lt)

# Panel 2: ESS vs D (N=D_SWEEP_N)
df_D = filter(r -> r.sweep == "D@N$D_SWEEP_N", results)
ax12 = Axis(fig[1, 2]; xlabel="D (feature dim)", ylabel="min ESS",
            title="ESS vs D  (N=$D_SWEEP_N)", xscale=log10, yscale=log10)
line_with_markers!(ax12, df_D[df_D.method .== "Gibbs", :D], safe_pos(df_D[df_D.method .== "Gibbs", :min_ess]), :crimson, "Gibbs")
line_with_markers!(ax12, df_D[df_D.method .== "NUTS",  :D], safe_pos(df_D[df_D.method .== "NUTS",  :min_ess]), :seagreen, "NUTS"; marker=:diamond)
hlines!(ax12, [100]; color=:gray, linestyle=:dash)
axislegend(ax12; position=:lb)

# Panel 3: ms/iter vs N
ax21 = Axis(fig[2, 1]; xlabel="N (observations)", ylabel="ms / iter",
            title="Per-iteration cost vs N  (D=2)", xscale=log10, yscale=log10)
line_with_markers!(ax21, df_N[df_N.method .== "Gibbs", :N], df_N[df_N.method .== "Gibbs", :t_per_iter_ms], :crimson, "Gibbs")
line_with_markers!(ax21, df_N[df_N.method .== "NUTS",  :N], df_N[df_N.method .== "NUTS",  :t_per_iter_ms], :seagreen, "NUTS"; marker=:diamond)
axislegend(ax21; position=:lt)

# Panel 4: ms/iter vs D
ax22 = Axis(fig[2, 2]; xlabel="D (feature dim)", ylabel="ms / iter",
            title="Per-iteration cost vs D  (N=$D_SWEEP_N)", xscale=log10, yscale=log10)
line_with_markers!(ax22, df_D[df_D.method .== "Gibbs", :D], df_D[df_D.method .== "Gibbs", :t_per_iter_ms], :crimson, "Gibbs")
line_with_markers!(ax22, df_D[df_D.method .== "NUTS",  :D], df_D[df_D.method .== "NUTS",  :t_per_iter_ms], :seagreen, "NUTS"; marker=:diamond)
axislegend(ax22; position=:lt)

save(joinpath(RESULTS_DIR, "scaling_ess_small.png"), fig; px_per_unit=3)
save(joinpath(RESULTS_DIR, "scaling_ess_small.pdf"), fig)
println("Saved scaling_ess_small.{png,pdf}")

println("\n" * "=" ^ 60)
println("Results table")
println("=" ^ 60)
show(stdout, results; allrows=true, allcols=true)
println()
