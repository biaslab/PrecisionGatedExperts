using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

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
const CSV_PATH = joinpath(RESULTS_DIR, "scaling_v2_mcmc.csv")

# ==================================================================
# Config
# ==================================================================
const N_EXPERTS = 7
const K_GH = 15

const PRIOR_PREC_W = 0.01
const PRIOR_RATE_τ = 1.0
const PRIOR_RATE_β = 1e3

const GIBBS_WARMUP = 500
const GIBBS_SAMPLES = 2000

const NUTS_WARMUP = 200
const NUTS_SAMPLES = 1000   # ≥1000 so ESS is a reliable estimate
const NUTS_DELTA = 0.95

# Sweep — 3 × 3 grid
const N_VALUES = [10, 50, 250]
const D_VALUES = [2, 5, 10]

const GH_NODES, GH_WEIGHTS = gausshermite_gw(K_GH)

# ==================================================================
# Header / incremental CSV
# ==================================================================
function empty_results()
    return DataFrame(
        method=String[], N=Int[], D=Int[],
        min_ess=Float64[], median_ess=Float64[],
        t_per_iter_ms=Float64[], max_rhat=Float64[], total_time_s=Float64[],
    )
end

CSV.write(CSV_PATH, empty_results())

function append_row!(row)
    df = empty_results()
    push!(df, row)
    CSV.write(CSV_PATH, df; append=true)
end

# ==================================================================
# Runners
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
    sample(model, NUTS(3, NUTS_DELTA; adtype=adtype), 3;
           discard_initial=3, progress=false)
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
# Run — outer loop D to amortize Enzyme compile
# ==================================================================
println("=" ^ 60)
println("scaling_v2: Gibbs + NUTS (1000 samples) at N ∈ $N_VALUES × D ∈ $D_VALUES")
println("=" ^ 60)

for D in D_VALUES
    println("\n[D = $D]")
    for N in N_VALUES
        print("  N=$N  Gibbs...")
        rg = run_gibbs_config(N, D)
        println(" ESS=$(round(Int, rg.min_ess))/med=$(round(Int, rg.median_ess))  " *
                "t/it=$(round(rg.t_per_iter_ms, digits=2))ms  total=$(round(rg.total_time_s, digits=2))s")
        append_row!(("Gibbs", N, D, rg.min_ess, rg.median_ess, rg.t_per_iter_ms, rg.max_rhat, rg.total_time_s))

        print("  N=$N  NUTS ...")
        rn = run_nuts_config(N, D)
        println(" ESS=$(round(Int, rn.min_ess))/med=$(round(Int, rn.median_ess))  " *
                "t/it=$(round(rn.t_per_iter_ms, digits=1))ms  R-hat=$(round(rn.max_rhat, digits=3))  total=$(round(rn.total_time_s, digits=1))s")
        append_row!(("NUTS", N, D, rn.min_ess, rn.median_ess, rn.t_per_iter_ms, rn.max_rhat, rn.total_time_s))
    end
end

println("\nSaved CSV: $CSV_PATH")
