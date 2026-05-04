using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using StableRNGs
using Statistics
using Turing
using MCMCChains
using ADTypes
using CSV
using DataFrames

import Enzyme

include("data_generation.jl")
include("gibbs_handcoded.jl")
include("turing_model_collapsed.jl")

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

const N = 10
const N_EXPERTS = 7
const D = 2
const K_GH = 15

const GIBBS_WARMUP = 1000
const GIBBS_SAMPLES = 50000

const NUTS_WARMUP = 1000
const NUTS_SAMPLES = 2000
const NUTS_DELTA = 0.95

# Priors (matched to VMP for fair comparison — tight τ so dynamic VMP converges)
const PRIOR_PREC_W = 0.01
const PRIOR_RATE_τ = 1.0
const PRIOR_RATE_β = 1e3

println("=" ^ 60)
println("save_chains_d2: Gibbs + collapsed NUTS at N=$N, d=$D, n_experts=$N_EXPERTS")
println("=" ^ 60)

data = generate_synthetic_data(
    N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(42),
    τ_range=(0.5, 5.0),   # tight range so posteriors can recover truth at N=10
)

# --- Save ground truth ---
truth_df = DataFrame(
    expert = Int[],
    param  = String[],
    value  = Float64[],
)
for i in 1:N_EXPERTS
    push!(truth_df, (i, "w[1]", data.w_true[i][1]))
    push!(truth_df, (i, "w[2]", data.w_true[i][2]))
    push!(truth_df, (i, "τ",    data.τ_true[i]))
    push!(truth_df, (i, "β",    data.β_true[i]))
end
CSV.write(joinpath(RESULTS_DIR, "truth_d2.csv"), truth_df)

# --- Gibbs ---
println("\n[Gibbs] $(GIBBS_SAMPLES) samples...")
gibbs_result = run_gibbs(
    data.y, data.features, data.predictions;
    n_experts=N_EXPERTS, d=D,
    n_warmup=GIBBS_WARMUP, n_samples=GIBBS_SAMPLES,
    rng=StableRNG(1),
    prior_prec_w=PRIOR_PREC_W, prior_rate_τ=PRIOR_RATE_τ, prior_rate_β=PRIOR_RATE_β,
)
println("  done in $(round(gibbs_result.elapsed_seconds, digits=1))s")

gibbs_df = DataFrame()
for i in 1:N_EXPERTS
    for k in 1:D
        col = Symbol("w_e$(i)_d$(k)")
        gibbs_df[!, col] = [gibbs_result.w_samples[s][i, k] for s in 1:GIBBS_SAMPLES]
    end
    gibbs_df[!, Symbol("tau_e$i")] = [gibbs_result.τ_samples[s][i] for s in 1:GIBBS_SAMPLES]
    gibbs_df[!, Symbol("beta_e$i")] = [gibbs_result.β_samples[s][i] for s in 1:GIBBS_SAMPLES]
end
CSV.write(joinpath(RESULTS_DIR, "gibbs_samples_d2.csv"), gibbs_df)
println("  Saved gibbs_samples_d2.csv  ($(size(gibbs_df, 1)) × $(size(gibbs_df, 2)))")

# --- NUTS collapsed (Enzyme) ---
println("\n[NUTS collapsed, Enzyme.Reverse] $(NUTS_WARMUP) warmup + $(NUTS_SAMPLES) samples...")
gh_nodes, gh_weights = gausshermite_gw(K_GH)
model = pge_ensemble_collapsed(
    data.y, data.features, data.predictions,
    N_EXPERTS, D, gh_nodes, gh_weights;
    prior_prec_w=PRIOR_PREC_W, prior_rate_τ=PRIOR_RATE_τ, prior_rate_β=PRIOR_RATE_β,
)
adtype = AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

print("  Pre-compiling...")
t_pre = @elapsed sample(model, NUTS(3, NUTS_DELTA; adtype=adtype), 3;
                         discard_initial=3, progress=false)
println(" $(round(t_pre, digits=1))s")

print("  Sampling...")
t0 = time()
chain = sample(model, NUTS(NUTS_WARMUP, NUTS_DELTA; adtype=adtype), NUTS_SAMPLES;
               discard_initial=NUTS_WARMUP, progress=false)
t_nuts = time() - t0
println(" $(round(t_nuts, digits=1))s  ($(round(t_nuts*1000/(NUTS_WARMUP+NUTS_SAMPLES), digits=1))ms/iter)")

nuts_df = DataFrame()
for i in 1:N_EXPERTS
    for k in 1:D
        col = Symbol("w_e$(i)_d$(k)")
        nuts_df[!, col] = vec(Array(chain[:, Symbol("w[$k, $i]"), :]))
    end
    nuts_df[!, Symbol("tau_e$i")] = vec(Array(chain[:, Symbol("τ[$i]"), :]))
    nuts_df[!, Symbol("beta_e$i")] = vec(Array(chain[:, Symbol("β[$i]"), :]))
end
CSV.write(joinpath(RESULTS_DIR, "nuts_samples_d2.csv"), nuts_df)
println("  Saved nuts_samples_d2.csv  ($(size(nuts_df, 1)) × $(size(nuts_df, 2)))")

# Diagnostics summary
ess_tbl = ess_rhat(chain)
println("\nNUTS diagnostics: min ESS = $(round(minimum(skipmissing(ess_tbl[:, :ess])), digits=1)), " *
        "max R-hat = $(round(maximum(skipmissing(ess_tbl[:, :rhat])), digits=3))")
println("Data:  $(joinpath(RESULTS_DIR, "truth_d2.csv"))")
