# Extend MCMC sweep to D ∈ [25, 65] at N ∈ [10, 50].
# Appends to the existing pareto_quality_mcmc.csv.

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
const CSV_PATH = joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv")

const N_EXPERTS = 7
const K_GH = 15
const PRIOR_PREC_W = 0.01
const PRIOR_RATE_τ = 1.0
const PRIOR_RATE_β = 1e3
const GIBBS_WARMUP = 500
const GIBBS_SAMPLES = 2000
const NUTS_WARMUP = 200
const NUTS_SAMPLES = 1000
const NUTS_DELTA = 0.95

const GH_NODES, GH_WEIGHTS = gausshermite_gw(K_GH)

# Only NEW (N, D) pairs — amortize compile by grouping by D
const NEW_CONFIGS = [(10, 25), (50, 25), (10, 65), (50, 65)]

function empty_results()
    return DataFrame(
        method=String[], N=Int[], D=Int[],
        rel_err_total=Float64[], rel_err_w=Float64[], rel_err_τ=Float64[], rel_err_β=Float64[],
        cov95=Float64[], cov50=Float64[],
        std_err_total=Float64[],
        total_time_s=Float64[],
    )
end
append_row!(row) = CSV.write(CSV_PATH, begin d = empty_results(); push!(d, row); d end; append=true)

function metrics_from_samples(samples::AbstractMatrix, truth::AbstractVector)
    n_params = length(truth)
    @assert size(samples, 2) == n_params
    post_mean = vec(mean(samples; dims=1))
    post_std  = vec(std(samples;  dims=1))
    q025  = [quantile(samples[:, p], 0.025) for p in 1:n_params]
    q975  = [quantile(samples[:, p], 0.975) for p in 1:n_params]
    q25   = [quantile(samples[:, p], 0.25)  for p in 1:n_params]
    q75   = [quantile(samples[:, p], 0.75)  for p in 1:n_params]
    cov95 = mean((truth .>= q025) .& (truth .<= q975))
    cov50 = mean((truth .>= q25)  .& (truth .<= q75))
    rel_err  = norm(post_mean .- truth) / max(norm(truth), 1e-12)
    post_std_safe = max.(post_std, 1e-12)
    std_err  = norm((post_mean .- truth) ./ post_std_safe) / sqrt(n_params)
    return (; rel_err, cov95, cov50, std_err, post_mean, post_std)
end

function build_truth_vector(data, n_experts, d)
    v = Float64[]
    for i in 1:n_experts, k in 1:d
        push!(v, data.w_true[i][k])
    end
    for i in 1:n_experts
        push!(v, data.τ_true[i]); push!(v, data.β_true[i])
    end
    w_end = n_experts * d
    τ_positions = [w_end + 2*(i-1) + 1 for i in 1:n_experts]
    β_positions = [w_end + 2*(i-1) + 2 for i in 1:n_experts]
    return (v, 1:w_end, τ_positions, β_positions)
end

function gibbs_samples_matrix(result, n_experts, d)
    nS = length(result.w_samples); n_params = n_experts * (d + 2)
    S = zeros(nS, n_params)
    for s in 1:nS
        idx = 1
        for i in 1:n_experts, k in 1:d
            S[s, idx] = result.w_samples[s][i, k]; idx += 1
        end
        for i in 1:n_experts
            S[s, idx] = result.τ_samples[s][i]; idx += 1
            S[s, idx] = result.β_samples[s][i]; idx += 1
        end
    end
    return S
end

function run_gibbs_cfg(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed),
                                   τ_range=(0.5, 5.0))
    result = run_gibbs(
        data.y, data.features, data.predictions;
        n_experts=N_EXPERTS, d=D,
        n_warmup=GIBBS_WARMUP, n_samples=GIBBS_SAMPLES,
        rng=StableRNG(seed + 1),
        prior_prec_w=PRIOR_PREC_W, prior_rate_τ=PRIOR_RATE_τ, prior_rate_β=PRIOR_RATE_β,
    )
    S = gibbs_samples_matrix(result, N_EXPERTS, D)
    truth, w_idx, τ_idx, β_idx = build_truth_vector(data, N_EXPERTS, D)
    mall = metrics_from_samples(S, truth)
    erw = norm(vec(mall.post_mean[w_idx]) .- truth[w_idx]) / max(norm(truth[w_idx]), 1e-12)
    erτ = norm(vec(mall.post_mean[τ_idx]) .- truth[τ_idx]) / max(norm(truth[τ_idx]), 1e-12)
    erβ = norm(vec(mall.post_mean[β_idx]) .- truth[β_idx]) / max(norm(truth[β_idx]), 1e-12)
    return (; rel_err=mall.rel_err, rel_err_w=erw, rel_err_τ=erτ, rel_err_β=erβ,
             cov95=mall.cov95, cov50=mall.cov50, std_err=mall.std_err,
             total_time_s=result.elapsed_seconds)
end

function run_nuts_cfg(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed),
                                   τ_range=(0.5, 5.0))
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

    n_params = N_EXPERTS * (D + 2); nS = NUTS_SAMPLES
    S = zeros(nS, n_params)
    idx = 1
    for i in 1:N_EXPERTS, k in 1:D
        S[:, idx] = vec(Array(chain[:, Symbol("w[$k, $i]"), :])); idx += 1
    end
    for i in 1:N_EXPERTS
        S[:, idx] = vec(Array(chain[:, Symbol("τ[$i]"), :])); idx += 1
        S[:, idx] = vec(Array(chain[:, Symbol("β[$i]"), :])); idx += 1
    end

    truth, w_idx, τ_idx, β_idx = build_truth_vector(data, N_EXPERTS, D)
    mall = metrics_from_samples(S, truth)
    erw = norm(vec(mall.post_mean[w_idx]) .- truth[w_idx]) / max(norm(truth[w_idx]), 1e-12)
    erτ = norm(vec(mall.post_mean[τ_idx]) .- truth[τ_idx]) / max(norm(truth[τ_idx]), 1e-12)
    erβ = norm(vec(mall.post_mean[β_idx]) .- truth[β_idx]) / max(norm(truth[β_idx]), 1e-12)
    return (; rel_err=mall.rel_err, rel_err_w=erw, rel_err_τ=erτ, rel_err_β=erβ,
             cov95=mall.cov95, cov50=mall.cov50, std_err=mall.std_err,
             total_time_s=elapsed)
end

println("=" ^ 70)
println("Extending MCMC sweep to big D: $NEW_CONFIGS")
println("=" ^ 70)

for (N, D) in NEW_CONFIGS
    println("\n[N=$N, D=$D]")
    print("  Gibbs...")
    rg = run_gibbs_cfg(N, D)
    println(" err=$(round(rg.rel_err, digits=3))  cov95=$(round(rg.cov95, digits=2))  " *
            "std_err=$(round(rg.std_err, digits=2))  t=$(round(rg.total_time_s, digits=2))s")
    append_row!(("Gibbs", N, D, rg.rel_err, rg.rel_err_w, rg.rel_err_τ, rg.rel_err_β,
                 rg.cov95, rg.cov50, rg.std_err, rg.total_time_s))

    print("  NUTS ... ")
    flush(stdout)
    rn = run_nuts_cfg(N, D)
    println("err=$(round(rn.rel_err, digits=3))  cov95=$(round(rn.cov95, digits=2))  " *
            "std_err=$(round(rn.std_err, digits=2))  t=$(round(rn.total_time_s, digits=1))s")
    append_row!(("NUTS", N, D, rn.rel_err, rn.rel_err_w, rn.rel_err_τ, rn.rel_err_β,
                 rn.cov95, rn.cov50, rn.std_err, rn.total_time_s))
end

println("\nSaved (appended) to $CSV_PATH")
