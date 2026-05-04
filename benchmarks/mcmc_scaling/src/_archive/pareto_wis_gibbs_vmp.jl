# Weighted Interval Score (WIS) sweep for Gibbs and VMP-dynamic over the full
# (N, D) grid.  WIS approximates CRPS from a few quantile pairs and is
# a proper scoring rule that jointly penalises wrong location AND wrong spread.
#
# Per parameter with samples s and truth y:
#     WIS = [w_0 · |y − median(s)| + Σ_k (α_k/2)·(U_k − L_k)
#            + Σ_k ((L_k − y)_+ + (y − U_k)_+)] / (K + 1/2)
# with α levels {0.5, 0.2, 0.1, 0.05} (→ 50/80/90/95 % central intervals),
# w_0 = 1/2, w_k = α_k/2.
#
# We report per-cell:
#   • mean raw WIS across K·D parameters  (wis)
#   • mean *relative* WIS, normalised per-param by max(|truth|, 1e-3)  (wis_rel)
#
# Output: results/pareto_wis.csv with columns
#   method, N, D, wis, wis_rel, total_time_s, status

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using LinearAlgebra
using StableRNGs
using Statistics
using DataFrames
using CSV
using Distributions

include(joinpath(@__DIR__, "data_generation.jl"))
include(joinpath(@__DIR__, "gibbs_handcoded.jl"))

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamily

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const CSV_PATH    = joinpath(RESULTS_DIR, "pareto_wis.csv")

const N_EXPERTS    = 7
const PRIOR_PREC_W = 0.01
const PRIOR_RATE_τ = 1.0
const PRIOR_RATE_β = 1e3
const GIBBS_WARMUP = 500
const GIBBS_SAMPLES = 2000
const VMP_ITERATIONS = 10

const N_VALUES = [10, 50, 250]
const D_VALUES = [2, 5, 10, 25, 65]

# α levels used by the COVID-19 Forecast Hub — these weight WIS ≈ CRPS
const ALPHAS = [0.5, 0.2, 0.1, 0.05]

# ---------------------------------------------------------------------------
# WIS computation
# ---------------------------------------------------------------------------
"""
    wis_from_quantiles(truth, median_val, lowers, uppers, alphas)

Quantile-based WIS.  `lowers[k]` and `uppers[k]` are the (α_k/2, 1−α_k/2)
quantiles.  Returns WIS for a single truth value.
"""
function wis_from_quantiles(truth::Real, median_val::Real,
                            lowers::AbstractVector, uppers::AbstractVector,
                            alphas::AbstractVector)
    K = length(alphas)
    s = 0.5 * abs(truth - median_val)
    for k in 1:K
        α = alphas[k]; L = lowers[k]; U = uppers[k]
        width   = U - L
        penalty = max(L - truth, 0.0) + max(truth - U, 0.0)
        s += (α/2) * width + penalty
    end
    return s / (K + 0.5)
end

# For MCMC: take quantiles from a sample vector.
function wis_from_samples(samples::AbstractVector, truth::Real; alphas=ALPHAS)
    med = median(samples)
    lowers = [quantile(samples, α/2)     for α in alphas]
    uppers = [quantile(samples, 1 - α/2) for α in alphas]
    return wis_from_quantiles(truth, med, lowers, uppers, alphas)
end

# For VMP: take quantiles from an analytic distribution.
function wis_from_distribution(dist, truth::Real; alphas=ALPHAS)
    med = quantile(dist, 0.5)
    lowers = [quantile(dist, α/2)     for α in alphas]
    uppers = [quantile(dist, 1 - α/2) for α in alphas]
    return wis_from_quantiles(truth, med, lowers, uppers, alphas)
end

# ---------------------------------------------------------------------------
# Gibbs: build n_samples × n_params matrix, per-param WIS, then averages.
# ---------------------------------------------------------------------------
function gibbs_samples_and_truth(data, result, n_experts, d)
    nS = length(result.w_samples)
    n_params = n_experts * (d + 2)
    S = zeros(nS, n_params)
    truth = zeros(n_params)
    idx = 1
    for i in 1:n_experts, k in 1:d
        for s in 1:nS; S[s, idx] = result.w_samples[s][i, k]; end
        truth[idx] = data.w_true[i][k]
        idx += 1
    end
    for i in 1:n_experts
        for s in 1:nS; S[s, idx] = result.τ_samples[s][i]; end
        truth[idx] = data.τ_true[i]; idx += 1
        for s in 1:nS; S[s, idx] = result.β_samples[s][i]; end
        truth[idx] = data.β_true[i]; idx += 1
    end
    return S, truth
end

function gibbs_wis(N::Int, D::Int; seed=42)
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
    S, truth = gibbs_samples_and_truth(data, result, N_EXPERTS, D)
    nparam = length(truth)
    # Build per-param WIS vector, then slice into (w, τ, β) families.
    w_per = [wis_from_samples(view(S, :, p), truth[p]) for p in 1:nparam]
    w_end = N_EXPERTS * D
    τ_positions = [w_end + 2*(i-1) + 1 for i in 1:N_EXPERTS]
    β_positions = [w_end + 2*(i-1) + 2 for i in 1:N_EXPERTS]
    wis_w = mean(w_per[1:w_end])
    wis_τ = mean(w_per[τ_positions])
    wis_β = mean(w_per[β_positions])
    wis_all = sum(w_per) / nparam
    rel_all = sum(w_per[p] / max(abs(truth[p]), 1e-3) for p in 1:nparam) / nparam
    return (; wis = wis_all,
             wis_w, wis_τ, wis_β,
             wis_rel = rel_all,
             total_time_s = result.elapsed_seconds,
             status = "OK")
end

# ---------------------------------------------------------------------------
# VMP: analytic per-parameter distribution (Normal for w, Gamma for τ, β).
# ---------------------------------------------------------------------------
function normal_from_mv(mv_dist, k::Int)
    return Distributions.Normal(mean(mv_dist)[k], sqrt(cov(mv_dist)[k, k]))
end

function gamma_from_ef(g)
    α = shape(g)
    rate = 1.0 / scale(g)        # ExponentialFamily Gamma uses scale = 1/rate
    return Distributions.Gamma(α, 1.0 / rate)
end

function vmp_wis(N::Int, D::Int; seed=42)
    data = generate_synthetic_data(
        N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(seed),
        τ_range=(0.5, 5.0),
    )
    priors = Dict{Symbol, Any}(
        :w => [MvNormalMeanScalePrecision(zeros(D), 0.01) for _ in 1:N_EXPERTS],
        :τ => [GammaShapeRate(1.0, 1.0)                   for _ in 1:N_EXPERTS],
        :β => [GammaShapeRate(1.0, 1e3)                   for _ in 1:N_EXPERTS],
    )
    model = ProbabilisticEnsembling.univariate_dynamic_ensemble(
        n_forecasters=N_EXPERTS, n_obs=N, priors=priors,
    )
    constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
    init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
    spec = (
        prediction_type = ProbabilisticEnsembling.Univariate(),
        model_type      = ProbabilisticEnsembling.Dynamic(),
        inference_iterations = VMP_ITERATIONS,
        subsample_size = nothing, subsample_percentage = nothing,
    )
    vmp_data = (y=data.y, features=data.features, predictions=data.predictions)
    t0 = time()
    local w_post, τ_post, β_post
    try
        result = ProbabilisticEnsembling.run_training_rxinfer(
            spec, model, vmp_data;
            constraints=constraints, initialization=init, showprogress=false,
        )
        w_post = result.posteriors[:w][end]
        τ_post = result.posteriors[:τ][end]
        β_post = result.posteriors[:β][end]
    catch e
        elapsed = time() - t0
        msg = "FAIL:" * first(split(sprint(showerror, e), '\n'))
        return (; wis=NaN, wis_w=NaN, wis_τ=NaN, wis_β=NaN, wis_rel=NaN,
                  total_time_s=elapsed, status=msg)
    end
    elapsed = time() - t0

    nparam = N_EXPERTS * (D + 2)
    wis_raw = 0.0
    wis_rel = 0.0
    sum_w_fam = 0.0; n_w_fam = 0
    sum_τ_fam = 0.0; n_τ_fam = 0
    sum_β_fam = 0.0; n_β_fam = 0
    for i in 1:N_EXPERTS, k in 1:D
        d = normal_from_mv(w_post[i], k)
        t = data.w_true[i][k]
        w_p = wis_from_distribution(d, t)
        wis_raw += w_p
        wis_rel += w_p / max(abs(t), 1e-3)
        sum_w_fam += w_p; n_w_fam += 1
    end
    for i in 1:N_EXPERTS
        dτ = gamma_from_ef(τ_post[i])
        w_p = wis_from_distribution(dτ, data.τ_true[i])
        wis_raw += w_p
        wis_rel += w_p / max(abs(data.τ_true[i]), 1e-3)
        sum_τ_fam += w_p; n_τ_fam += 1

        dβ = gamma_from_ef(β_post[i])
        w_p = wis_from_distribution(dβ, data.β_true[i])
        wis_raw += w_p
        wis_rel += w_p / max(abs(data.β_true[i]), 1e-3)
        sum_β_fam += w_p; n_β_fam += 1
    end
    return (; wis = wis_raw / nparam,
             wis_w = sum_w_fam / n_w_fam,
             wis_τ = sum_τ_fam / n_τ_fam,
             wis_β = sum_β_fam / n_β_fam,
             wis_rel = wis_rel / nparam,
             total_time_s = elapsed, status = "OK")
end

# ---------------------------------------------------------------------------
# CSV scaffold
# ---------------------------------------------------------------------------
function empty_results()
    return DataFrame(
        method=String[], N=Int[], D=Int[],
        wis=Float64[], wis_w=Float64[], wis_τ=Float64[], wis_β=Float64[],
        wis_rel=Float64[],
        total_time_s=Float64[], status=String[],
    )
end
CSV.write(CSV_PATH, empty_results())
append_row!(row) = CSV.write(CSV_PATH,
    begin df = empty_results(); push!(df, row); df end; append=true)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
println("=" ^ 72)
println("WIS sweep  —  Gibbs and VMP-dynamic over (N, D) grid")
println("α levels: $(ALPHAS)   (K=$(length(ALPHAS)) intervals, ≈ CRPS)")
println("=" ^ 72)

println("\n[warm-up VMP at N=5, D=2]")
_ = vmp_wis(5, 2)

for D in D_VALUES
    println("\n[D = $D]")
    for N in N_VALUES
        print("  N=$N  Gibbs...")
        r = gibbs_wis(N, D)
        println("  WIS=$(round(r.wis, digits=3))  w=$(round(r.wis_w, digits=3))  τ=$(round(r.wis_τ, digits=3))  β=$(round(r.wis_β, digits=3))  t=$(round(r.total_time_s, digits=2))s")
        append_row!(("Gibbs", N, D, r.wis, r.wis_w, r.wis_τ, r.wis_β, r.wis_rel, r.total_time_s, r.status))

        print("  N=$N  VMP-10...")
        r = vmp_wis(N, D)
        if r.status == "OK"
            println("  WIS=$(round(r.wis, digits=3))  w=$(round(r.wis_w, digits=3))  τ=$(round(r.wis_τ, digits=3))  β=$(round(r.wis_β, digits=3))  t=$(round(r.total_time_s, digits=2))s")
        else
            println("  $(r.status)  t=$(round(r.total_time_s, digits=2))s")
        end
        append_row!(("VMP-dynamic", N, D, r.wis, r.wis_w, r.wis_τ, r.wis_β, r.wis_rel, r.total_time_s, r.status))
    end
end

println("\nSaved $CSV_PATH")
df_final = CSV.read(CSV_PATH, DataFrame)
show(stdout, df_final; allrows=true, allcols=true); println()
