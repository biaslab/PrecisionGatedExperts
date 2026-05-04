# 100-iter run of baseline vs fast VMP at N=250, D=10.
# Saves full FE trace + final WIS per family + convergence diagnostic.

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamily
using StableRNGs
using Distributions
using Statistics
using LinearAlgebra
using Printf
using DataFrames, CSV

include(joinpath(@__DIR__, "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const TRACE_CSV   = joinpath(RESULTS_DIR, "vmp_fast_both_n250_d10_100iter.csv")
const WIS_CSV     = joinpath(RESULTS_DIR, "vmp_fast_vs_baseline_wis_100iter.csv")

const N, D, N_EXPERTS, ITERS = 250, 10, 7, 100
const ALPHAS = [0.5, 0.2, 0.1, 0.05]

# ------------------------------------------------------------- WIS helpers --
function wis_from_quantiles(truth, med, lowers, uppers, alphas)
    K = length(alphas); s = 0.5 * abs(truth - med)
    for k in 1:K
        α = alphas[k]; L = lowers[k]; U = uppers[k]
        s += (α/2)*(U - L) + max(L - truth, 0.0) + max(truth - U, 0.0)
    end
    return s / (K + 0.5)
end
function wis_from_distribution(dist, truth::Real; alphas=ALPHAS)
    med    = quantile(dist, 0.5)
    lowers = [quantile(dist, α/2)     for α in alphas]
    uppers = [quantile(dist, 1 - α/2) for α in alphas]
    return wis_from_quantiles(truth, med, lowers, uppers, alphas)
end
normal_from_mv(mv, k::Int) = Normal(mean(mv)[k], sqrt(cov(mv)[k, k]))
gamma_from_ef(g) = Gamma(shape(g), 1.0 / (1.0 / scale(g)))

function score_posteriors(w_post, τ_post, β_post, data)
    sum_w=0.0; n_w=0; sum_τ=0.0; n_τ=0; sum_β=0.0; n_β=0; per = Float64[]
    for i in 1:N_EXPERTS, k in 1:D
        v = wis_from_distribution(normal_from_mv(w_post[i], k), data.w_true[i][k])
        sum_w += v; n_w += 1; push!(per, v)
    end
    for i in 1:N_EXPERTS
        v = wis_from_distribution(gamma_from_ef(τ_post[i]), data.τ_true[i])
        sum_τ += v; n_τ += 1; push!(per, v)
        v = wis_from_distribution(gamma_from_ef(β_post[i]), data.β_true[i])
        sum_β += v; n_β += 1; push!(per, v)
    end
    return (; wis = mean(per), wis_w = sum_w/n_w, wis_τ = sum_τ/n_τ, wis_β = sum_β/n_β)
end

# Convergence diagnostic: drift of FE/N over last `win` iterations, relative.
function tail_drift(v::AbstractVector, win::Int = 10)
    tail = v[max(end - win + 1, 1):end]
    return (maximum(tail) - minimum(tail)) / max(abs(tail[end]), 1e-12)
end

# --------------------------------------------------------------- model run --
data = generate_synthetic_data(
    N = N, n_experts = N_EXPERTS, d = D,
    rng = StableRNG(42), τ_range = (0.5, 5.0),
)
priors = Dict{Symbol,Any}(
    :w => [MvNormalMeanScalePrecision(zeros(D), 0.01) for _ in 1:N_EXPERTS],
    :τ => [GammaShapeRate(1.0, 1.0)                   for _ in 1:N_EXPERTS],
    :β => [GammaShapeRate(1.0, 1e3)                   for _ in 1:N_EXPERTS],
)
model = ProbabilisticEnsembling.univariate_dynamic_ensemble(
    n_forecasters = N_EXPERTS, n_obs = N, priors = priors,
)
init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type      = ProbabilisticEnsembling.Dynamic(),
    inference_iterations = ITERS,
    subsample_size = nothing, subsample_percentage = nothing,
)
vmp_data = (y = data.y, features = data.features, predictions = data.predictions)

cA = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
cB = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints_fast(priors, false)

println("Warm-up (3 iters each)...")
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations = 3),
    model, vmp_data; constraints = cA, initialization = init, showprogress = false,
)
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations = 3),
    model, vmp_data; constraints = cB, initialization = init, showprogress = false,
)

println("\nRunning baseline (A) at $(ITERS) iters...")
tA = @elapsed rA = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model, vmp_data;
    constraints = cA, initialization = init, showprogress = false,
)
fe_A = collect(rA.free_energy) ./ N
scoreA = score_posteriors(rA.posteriors[:w][end], rA.posteriors[:τ][end],
                          rA.posteriors[:β][end], data)

println("\nRunning fast (B) at $(ITERS) iters...")
tB = @elapsed rB = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model, vmp_data;
    constraints = cB, initialization = init, showprogress = false,
)
fe_B = collect(rB.free_energy) ./ N
scoreB = score_posteriors(rB.posteriors[:w][end], rB.posteriors[:τ][end],
                          rB.posteriors[:β][end], data)

drift_A_last10 = tail_drift(fe_A, 10)
drift_A_last20 = tail_drift(fe_A, 20)
drift_B_last10 = tail_drift(fe_B, 10)
drift_B_last20 = tail_drift(fe_B, 20)

println("\n=====  50-iter summary vs 100-iter summary  =====")
@printf("baseline FE/N @ iter  50: %.4f    @ iter 100: %.4f    Δ = %+.4f\n",
    fe_A[min(50, length(fe_A))], fe_A[end], fe_A[end] - fe_A[min(50, length(fe_A))])
@printf("fast     FE/N @ iter  50: %.4f    @ iter 100: %.4f    Δ = %+.4f\n",
    fe_B[min(50, length(fe_B))], fe_B[end], fe_B[end] - fe_B[min(50, length(fe_B))])

println("\n=====  Final (iter $(ITERS)) results  =====")
@printf("baseline:  time = %.1f s,  FE/N = %.4f,  WIS = %.4f  (w %.4f  τ %.4f  β %.4f)\n",
    tA, fe_A[end], scoreA.wis, scoreA.wis_w, scoreA.wis_τ, scoreA.wis_β)
@printf("fast:      time = %.1f s,  FE/N = %.4f,  WIS = %.4f  (w %.4f  τ %.4f  β %.4f)\n",
    tB, fe_B[end], scoreB.wis, scoreB.wis_w, scoreB.wis_τ, scoreB.wis_β)

println("\n=====  Convergence (drift over last N iters, relative %)  =====")
@printf("baseline:  last 10 = %.3f %%,  last 20 = %.3f %%\n",
    100*drift_A_last10, 100*drift_A_last20)
@printf("fast:      last 10 = %.3f %%,  last 20 = %.3f %%\n",
    100*drift_B_last10, 100*drift_B_last20)
verdict(d) = d < 0.001 ? "converged"        :
             d < 0.005 ? "settled"          :
             d < 0.02  ? "slowly moving"    :
                          "still descending"
println("  baseline → $(verdict(drift_A_last10));  fast → $(verdict(drift_B_last10))")

println("\n=====  Δ (fast − baseline)  =====")
@printf("  ΔFE/N = %+.4f   Δtime = %+.1f s  (%.2fx speedup)\n",
    fe_B[end] - fe_A[end], tB - tA, tA / tB)
@printf("  ΔWIS_w = %+.4f   ΔWIS_τ = %+.4f   ΔWIS_β = %+.4f\n",
    scoreB.wis_w - scoreA.wis_w, scoreB.wis_τ - scoreA.wis_τ,
    scoreB.wis_β - scoreA.wis_β)

CSV.write(TRACE_CSV, DataFrame(
    iter        = 1:ITERS,
    fe_baseline = fe_A,
    fe_fast     = fe_B,
))
CSV.write(WIS_CSV, DataFrame(
    variant           = ["baseline", "fast"],
    total_time_s      = [tA, tB],
    fe_per_N_final    = [fe_A[end], fe_B[end]],
    drift_last10_pct  = [100*drift_A_last10, 100*drift_B_last10],
    drift_last20_pct  = [100*drift_A_last20, 100*drift_B_last20],
    wis               = [scoreA.wis,   scoreB.wis],
    wis_w             = [scoreA.wis_w, scoreB.wis_w],
    wis_τ             = [scoreA.wis_τ, scoreB.wis_τ],
    wis_β             = [scoreA.wis_β, scoreB.wis_β],
))
println("\nSaved $TRACE_CSV\nSaved $WIS_CSV")
