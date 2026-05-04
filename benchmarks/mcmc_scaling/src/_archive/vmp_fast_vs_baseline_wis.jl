# Posterior-quality comparison between baseline VMP and fast VMP
# (FastManifoldStrategy).  Mirrors vmp_fast_compare_50iter.jl's setup exactly
# (N=250, D=10, 50 iters, seed 42, τ_range (0.5, 5.0)) then extracts
# posteriors and scores them with the Weighted Interval Score per parameter
# family (w, τ, β).

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
const OUT_CSV     = joinpath(RESULTS_DIR, "vmp_fast_vs_baseline_wis.csv")

const N, D, N_EXPERTS, ITERS = 250, 10, 7, 50
const ALPHAS = [0.5, 0.2, 0.1, 0.05]

# ------------------------------------------------------------- WIS helpers --
function wis_from_quantiles(truth, med, lowers, uppers, alphas)
    K = length(alphas)
    s = 0.5 * abs(truth - med)
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
function gamma_from_ef(g)
    α    = shape(g)
    rate = 1.0 / scale(g)
    return Gamma(α, 1.0 / rate)
end

# Per-family WIS given w/τ/β posterior vectors
function score_posteriors(w_post, τ_post, β_post, data)
    sum_w = 0.0; n_w = 0
    sum_τ = 0.0; n_τ = 0
    sum_β = 0.0; n_β = 0
    per_param = Float64[]
    for i in 1:N_EXPERTS, k in 1:D
        w_p = wis_from_distribution(normal_from_mv(w_post[i], k), data.w_true[i][k])
        sum_w += w_p; n_w += 1
        push!(per_param, w_p)
    end
    for i in 1:N_EXPERTS
        w_p = wis_from_distribution(gamma_from_ef(τ_post[i]), data.τ_true[i])
        sum_τ += w_p; n_τ += 1
        push!(per_param, w_p)
        w_p = wis_from_distribution(gamma_from_ef(β_post[i]), data.β_true[i])
        sum_β += w_p; n_β += 1
        push!(per_param, w_p)
    end
    return (; wis = mean(per_param),
             wis_w = sum_w / n_w,
             wis_τ = sum_τ / n_τ,
             wis_β = sum_β / n_β)
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
    subsample_size = nothing,
    subsample_percentage = nothing,
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
fe_A = collect(rA.free_energy)[end]
scoreA = score_posteriors(rA.posteriors[:w][end], rA.posteriors[:τ][end],
                          rA.posteriors[:β][end], data)
@printf("baseline:  time = %.2f s,  FE/N = %.4f,  WIS = %.4f  (w %.4f  τ %.4f  β %.4f)\n",
    tA, fe_A/N, scoreA.wis, scoreA.wis_w, scoreA.wis_τ, scoreA.wis_β)

println("\nRunning fast (B) at $(ITERS) iters...")
tB = @elapsed rB = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model, vmp_data;
    constraints = cB, initialization = init, showprogress = false,
)
fe_B = collect(rB.free_energy)[end]
scoreB = score_posteriors(rB.posteriors[:w][end], rB.posteriors[:τ][end],
                          rB.posteriors[:β][end], data)
@printf("fast:      time = %.2f s,  FE/N = %.4f,  WIS = %.4f  (w %.4f  τ %.4f  β %.4f)\n",
    tB, fe_B/N, scoreB.wis, scoreB.wis_w, scoreB.wis_τ, scoreB.wis_β)

println("\nDelta (fast − baseline):")
@printf("  ΔFE/N     = %+.4f   (negative = fast better-fitting)\n", (fe_B - fe_A)/N)
@printf("  ΔWIS       = %+.4f\n", scoreB.wis    - scoreA.wis)
@printf("  ΔWIS_w     = %+.4f   (negative = fast better)\n", scoreB.wis_w - scoreA.wis_w)
@printf("  ΔWIS_τ     = %+.4f\n", scoreB.wis_τ - scoreA.wis_τ)
@printf("  ΔWIS_β     = %+.4f\n", scoreB.wis_β - scoreA.wis_β)
@printf("  speedup    = %.2fx\n", tA / tB)

CSV.write(OUT_CSV, DataFrame(
    variant      = ["baseline", "fast"],
    total_time_s = [tA, tB],
    fe_per_N     = [fe_A/N, fe_B/N],
    wis          = [scoreA.wis,   scoreB.wis],
    wis_w        = [scoreA.wis_w, scoreB.wis_w],
    wis_τ        = [scoreA.wis_τ, scoreB.wis_τ],
    wis_β        = [scoreA.wis_β, scoreB.wis_β],
))
println("\nSaved $OUT_CSV")
