# Compare VMP posteriors on dynamic (original, w/τ/β) vs dynamic_exp
# (reparametrized, w/τ, γ=exp(z)) models at N=250, D=10, 100 iters.
#
# Hypothesis: the DynamicExp model removes β as a free precision prior and
# replaces it with γ = exp(z), tying the per-obs precision directly to z.
# This should *reduce* the mean-field over-smoothing on τ that we observed
# on the original model (where β's broad prior contaminated the τ posterior).
#
# Metric: WIS per parameter family  (w, τ only — γ is a per-obs latent, not
# a global parameter).
#
# Output CSV: results/vmp_exp_vs_original_wis.csv

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
using CairoMakie

include(joinpath(@__DIR__, "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const OUT_CSV     = joinpath(RESULTS_DIR, "vmp_exp_vs_original_wis.csv")
const TRACE_CSV   = joinpath(RESULTS_DIR, "vmp_exp_vs_original_trace.csv")

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

function score_w_τ(w_post, τ_post, data)
    sum_w = 0.0; n_w = 0; sum_τ = 0.0; n_τ = 0
    for i in 1:N_EXPERTS, k in 1:D
        sum_w += wis_from_distribution(normal_from_mv(w_post[i], k), data.w_true[i][k])
        n_w += 1
    end
    for i in 1:N_EXPERTS
        sum_τ += wis_from_distribution(gamma_from_ef(τ_post[i]), data.τ_true[i])
        n_τ += 1
    end
    wis_w = sum_w / n_w
    wis_τ = sum_τ / n_τ
    return (; wis_w, wis_τ, wis = (sum_w + sum_τ) / (n_w + n_τ))
end

# -------------------------------------------------------------- generate --
data = generate_synthetic_data(
    N = N, n_experts = N_EXPERTS, d = D,
    rng = StableRNG(42), τ_range = (0.5, 5.0),
)

# Priors — same w/τ across both, β only present for original model.
priors_common = Dict{Symbol,Any}(
    :w => [MvNormalMeanScalePrecision(zeros(D), 0.01) for _ in 1:N_EXPERTS],
    :τ => [GammaShapeRate(1.0, 1.0)                   for _ in 1:N_EXPERTS],
)
priors_original = merge(priors_common, Dict(
    :β => [GammaShapeRate(1.0, 1e3) for _ in 1:N_EXPERTS],
))
priors_exp = priors_common

# -------------------------------- ORIGINAL  (dynamic) -------------------
model_orig = ProbabilisticEnsembling.univariate_dynamic_ensemble(
    n_forecasters = N_EXPERTS, n_obs = N, priors = priors_original,
)
init_orig   = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors_original)
cA_orig     = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors_original, false)
cB_orig     = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints_fast(priors_original, false)

# -------------------------------- EXP       (dynamic_exp) ----------------
model_exp = ProbabilisticEnsembling.univariate_dynamic_exp_ensemble(
    n_forecasters = N_EXPERTS, n_obs = N, priors = priors_exp,
)
init_exp  = ProbabilisticEnsembling.univariate_dynamic_exp_ensemble_init(priors_exp)
c_exp     = ProbabilisticEnsembling.univariate_dynamic_exp_ensemble_constraints(priors_exp, false)

spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type      = ProbabilisticEnsembling.Dynamic(),
    inference_iterations = ITERS,
    subsample_size = nothing, subsample_percentage = nothing,
)
spec_exp = (; spec..., model_type = ProbabilisticEnsembling.DynamicExp())

vmp_data = (y = data.y, features = data.features, predictions = data.predictions)

# Warm-up
println("Warm-up (3 iters each)...")
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations=3), model_orig, vmp_data;
    constraints = cA_orig, initialization = init_orig, showprogress = false)
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations=3), model_orig, vmp_data;
    constraints = cB_orig, initialization = init_orig, showprogress = false)
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec_exp..., inference_iterations=3), model_exp, vmp_data;
    constraints = c_exp, initialization = init_exp, showprogress = false)

# ORIGINAL baseline
println("\nOriginal-baseline ($(ITERS) iters)...")
tA = @elapsed rA = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model_orig, vmp_data;
    constraints = cA_orig, initialization = init_orig, showprogress = false)

# ORIGINAL fast
println("Original-fast ($(ITERS) iters)...")
tB = @elapsed rB = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model_orig, vmp_data;
    constraints = cB_orig, initialization = init_orig, showprogress = false)

# EXP
println("DynamicExp ($(ITERS) iters)...")
tC = @elapsed rC = ProbabilisticEnsembling.run_training_rxinfer(
    spec_exp, model_exp, vmp_data;
    constraints = c_exp, initialization = init_exp, showprogress = false)

# -------------------------------- Score per-iteration --------------------
function per_iter_scores(r, iters=ITERS)
    wis_w = fill(NaN, iters); wis_τ = fill(NaN, iters)
    for k in 1:iters
        s = score_w_τ(r.posteriors[:w][k], r.posteriors[:τ][k], data)
        wis_w[k] = s.wis_w; wis_τ[k] = s.wis_τ
    end
    return wis_w, wis_τ
end
fe_A = collect(rA.free_energy) ./ N; wis_w_A, wis_τ_A = per_iter_scores(rA)
fe_B = collect(rB.free_energy) ./ N; wis_w_B, wis_τ_B = per_iter_scores(rB)
fe_C = collect(rC.free_energy) ./ N; wis_w_C, wis_τ_C = per_iter_scores(rC)

sA = score_w_τ(rA.posteriors[:w][end], rA.posteriors[:τ][end], data)
sB = score_w_τ(rB.posteriors[:w][end], rB.posteriors[:τ][end], data)
sC = score_w_τ(rC.posteriors[:w][end], rC.posteriors[:τ][end], data)

println("\n=====  Final (iter $(ITERS))  =====")
@printf("original-baseline:  time=%.1fs  FE/N=%.4f  WIS_w=%.4f  WIS_τ=%.4f\n",
        tA, fe_A[end], sA.wis_w, sA.wis_τ)
@printf("original-fast:      time=%.1fs  FE/N=%.4f  WIS_w=%.4f  WIS_τ=%.4f\n",
        tB, fe_B[end], sB.wis_w, sB.wis_τ)
@printf("dynamic_exp:        time=%.1fs  FE/N=%.4f  WIS_w=%.4f  WIS_τ=%.4f\n",
        tC, fe_C[end], sC.wis_w, sC.wis_τ)

CSV.write(OUT_CSV, DataFrame(
    variant          = ["original-baseline", "original-fast", "dynamic_exp"],
    total_time_s     = [tA, tB, tC],
    fe_per_N_final   = [fe_A[end], fe_B[end], fe_C[end]],
    wis_w            = [sA.wis_w, sB.wis_w, sC.wis_w],
    wis_τ            = [sA.wis_τ, sB.wis_τ, sC.wis_τ],
))
println("\nSaved $OUT_CSV")

CSV.write(TRACE_CSV, DataFrame(
    iter       = 1:ITERS,
    fe_A       = fe_A,       fe_B       = fe_B,       fe_C       = fe_C,
    wis_w_A    = wis_w_A,    wis_w_B    = wis_w_B,    wis_w_C    = wis_w_C,
    wis_τ_A    = wis_τ_A,    wis_τ_B    = wis_τ_B,    wis_τ_C    = wis_τ_C,
))
println("Saved $TRACE_CSV")

# ------------------------------------------------------- figure: 3 panels --
fig = Figure(size = (1800, 650), fontsize = 14)
Label(fig[0, 1:3],
      "Original (baseline / fast)  vs  DynamicExp  —  FE/N and WIS over iters, N=250, D=10",
      fontsize=16, font=:bold)

function panel!(gp, title, yA, yB, yC, ylab)
    ax = Axis(gp; title=title, xlabel="iter", ylabel=ylab)
    lines!(ax, 1:ITERS, yA; color=:steelblue,  linewidth=2.5, label="original-baseline")
    lines!(ax, 1:ITERS, yB; color=:darkorange, linewidth=2.5, label="original-fast")
    lines!(ax, 1:ITERS, yC; color=:forestgreen,linewidth=2.8, label="dynamic_exp")
    scatter!(ax, [ITERS], [yA[end]]; color=:steelblue,   markersize=9)
    scatter!(ax, [ITERS], [yB[end]]; color=:darkorange,  markersize=9, marker=:diamond)
    scatter!(ax, [ITERS], [yC[end]]; color=:forestgreen, markersize=9, marker=:utriangle)
    axislegend(ax; position=:rt, labelsize=10)
    return ax
end

panel!(fig[1, 1], "FE / N",  fe_A,  fe_B,  fe_C,  "FE / N")
panel!(fig[1, 2], "WIS_w",   wis_w_A, wis_w_B, wis_w_C, "WIS_w")
panel!(fig[1, 3], "WIS_τ",   wis_τ_A, wis_τ_B, wis_τ_C, "WIS_τ")

save(joinpath(RESULTS_DIR, "vmp_exp_vs_original_trace.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "vmp_exp_vs_original_trace.pdf"), fig)
println("Saved vmp_exp_vs_original_trace.{png,pdf}")
