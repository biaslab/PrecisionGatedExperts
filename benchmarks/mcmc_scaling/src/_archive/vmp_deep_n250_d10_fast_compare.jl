# vmp_deep_n250_d10_fast_compare.jl
# Re-runs the Deep VMP N=250, D=10 benchmark twice:
#   (A) baseline — stock `univariate_dynamic_ensemble_constraints`
#       (ClosedFormStrategy on the product manifold)
#   (B) fast    — `univariate_dynamic_ensemble_constraints_fast`
#       (FastManifoldStrategy wrapping ClosedFormStrategy — flat-Vector
#        Gamma manifold + log-Euclidean retraction)
#
# Reports FE/N convergence and wall-time per iter for both.

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamily
using StableRNGs
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Printf

include(joinpath(@__DIR__, "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

const N = 250
const D = 10
const N_EXPERTS = 7
const ITERS = 100

println("=" ^ 72)
println("VMP N=$N, D=$D, $ITERS iters — baseline vs FastManifoldStrategy on γ")
println("=" ^ 72)

# ---- shared setup --------------------------------------------------------

data = generate_synthetic_data(
    N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(42),
    τ_range=(0.5, 5.0),
)

priors = Dict{Symbol, Any}(
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
    model_type = ProbabilisticEnsembling.Dynamic(),
    inference_iterations = ITERS,
    subsample_size = nothing,
    subsample_percentage = nothing,
)
vmp_data = (y=data.y, features=data.features, predictions=data.predictions)

# ---- (A) baseline --------------------------------------------------------

constraints_A = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
println("\n[A] baseline: ClosedFormStrategy on product manifold")
# Warm-up iteration (compilation + cache setup)
result_warm = ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations = 3),
    model, vmp_data; constraints = constraints_A, initialization = init,
    showprogress = false,
)
t_A = @elapsed result_A = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model, vmp_data;
    constraints = constraints_A, initialization = init, showprogress = false,
)
fe_A = collect(result_A.free_energy) ./ N
@printf("    total=%.2fs   per-iter=%.2fms   final FE/N=%.4f\n",
    t_A, t_A / ITERS * 1e3, fe_A[end])

# ---- (B) fast ------------------------------------------------------------

constraints_B = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints_fast(priors, false)
println("\n[B] fast: FastManifoldStrategy(ClosedFormStrategy)")
result_warm = ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations = 3),
    model, vmp_data; constraints = constraints_B, initialization = init,
    showprogress = false,
)
t_B = @elapsed result_B = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model, vmp_data;
    constraints = constraints_B, initialization = init, showprogress = false,
)
fe_B = collect(result_B.free_energy) ./ N
@printf("    total=%.2fs   per-iter=%.2fms   final FE/N=%.4f\n",
    t_B, t_B / ITERS * 1e3, fe_B[end])

# ---- summary -------------------------------------------------------------

speedup = t_A / t_B
fe_diff = abs(fe_A[end] - fe_B[end])

println("\n" * "=" ^ 72)
@printf("  baseline wall-time: %.2fs  (%.2f ms / iter)\n", t_A, t_A/ITERS*1e3)
@printf("  fast wall-time    : %.2fs  (%.2f ms / iter)\n", t_B, t_B/ITERS*1e3)
@printf("  speedup           : %.2fx\n", speedup)
@printf("  FE/N baseline final:  %.6f\n", fe_A[end])
@printf("  FE/N fast    final:  %.6f\n", fe_B[end])
@printf("  |ΔFE/N|           : %.2e\n", fe_diff)
println("=" ^ 72)

# ---- save CSV ------------------------------------------------------------

df = DataFrame(iter = 1:ITERS, fe_baseline = fe_A, fe_fast = fe_B)
out_path = joinpath(RESULTS_DIR, "vmp_deep_n250_d10_fast_compare.csv")
CSV.write(out_path, df)

summary_df = DataFrame(
    variant = ["baseline", "fast"],
    wall_s  = [t_A, t_B],
    per_iter_ms = [t_A/ITERS*1e3, t_B/ITERS*1e3],
    final_fe_per_n = [fe_A[end], fe_B[end]],
)
CSV.write(joinpath(RESULTS_DIR, "vmp_deep_n250_d10_fast_compare_summary.csv"), summary_df)
println("\nsaved: $out_path")
