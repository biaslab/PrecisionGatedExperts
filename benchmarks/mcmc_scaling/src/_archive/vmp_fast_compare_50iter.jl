# vmp_fast_compare_50iter.jl
#
# Side-by-side comparison at N=250, D=10, 50 VMP iters:
#   A — baseline constraints (ClosedFormStrategy on product manifold)
#   B — fast constraints     (FastManifoldStrategy(ClosedFormStrategy()) on
#                             both q(z) — NormalMeanVariance — and q(γ) — Gamma)
#
# Both variants are warmed (3-iter run) before timing. Two alternating
# rounds so thermal/scheduling bias between A and B is balanced. Final
# FE/N is the per-observation free energy at the last iteration; the full
# per-iter trace is written as CSV for plotting convergence.

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamily
using StableRNGs
using Statistics
using Printf
using CSV
using DataFrames

include(joinpath(@__DIR__, "data_generation.jl"))

const N, D, N_EXPERTS, ITERS = 250, 10, 7, 50

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

# Warmup to finish compilation before timing
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations = 3),
    model, vmp_data; constraints = cA, initialization = init, showprogress = false,
)
ProbabilisticEnsembling.run_training_rxinfer(
    (; spec..., inference_iterations = 3),
    model, vmp_data; constraints = cB, initialization = init, showprogress = false,
)

ts_A = Float64[]; ts_B = Float64[]
fe_A_full = Vector{Float64}[]; fe_B_full = Vector{Float64}[]
for round in 1:2
    t = @elapsed rA = ProbabilisticEnsembling.run_training_rxinfer(
        spec, model, vmp_data;
        constraints = cA, initialization = init, showprogress = false,
    )
    push!(ts_A, t); push!(fe_A_full, collect(rA.free_energy) ./ N)
    @printf("round %d  A baseline %.2fs  (%.1f ms/iter)  final FE/N=%.4f\n",
        round, t, t/ITERS*1e3, fe_A_full[end][end])

    t = @elapsed rB = ProbabilisticEnsembling.run_training_rxinfer(
        spec, model, vmp_data;
        constraints = cB, initialization = init, showprogress = false,
    )
    push!(ts_B, t); push!(fe_B_full, collect(rB.free_energy) ./ N)
    @printf("round %d  B fast     %.2fs  (%.1f ms/iter)  final FE/N=%.4f\n",
        round, t, t/ITERS*1e3, fe_B_full[end][end])
end

@printf("\nmedian A: %.2fs (%.1f ms/iter)  final FE/N=%.6f\n",
    median(ts_A), median(ts_A)/ITERS*1e3, fe_A_full[end][end])
@printf("median B: %.2fs (%.1f ms/iter)  final FE/N=%.6f\n",
    median(ts_B), median(ts_B)/ITERS*1e3, fe_B_full[end][end])
@printf("speedup : %.2fx\n", median(ts_A) / median(ts_B))
@printf("|ΔFE/N at iter %d|: %.3e\n", ITERS,
    abs(fe_A_full[end][end] - fe_B_full[end][end]))

# Save per-iter FE trace for plotting
df = DataFrame(
    iter        = 1:ITERS,
    fe_baseline = fe_A_full[end],
    fe_fast     = fe_B_full[end],
)
out = joinpath(@__DIR__, "..", "results", "vmp_fast_both_n250_d10_$(ITERS)iter.csv")
mkpath(dirname(out))
CSV.write(out, df)
println("\nsaved trace: ", out)
