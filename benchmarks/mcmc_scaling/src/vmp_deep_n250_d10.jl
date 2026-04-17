# Deep VMP run at N=250, D=10 to see whether FE truly plateaus or keeps dropping.
# Runs from MAIN env (RxInfer / ProbabilisticEnsembling).

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
using CairoMakie

include(joinpath(@__DIR__, "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

const N = 250
const D = 10
const N_EXPERTS = 7
const ITERS = 100

println("=" ^ 70)
println("Deep VMP investigation: N=$N, D=$D, $ITERS iterations")
println("=" ^ 70)

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
constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
spec = (
    prediction_type = ProbabilisticEnsembling.Univariate(),
    model_type = ProbabilisticEnsembling.Dynamic(),
    inference_iterations = ITERS,
    subsample_size = nothing,
    subsample_percentage = nothing,
)
vmp_data = (y=data.y, features=data.features, predictions=data.predictions)

println("\nRunning $ITERS VMP iterations...")
t0 = time()
result = ProbabilisticEnsembling.run_training_rxinfer(
    spec, model, vmp_data;
    constraints = constraints, initialization = init, showprogress = true,
)
elapsed = time() - t0
println("\nDone in $(round(elapsed, digits=1))s  ($(round(elapsed/ITERS*1000, digits=1))ms / iter)")

fe = collect(result.free_energy)
fe_n = fe ./ N

# Save FE trace
CSV.write(joinpath(RESULTS_DIR, "vmp_deep_n250_d10_fe.csv"),
          DataFrame(iter=1:ITERS, free_energy=fe, fe_per_N=fe_n))

# Posterior summary at final iter
w_post = result.posteriors[:w][end]
τ_post = result.posteriors[:τ][end]
β_post = result.posteriors[:β][end]

println("\nPosterior means (first 3 experts) vs truth:")
for i in 1:min(3, N_EXPERTS)
    wi = mean(w_post[i])
    τi = mean(τ_post[i])
    βi = mean(β_post[i])
    println("  Expert $i:")
    println("    truth: w[1]=$(round(data.w_true[i][1], digits=3))  w[2]=$(round(data.w_true[i][2], digits=3))  τ=$(round(data.τ_true[i], digits=3))  β=$(round(data.β_true[i], digits=4))")
    println("    VMP  : w[1]=$(round(wi[1], digits=3))  w[2]=$(round(wi[2], digits=3))  τ=$(round(τi, digits=3))  β=$(round(βi, digits=4))")
end

# Save posterior summaries
post_df = DataFrame(
    expert = Int[],
    param  = String[],
    truth  = Float64[],
    vmp_mean = Float64[],
    vmp_var  = Float64[],
)
for i in 1:N_EXPERTS
    wi_mean = mean(w_post[i])
    wi_var = diag(cov(w_post[i]))
    for k in 1:D
        push!(post_df, (i, "w[$k]", data.w_true[i][k], wi_mean[k], wi_var[k]))
    end
    push!(post_df, (i, "τ", data.τ_true[i], mean(τ_post[i]), var(τ_post[i])))
    push!(post_df, (i, "β", data.β_true[i], mean(β_post[i]), var(β_post[i])))
end
CSV.write(joinpath(RESULTS_DIR, "vmp_deep_n250_d10_posterior.csv"), post_df)

# ======================================================
# Plot FE trace
# ======================================================
fig = Figure(size=(1400, 600), fontsize=13)

ax1 = Axis(fig[1, 1];
           xlabel = "VMP iteration", ylabel = "FE / N",
           title = "Free energy per observation — $ITERS iterations")
lines!(ax1, 1:ITERS, fe_n; color=:darkorange, linewidth=2)
scatter!(ax1, 1:ITERS, fe_n; color=:darkorange, markersize=5)

ax2 = Axis(fig[1, 2];
           xlabel = "VMP iteration", ylabel = "|ΔFE/N| per step (log)",
           title = "Per-iter step size — decrease rate",
           yscale = log10)
dfe = abs.(diff(fe_n))
# Avoid log(0)
dfe = max.(dfe, 1e-10)
lines!(ax2, 2:ITERS, dfe; color=:crimson, linewidth=2)
scatter!(ax2, 2:ITERS, dfe; color=:crimson, markersize=5)

Label(fig[0, 1:2],
      "VMP deep run at N=$N, D=$D, $N_EXPERTS experts  —  is it converging slowly or collapsing?",
      fontsize=15, font=:bold)

save(joinpath(RESULTS_DIR, "vmp_deep_n250_d10.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "vmp_deep_n250_d10.pdf"), fig)
println("\nSaved vmp_deep_n250_d10.{png,pdf}")

# Convergence verdict
last_10 = fe_n[end-9:end]
rel_change = (maximum(last_10) - minimum(last_10)) / max(abs(fe_n[end]), 1.0)
println("\n" * "=" ^ 70)
println("Verdict")
println("=" ^ 70)
println("  Final FE/N:             $(round(fe_n[end], digits=3))")
println("  Δ in last 10 iters:     $(round(100*rel_change, digits=2))%")
println("  Stable (<1%):           $(rel_change < 0.01 ? "YES" : "NO")")
println("  Total wall-time:        $(round(elapsed, digits=1))s")
