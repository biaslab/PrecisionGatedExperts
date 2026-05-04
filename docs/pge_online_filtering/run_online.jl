# Online (Kalman-like) VMP on the PGE dynamic ensemble.
# Each batch uses the previous batch's posterior as its prior.
#
# Run from repo root:
#   julia --project=. docs/pge_online_filtering/run_online.jl

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamily
using Distributions
using StableRNGs
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "benchmarks", "mcmc_scaling", "src", "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

# ============================================================================
# Config
# ============================================================================
const N_EXPERTS     = 7
const D             = 2            # start small (where the marginals were clean)
const BATCH_SIZE    = 10
const N_BATCHES     = 10           # 10 batches × 10 obs = N_total = 100
const VMP_ITERS     = 20           # per-batch budget

const N_TOTAL = N_BATCHES * BATCH_SIZE

# Initial (broad-ish) priors — match the d=2 Gibbs/NUTS setup.
function initial_priors(D)
    return Dict{Symbol, Any}(
        :w => [MvNormalMeanScalePrecision(zeros(D), 0.01) for _ in 1:N_EXPERTS],
        :τ => [GammaShapeRate(1.0, 1.0)                   for _ in 1:N_EXPERTS],
        :β => [GammaShapeRate(1.0, 1e3)                   for _ in 1:N_EXPERTS],
    )
end

# ============================================================================
# Helper: run one VMP pass, return posterior + FE trace + time
# ============================================================================
function run_one_batch(y_b, features_b, predictions_b, priors; iters=VMP_ITERS)
    model = ProbabilisticEnsembling.univariate_dynamic_ensemble(
        n_forecasters = N_EXPERTS, n_obs = length(y_b), priors = priors,
    )
    constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
    init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
    spec = (
        prediction_type = ProbabilisticEnsembling.Univariate(),
        model_type = ProbabilisticEnsembling.Dynamic(),
        inference_iterations = iters,
        subsample_size = nothing, subsample_percentage = nothing,
    )
    data_tuple = (y=y_b, features=features_b, predictions=predictions_b)
    t0 = time()
    result = ProbabilisticEnsembling.run_training_rxinfer(
        spec, model, data_tuple;
        constraints=constraints, initialization=init, showprogress=false,
    )
    elapsed = time() - t0
    return (;
        w_post = result.posteriors[:w][end],
        τ_post = result.posteriors[:τ][end],
        β_post = result.posteriors[:β][end],
        fe_trace = collect(result.free_energy),
        elapsed = elapsed,
    )
end

"""Turn per-expert posteriors back into a priors-shaped Dict for the next batch."""
function posterior_to_priors(w_post, τ_post, β_post)
    return Dict{Symbol, Any}(
        :w => [w_post[i] for i in 1:N_EXPERTS],
        :τ => [τ_post[i] for i in 1:N_EXPERTS],
        :β => [β_post[i] for i in 1:N_EXPERTS],
    )
end

"""Posterior-mean error vs truth for global params."""
function error_vs_truth(w_post, τ_post, β_post, data)
    truth = Float64[]
    post_mean = Float64[]
    for i in 1:N_EXPERTS, k in 1:D
        push!(truth, data.w_true[i][k])
        push!(post_mean, mean(w_post[i])[k])
    end
    for i in 1:N_EXPERTS
        push!(truth, data.τ_true[i]); push!(post_mean, mean(τ_post[i]))
        push!(truth, data.β_true[i]); push!(post_mean, mean(β_post[i]))
    end
    return norm(post_mean .- truth) / max(norm(truth), 1e-12)
end

# ============================================================================
# Generate data once, shared across online + full-batch runs
# ============================================================================
println("=" ^ 70)
println("PGE online VMP filtering — $N_BATCHES batches × $BATCH_SIZE obs  (N_total=$N_TOTAL, D=$D)")
println("=" ^ 70)

data = generate_synthetic_data(
    N = N_TOTAL, n_experts = N_EXPERTS, d = D, rng = StableRNG(42),
    τ_range = (0.5, 5.0),
)

# ============================================================================
# Online (Kalman-like) sweep
# ============================================================================
println("\n--- Online filtering ---")
rows = DataFrame(
    batch=Int[], t_obs=Int[],
    fe_first=Float64[], fe_last=Float64[], fe_per_N=Float64[],
    elapsed=Float64[], cumulative_time=Float64[],
    rel_err_vs_truth=Float64[],
)

priors = initial_priors(D)
cum_time = 0.0
# Warm-up: tiny throwaway run to prime RxInfer compilation (doesn't count)
print("  Warm-up...")
_ = run_one_batch([data.y[1]], [data.features[1]], data.predictions[:, 1:1], priors; iters=2)
println(" ok")

for b in 1:N_BATCHES
    # Slice the b-th batch
    lo = (b - 1) * BATCH_SIZE + 1
    hi = b * BATCH_SIZE
    y_b           = data.y[lo:hi]
    features_b    = data.features[lo:hi]
    predictions_b = data.predictions[:, lo:hi]

    out = run_one_batch(y_b, features_b, predictions_b, priors)
    global cum_time += out.elapsed

    rel_err = error_vs_truth(out.w_post, out.τ_post, out.β_post, data)

    push!(rows, (
        b, hi,
        out.fe_trace[1], out.fe_trace[end], out.fe_trace[end] / BATCH_SIZE,
        out.elapsed, cum_time,
        rel_err,
    ))
    println("  batch $b (obs 1..$hi) : " *
            "FE/B=$(round(out.fe_trace[end]/BATCH_SIZE, digits=2))  " *
            "rel-err=$(round(rel_err, digits=3))  " *
            "t=$(round(out.elapsed, digits=2))s  cum=$(round(cum_time, digits=1))s")

    # Pass the posterior as next batch's prior
    global priors = posterior_to_priors(out.w_post, out.τ_post, out.β_post)
end

CSV.write(joinpath(RESULTS_DIR, "online_trajectory.csv"), rows)

# ============================================================================
# Full-batch baseline
# ============================================================================
println("\n--- Full-batch baseline ($N_TOTAL obs at once, $VMP_ITERS iters) ---")
full_out = run_one_batch(
    data.y, data.features, data.predictions, initial_priors(D); iters=VMP_ITERS,
)
full_err = error_vs_truth(full_out.w_post, full_out.τ_post, full_out.β_post, data)
println("  full FE/N=$(round(full_out.fe_trace[end] / N_TOTAL, digits=2))  " *
        "rel-err=$(round(full_err, digits=3))  t=$(round(full_out.elapsed, digits=2))s")

# ============================================================================
# Figure
# ============================================================================
fig = Figure(size=(1500, 700), fontsize=14)
Label(fig[0, 1:3],
      "Online VMP filtering  (D=$D, N_total=$N_TOTAL, batch=$BATCH_SIZE)",
      fontsize=16, font=:bold)

# FE/B per batch
ax1 = Axis(fig[1, 1];
           xlabel = "batch", ylabel = "FE / batch_size",
           title = "Per-batch FE (lower = better local fit)")
lines!(ax1, rows.batch, rows.fe_per_N; color=:darkorange, linewidth=2.5)
scatter!(ax1, rows.batch, rows.fe_per_N; color=:darkorange, markersize=10)
hlines!(ax1, [full_out.fe_trace[end] / N_TOTAL];
        color=:crimson, linestyle=:dash, linewidth=1.5, label="full-batch FE/N")
axislegend(ax1; position=:rt)

# rel-err vs truth
ax2 = Axis(fig[1, 2];
           xlabel = "batch", ylabel = "rel-L2 err vs truth",
           title = "Posterior-mean error (lower = better)")
lines!(ax2, rows.batch, rows.rel_err_vs_truth; color=:seagreen, linewidth=2.5)
scatter!(ax2, rows.batch, rows.rel_err_vs_truth; color=:seagreen, markersize=10)
hlines!(ax2, [full_err]; color=:crimson, linestyle=:dash, linewidth=1.5,
        label="full-batch err")
axislegend(ax2; position=:rt)

# Wall-clock cumulative
ax3 = Axis(fig[1, 3];
           xlabel = "batch", ylabel = "cumulative time (s)",
           title = "Cumulative wall-clock")
lines!(ax3, rows.batch, rows.cumulative_time; color=:steelblue, linewidth=2.5)
scatter!(ax3, rows.batch, rows.cumulative_time; color=:steelblue, markersize=10)
hlines!(ax3, [full_out.elapsed]; color=:crimson, linestyle=:dash, linewidth=1.5,
        label="full-batch time")
axislegend(ax3; position=:rb)

save(joinpath(RESULTS_DIR, "online_trajectory.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "online_trajectory.pdf"), fig)
println("\nSaved $(joinpath(RESULTS_DIR, "online_trajectory.{png,pdf}"))")

println("\nSummary:")
println(rpad("batch", 8), rpad("FE/B", 12), rpad("rel-err", 12), rpad("t (s)", 10))
for r in eachrow(rows)
    println(rpad(r.batch, 8),
            rpad(round(r.fe_per_N, digits=2), 12),
            rpad(round(r.rel_err_vs_truth, digits=3), 12),
            rpad(round(r.elapsed, digits=2), 10))
end
println("\nFull-batch: FE/N=$(round(full_out.fe_trace[end] / N_TOTAL, digits=2))  " *
        "err=$(round(full_err, digits=3))  t=$(round(full_out.elapsed, digits=2))s")
