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

include(joinpath(@__DIR__, "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)
const CSV_PATH = joinpath(RESULTS_DIR, "pareto_quality_vmp.csv")

const N_EXPERTS = 7
const VMP_ITERATIONS = 10
const N_VALUES = [10, 50, 250]
const D_VALUES = [2, 5, 10]

function empty_results()
    return DataFrame(
        method=String[], N=Int[], D=Int[],
        rel_err_total=Float64[], rel_err_w=Float64[], rel_err_τ=Float64[], rel_err_β=Float64[],
        cov95=Float64[], cov50=Float64[],
        std_err_total=Float64[],
        total_time_s=Float64[], free_energy=Float64[], status=String[],
    )
end
CSV.write(CSV_PATH, empty_results())
append_row!(row) = CSV.write(CSV_PATH, begin d = empty_results(); push!(d, row); d end; append=true)

# ---------------------------------------------------------------------------
# Per-parameter Gaussian/Gamma handles, with mean/std/quantile for coverage.
# ---------------------------------------------------------------------------
function mvnormal_component(mv_dist, k::Int)
    μ = mean(mv_dist)[k]
    σ = sqrt(cov(mv_dist)[k, k])
    return Distributions.Normal(μ, σ)
end

function gamma_dist(g)
    α = shape(g)
    β_rate = 1.0 / scale(g)     # scale = 1/rate
    return Distributions.Gamma(α, 1.0 / β_rate)   # Distributions uses (shape, scale)
end

# ---------------------------------------------------------------------------
# Build all per-parameter distributions in the same order as Gibbs:
# w[i,k] first (all experts × all d), then for each expert: τ[i], β[i].
# Also return corresponding truth vector.
# ---------------------------------------------------------------------------
function build_param_dists(w_post, τ_post, β_post, data)
    d = length(data.w_true[1])
    n_experts = length(data.w_true)
    dists = []
    truth = Float64[]
    for i in 1:n_experts, k in 1:d
        push!(dists, mvnormal_component(w_post[i], k))
        push!(truth, data.w_true[i][k])
    end
    for i in 1:n_experts
        push!(dists, gamma_dist(τ_post[i]));  push!(truth, data.τ_true[i])
        push!(dists, gamma_dist(β_post[i]));  push!(truth, data.β_true[i])
    end
    return dists, truth
end

function compute_metrics_vmp(dists, truth)
    n = length(truth)
    post_mean = [Float64(mean(d)) for d in dists]
    post_std  = [Float64(std(d))  for d in dists]
    q025 = [Float64(quantile(d, 0.025)) for d in dists]
    q975 = [Float64(quantile(d, 0.975)) for d in dists]
    q25  = [Float64(quantile(d, 0.25))  for d in dists]
    q75  = [Float64(quantile(d, 0.75))  for d in dists]

    cov95 = mean((truth .>= q025) .& (truth .<= q975))
    cov50 = mean((truth .>= q25)  .& (truth .<= q75))

    rel_err = norm(post_mean .- truth) / max(norm(truth), 1e-12)
    std_err = norm((post_mean .- truth) ./ max.(post_std, 1e-12)) / sqrt(n)

    # Per-family (w, τ, β) rel-errs
    d = length(data_w_dim[])
    return (; rel_err, cov95, cov50, std_err, post_mean, post_std)
end

# Helper to reuse the same block structure — inefficient but clear.
function per_family_errors(post_mean, truth, n_experts, d)
    w_end = n_experts * d
    pm_w = post_mean[1:w_end];            tr_w = truth[1:w_end]
    pm_τ = post_mean[(w_end+1):2:end];    tr_τ = truth[(w_end+1):2:end]
    pm_β = post_mean[(w_end+2):2:end];    tr_β = truth[(w_end+2):2:end]
    erw = norm(pm_w .- tr_w) / max(norm(tr_w), 1e-12)
    erτ = norm(pm_τ .- tr_τ) / max(norm(tr_τ), 1e-12)
    erβ = norm(pm_β .- tr_β) / max(norm(tr_β), 1e-12)
    return (erw, erτ, erβ)
end

const data_w_dim = Ref(0)

function run_vmp_config(N::Int, D::Int; seed=42)
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
        model_type = ProbabilisticEnsembling.Dynamic(),
        inference_iterations = VMP_ITERATIONS,
        subsample_size = nothing, subsample_percentage = nothing,
    )
    vmp_data = (y=data.y, features=data.features, predictions=data.predictions)

    t0 = time()
    try
        result = ProbabilisticEnsembling.run_training_rxinfer(
            spec, model, vmp_data;
            constraints=constraints, initialization=init, showprogress=false,
        )
        elapsed = time() - t0
        w_post = result.posteriors[:w][end]
        τ_post = result.posteriors[:τ][end]
        β_post = result.posteriors[:β][end]

        dists, truth = build_param_dists(w_post, τ_post, β_post, data)
        post_mean = [Float64(mean(d)) for d in dists]
        post_std  = [Float64(std(d))  for d in dists]
        q025 = [Float64(quantile(d, 0.025)) for d in dists]
        q975 = [Float64(quantile(d, 0.975)) for d in dists]
        q25  = [Float64(quantile(d, 0.25))  for d in dists]
        q75  = [Float64(quantile(d, 0.75))  for d in dists]

        cov95 = mean((truth .>= q025) .& (truth .<= q975))
        cov50 = mean((truth .>= q25)  .& (truth .<= q75))
        rel_err = norm(post_mean .- truth) / max(norm(truth), 1e-12)
        std_err = norm((post_mean .- truth) ./ max.(post_std, 1e-12)) / sqrt(length(truth))
        erw, erτ, erβ = per_family_errors(post_mean, truth, N_EXPERTS, D)

        return (; rel_err, rel_err_w=erw, rel_err_τ=erτ, rel_err_β=erβ,
                  cov95, cov50, std_err, total_time_s=elapsed,
                  free_energy=result.free_energy[end], status="OK")
    catch e
        elapsed = time() - t0
        msg = "FAIL:" * first(split(sprint(showerror, e), '\n'))
        return (; rel_err=NaN, rel_err_w=NaN, rel_err_τ=NaN, rel_err_β=NaN,
                  cov95=NaN, cov50=NaN, std_err=NaN, total_time_s=elapsed,
                  free_energy=NaN, status=msg)
    end
end

println("=" ^ 70)
println("Pareto quality (VMP, dynamic, $VMP_ITERATIONS iters): metrics vs time")
println("=" ^ 70)

println("\n[warm-up N=5, D=2]")
_ = run_vmp_config(5, 2)

for D in D_VALUES
    println("\n[D = $D]")
    for N in N_VALUES
        print("  N=$N  VMP dynamic...")
        r = run_vmp_config(N, D)
        if r.status == "OK"
            println(" err=$(round(r.rel_err, digits=3))  cov95=$(round(r.cov95, digits=2))  cov50=$(round(r.cov50, digits=2))  " *
                    "std_err=$(round(r.std_err, digits=2))  t=$(round(r.total_time_s, digits=2))s")
        else
            println(" $(r.status)  t=$(round(r.total_time_s, digits=2))s")
        end
        append_row!(("VMP-dynamic", N, D, r.rel_err, r.rel_err_w, r.rel_err_τ, r.rel_err_β,
                     r.cov95, r.cov50, r.std_err, r.total_time_s, r.free_energy, r.status))
    end
end

println("\nSaved $CSV_PATH")
