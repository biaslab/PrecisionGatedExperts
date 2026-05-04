# VMP scaling — same (N, D) grid as scaling_v2.jl
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

include(joinpath(@__DIR__, "data_generation.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)
const CSV_PATH = joinpath(RESULTS_DIR, "scaling_v2_vmp.csv")
const FE_CSV   = joinpath(RESULTS_DIR, "scaling_v2_vmp_fe.csv")

const N_EXPERTS = 7
const VMP_ITERATIONS = 10   # 10 so FE trace has shape (5 is paper default; verify convergence here)

const N_VALUES = [10, 50, 250]
const D_VALUES = [2, 5, 10]

function empty_results()
    return DataFrame(
        method=String[], N=Int[], D=Int[],
        total_time_s=Float64[], free_energy=Float64[],
        status=String[],
    )
end
function empty_fe()
    return DataFrame(N=Int[], D=Int[], iter=Int[], free_energy=Float64[])
end

CSV.write(CSV_PATH, empty_results())
CSV.write(FE_CSV,   empty_fe())

function append_row!(row)
    df = empty_results()
    push!(df, row)
    CSV.write(CSV_PATH, df; append=true)
end

function append_fe_trace!(N::Int, D::Int, fe::AbstractVector)
    df = empty_fe()
    for (k, v) in enumerate(fe)
        push!(df, (N, D, k, Float64(v)))
    end
    CSV.write(FE_CSV, df; append=true)
end

function run_vmp_dynamic(N::Int, D::Int; seed=42)
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
        n_forecasters = N_EXPERTS, n_obs = N, priors = priors,
    )
    constraints = ProbabilisticEnsembling.univariate_dynamic_ensemble_constraints(priors, false)
    init = ProbabilisticEnsembling.univariate_dynamic_ensemble_init(priors)
    spec = (
        prediction_type = ProbabilisticEnsembling.Univariate(),
        model_type = ProbabilisticEnsembling.Dynamic(),
        inference_iterations = VMP_ITERATIONS,
        subsample_size = nothing,
        subsample_percentage = nothing,
    )
    vmp_data = (y=data.y, features=data.features, predictions=data.predictions)

    t0 = time()
    try
        result = ProbabilisticEnsembling.run_training_rxinfer(
            spec, model, vmp_data;
            constraints = constraints, initialization = init, showprogress = false,
        )
        elapsed = time() - t0
        fe_trace = collect(result.free_energy)
        return (total_time_s = elapsed, free_energy = fe_trace[end],
                fe_trace = fe_trace, status = "OK")
    catch e
        elapsed = time() - t0
        return (total_time_s = elapsed, free_energy = NaN, fe_trace = Float64[],
                status = "FAIL:" * first(split(sprint(showerror, e), '\n')))
    end
end

println("=" ^ 60)
println("VMP scaling (dynamic) — $VMP_ITERATIONS iterations per config")
println("=" ^ 60)

# Warm up RxInfer / DynamicPPL by doing a tiny call first
println("\n[Warm-up: tiny N=5 D=2 run]")
_ = run_vmp_dynamic(5, 2)

for D in D_VALUES
    println("\n[D = $D]")
    for N in N_VALUES
        print("  N=$N  VMP dynamic...")
        r = run_vmp_dynamic(N, D)
        println(" $(r.status)  FE=$(round(r.free_energy, digits=2))  t=$(round(r.total_time_s, digits=2))s")
        append_row!(("VMP-dynamic", N, D, r.total_time_s, r.free_energy, r.status))
        if r.status == "OK" && !isempty(r.fe_trace)
            append_fe_trace!(N, D, r.fe_trace)
        end
    end
end

println("\nSaved $CSV_PATH")
