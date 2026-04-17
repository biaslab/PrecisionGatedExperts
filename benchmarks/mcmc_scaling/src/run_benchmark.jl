using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using StableRNGs
using Statistics
using DataFrames
using CSV

include("data_generation.jl")
include("turing_model.jl")
include("gibbs_model.jl")

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

const TIMEOUT_SECONDS = 30 * 60  # 30 minutes
const N_VALUES = [10, 25, 50, 100, 250]
const N_EXPERTS = 7
const D = 65

# Sample counts: enough to measure per-iteration cost and basic ESS,
# but feasible within timeout. The argument rests on time-per-iteration
# scaling, not on converged posteriors at large N.
const NUTS_WARMUP = 50
const NUTS_SAMPLES = 100
const GIBBS_WARMUP = 20
const GIBBS_SAMPLES = 50

"""
Run sampling with a timeout. Returns (chain, elapsed_seconds) or (nothing, elapsed) on timeout.
"""
function run_with_timeout(model, sampler, n_samples; warmup=0, timeout=TIMEOUT_SECONDS)
    result = Ref{Any}(nothing)
    elapsed = Ref(0.0)

    task = @task begin
        t0 = time()
        try
            chain = sample(model, sampler, n_samples; discard_initial=warmup, progress=true)
            elapsed[] = time() - t0
            result[] = chain
        catch e
            elapsed[] = time() - t0
            @warn "Sampling failed" exception = (e, catch_backtrace())
            result[] = nothing
        end
    end

    schedule(task)

    t_start = time()
    while !istaskdone(task)
        if (time() - t_start) > timeout
            @warn "Timeout after $(timeout)s"
            return (nothing, timeout)
        end
        sleep(1.0)
    end

    return (result[], elapsed[])
end

"""
Extract ESS and R-hat diagnostics from a chain.
"""
function chain_diagnostics(chain)
    if chain === nothing
        return (; min_ess=NaN, max_rhat=NaN)
    end

    ess_vals = ess_rhat(chain)
    ess_col = ess_vals[:, :ess]
    rhat_col = ess_vals[:, :rhat]

    return (;
        min_ess = minimum(skipmissing(ess_col)),
        max_rhat = maximum(skipmissing(rhat_col)),
    )
end

# ============================================================
# Pre-compilation: throwaway run with N=2 (excludes JIT)
# ============================================================
println("=" ^ 60)
println("Pre-compiling with N=2...")
println("=" ^ 60)

precomp_data = generate_synthetic_data(N=2, n_experts=N_EXPERTS, d=D, rng=StableRNG(0))

print("  NUTS...")
precomp_m = pge_ensemble(precomp_data.y, precomp_data.features, precomp_data.predictions, N_EXPERTS, D)
try
    sample(precomp_m, NUTS(0.8), 3; discard_initial=3, progress=false)
    println(" done.")
catch e
    println(" failed: $e")
end

print("  Gibbs...")
precomp_mg = pge_ensemble_gibbs(precomp_data.y, precomp_data.features, precomp_data.predictions, N_EXPERTS, D)
try
    sample(precomp_mg, build_gibbs_sampler(), 3; progress=false)
    println(" done.")
catch e
    println(" failed: $e")
end

println()

# ============================================================
# Main scaling experiment
# ============================================================
results = DataFrame(
    sampler=String[], N=Int[], latent_dims=Int[],
    total_iters=Int[], time_s=Float64[], time_per_iter_s=Float64[],
    min_ess=Float64[], max_rhat=Float64[], ess_per_s=Float64[],
    status=String[],
)

for N in N_VALUES
    latent_dims = N_EXPERTS * N + N_EXPERTS * D + 2 * N_EXPERTS
    println("=" ^ 60)
    println("N = $N | Latent dims = $latent_dims")
    println("=" ^ 60)

    data = generate_synthetic_data(N=N, n_experts=N_EXPERTS, d=D, rng=StableRNG(42))

    # --- NUTS ---
    println("\n  [NUTS] $(NUTS_WARMUP) warmup + $(NUTS_SAMPLES) samples...")
    model_nuts = pge_ensemble(data.y, data.features, data.predictions, N_EXPERTS, D)
    total_nuts = NUTS_WARMUP + NUTS_SAMPLES
    chain_nuts, time_nuts = run_with_timeout(
        model_nuts, NUTS(0.8), NUTS_SAMPLES;
        warmup=NUTS_WARMUP, timeout=TIMEOUT_SECONDS,
    )

    if chain_nuts !== nothing
        diag = chain_diagnostics(chain_nuts)
        t_per_iter = time_nuts / total_nuts
        ess_s = diag.min_ess / time_nuts
        # Extrapolate to 1000+1000 for the table
        extrapolated = t_per_iter * 2000

        println("  [NUTS] $(round(time_nuts, digits=1))s total | $(round(t_per_iter, digits=2))s/iter | ESS/s=$(round(ess_s, digits=4))")
        println("  [NUTS] Extrapolated 1000+1000: $(round(extrapolated, digits=0))s ($(round(extrapolated/60, digits=1)) min)")
        push!(results, ("NUTS", N, latent_dims, total_nuts, time_nuts, t_per_iter, diag.min_ess, diag.max_rhat, ess_s, "OK"))
    else
        println("  [NUTS] TIMEOUT after $(TIMEOUT_SECONDS)s")
        push!(results, ("NUTS", N, latent_dims, total_nuts, Float64(TIMEOUT_SECONDS), NaN, NaN, NaN, NaN, "TIMEOUT"))
    end

    # --- Gibbs ---
    println("\n  [Gibbs] $(GIBBS_WARMUP) warmup + $(GIBBS_SAMPLES) sweeps...")
    model_gibbs = pge_ensemble_gibbs(data.y, data.features, data.predictions, N_EXPERTS, D)
    total_gibbs = GIBBS_WARMUP + GIBBS_SAMPLES
    chain_gibbs, time_gibbs = run_with_timeout(
        model_gibbs, build_gibbs_sampler(), GIBBS_SAMPLES;
        warmup=GIBBS_WARMUP, timeout=TIMEOUT_SECONDS,
    )

    if chain_gibbs !== nothing
        diag = chain_diagnostics(chain_gibbs)
        t_per_sweep = time_gibbs / total_gibbs
        ess_s = diag.min_ess / time_gibbs
        extrapolated = t_per_sweep * 700  # 200 warmup + 500 sampling

        println("  [Gibbs] $(round(time_gibbs, digits=1))s total | $(round(t_per_sweep, digits=2))s/sweep | ESS/s=$(round(ess_s, digits=4))")
        println("  [Gibbs] Extrapolated 200+500: $(round(extrapolated, digits=0))s ($(round(extrapolated/60, digits=1)) min)")
        push!(results, ("Gibbs", N, latent_dims, total_gibbs, time_gibbs, t_per_sweep, diag.min_ess, diag.max_rhat, ess_s, "OK"))
    else
        println("  [Gibbs] TIMEOUT after $(TIMEOUT_SECONDS)s")
        push!(results, ("Gibbs", N, latent_dims, total_gibbs, Float64(TIMEOUT_SECONDS), NaN, NaN, NaN, NaN, "TIMEOUT"))
    end

    println()
end

# ============================================================
# Save results
# ============================================================
CSV.write(joinpath(RESULTS_DIR, "scaling_results.csv"), results)

println("=" ^ 60)
println("Results saved to $(joinpath(RESULTS_DIR, "scaling_results.csv"))")
println("=" ^ 60)
println()
show(stdout, results; allrows=true, allcols=true)
println()
