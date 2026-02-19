#!/usr/bin/env julia

"""
Targeted timing breakdown: which part of one VMP iteration is slowest?

Usage:
    julia --project scripts/profile_bottleneck.jl
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using ClosedFormExpectations: LogGamma
using Distributions
using Statistics
using LinearAlgebra
using ProbabilisticEnsembling
using Random
using BenchmarkTools

# ─── Isolated cost of each operation ──────────────────────────────────────────

function bench_softdot_messages(n_feat, d; n_samples=1000)
    rng = MersenneTwister(42)
    mx = randn(rng, n_feat)
    mγ = 1.5
    my = 0.3
    mθ = randn(rng, n_feat)
    Vθ = 0.1 * I(n_feat) |> Matrix
    Vx_diag = ones(n_feat)

    # softdot :θ message (LowRankNormalWMP)
    t_θ = @belapsed LowRankNormalWeightedMeanPrecision($mγ * $mx * $my, $mx, $mγ)

    # softdot :γ message (GammaShapeRate) — needs mean_cov of q_θ
    function softdot_gamma_msg()
        # simplified from ReactiveMP softdot :γ rule
        β = (my^2) / 2
        β -= my * dot(mθ, mx)
        β += (sum(Vx_diag .* diag(Vθ)) + dot(mθ, Vx_diag .* mθ) + dot(mx, diag(Vθ) .* mx) + dot(mθ, mx)^2) / 2
        return (1.5, β)
    end
    t_γ = @belapsed $softdot_gamma_msg()

    return (softdot_θ=t_θ, softdot_γ=t_γ)
end

function bench_log_marginal_projection(; n_samples=100)
    # Simulate what happens at each Log node marginal computation
    # m_out: Normal message on z (from softdot side)
    # m_in: Gamma message on γ (from GammaShapeScale prior side)
    m_out_mean = 0.5
    m_out_std = 1.0
    m_in = GammaShapeScale(1.0, 1.0)

    function log_marginal_project()
        σ = max(m_out_std, sqrt(eps(Float64)))
        log_normal = LogNormal(m_out_mean, σ)
        prj = ProjectedTo(Gamma; parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            niterations = 50,
            tolerance = 1e-6,
            stepsize = ExponentialFamilyProjection.Manopt.ArmijoLinesearch(
                initial_stepsize = 1e-2,
                stop_increasing_at_step = 0
            ),
            direction = ExponentialFamilyProjection.BoundedNormUpdateRule(0.5)
        ))
        supplementary = convert(Gamma, m_in)
        α = max(shape(supplementary), sqrt(eps(Float64)))
        θ = max(scale(supplementary), sqrt(eps(Float64)))
        initial = Gamma(α, max(0.1 * θ, sqrt(eps(Float64))))
        return project_to(prj, log_normal, supplementary; initialpoint = initial)
    end

    # Warmup
    log_marginal_project()

    t = @belapsed $log_marginal_project()
    allocs = @allocated log_marginal_project()
    return (time=t, allocs=allocs)
end

function bench_z_projection(; n_samples=100)
    # q(z) projection: ProjectedTo(NormalMeanVariance) with ClosedFormStrategy
    # This is the marginal of z given messages from softdot and Log
    m_softdot = NormalMeanPrecision(0.5, 2.0)  # message from softdot towards z
    m_log = LogGamma(1.0, 1.0)  # message from Log towards z (LogGamma)

    function z_project()
        prj = ProjectedTo(NormalMeanVariance; parameters = ProjectionParameters(
            strategy = ClosedFormStrategy()
        ))
        return project_to(prj, m_log, m_softdot)
    end

    # Warmup
    z_project()

    t = @belapsed $z_project()
    allocs = @allocated z_project()
    return (time=t, allocs=allocs)
end

function bench_mvnormal_gamma_message(d)
    # MvNormalMeanScalePrecision :γ rule
    # Computes Gamma message from y observation towards γ
    m_out = randn(d)
    m_μ = randn(d)
    v_out = 0.1 * ones(d)
    v_μ = 0.1 * ones(d)

    function gamma_from_mvnormal()
        α = d / 2 + 1
        β = (sum(v_μ) + sum(v_out) + sum((m_μ .- m_out).^2)) / 2
        return (α, β)
    end

    t = @belapsed $gamma_from_mvnormal()
    return (time=t,)
end

function bench_ger_accumulation(n_feat, n_obs)
    rng = MersenneTwister(42)
    xs = [randn(rng, n_feat) for _ in 1:n_obs]
    gammas = [abs(randn(rng)) + 0.1 for _ in 1:n_obs]

    function accumulate()
        Λ = zeros(n_feat, n_feat)
        xi = zeros(n_feat)
        for i in 1:n_obs
            BLAS.ger!(gammas[i], xs[i], xs[i], Λ)
            xi .+= gammas[i] .* xs[i]
        end
        return Λ, xi
    end

    accumulate()
    t = @belapsed $accumulate()
    allocs = @allocated accumulate()
    return (time=t, allocs=allocs)
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    println("=" ^ 80)
    println("BOTTLENECK ANALYSIS: Per-node cost breakdown")
    println("=" ^ 80)

    n_feat = 964   # electricity features
    d = 321        # electricity output dim
    n_obs = 5222   # electricity train obs
    n_forecasters = 4

    println("\n─── 1. Log marginal projection (project_to Gamma, 50 iters) ───")
    r_log = bench_log_marginal_projection()
    println("  Single call: $(round(r_log.time * 1e6, digits=1)) μs, $(round(r_log.allocs/1024, digits=1)) KB")
    total_log = r_log.time * n_obs * n_forecasters
    println("  Per VMP iteration ($(n_obs) × $(n_forecasters) = $(n_obs * n_forecasters) calls): $(round(total_log, digits=1)) s")

    println("\n─── 2. z-marginal projection (project_to NormalMeanVariance) ───")
    r_z = bench_z_projection()
    println("  Single call: $(round(r_z.time * 1e6, digits=1)) μs, $(round(r_z.allocs/1024, digits=1)) KB")
    total_z = r_z.time * n_obs * n_forecasters
    println("  Per VMP iteration ($(n_obs * n_forecasters) calls): $(round(total_z, digits=1)) s")

    println("\n─── 3. Softdot message costs ───")
    r_sd = bench_softdot_messages(n_feat, d)
    println("  softdot :θ (LR msg): $(round(r_sd.softdot_θ * 1e9, digits=1)) ns")
    println("  softdot :γ (Gamma msg): $(round(r_sd.softdot_γ * 1e9, digits=1)) ns")
    total_sd = (r_sd.softdot_θ + r_sd.softdot_γ) * n_obs * n_forecasters
    println("  Per VMP iteration: $(round(total_sd * 1000, digits=1)) ms")

    println("\n─── 4. MvNormalMeanScalePrecision :γ message (d=$d) ───")
    r_mv = bench_mvnormal_gamma_message(d)
    total_mv = r_mv.time * n_obs * n_forecasters
    println("  Single call: $(round(r_mv.time * 1e9, digits=1)) ns")
    println("  Per VMP iteration: $(round(total_mv * 1000, digits=1)) ms")

    println("\n─── 5. w-marginal accumulation (n_feat=$n_feat, n_obs=$n_obs) ───")
    r_w = bench_ger_accumulation(n_feat, n_obs)
    total_w = r_w.time * n_forecasters
    println("  Single w[i] accumulation: $(round(r_w.time * 1000, digits=1)) ms, $(round(r_w.allocs/1024/1024, digits=1)) MB")
    println("  All $(n_forecasters) w variables: $(round(total_w * 1000, digits=1)) ms")

    println("\n" * "=" ^ 80)
    println("SUMMARY: Estimated time per VMP iteration")
    println("=" ^ 80)
    items = [
        ("Log marginal projection", total_log),
        ("z-marginal projection", total_z),
        ("Softdot messages", total_sd),
        ("MvNormal :γ messages", total_mv),
        ("w-marginal accumulation", total_w),
    ]
    total = sum(x[2] for x in items)
    for (name, t) in sort(items, by=x -> -x[2])
        pct = round(100 * t / total, digits=1)
        println("  $(rpad(name, 30)) $(lpad(round(t, digits=2), 8)) s  ($pct%)")
    end
    println("  $(rpad("TOTAL (estimated)", 30)) $(lpad(round(total, digits=2), 8)) s")
    println("\n  With 100 iterations: $(round(total * 100 / 60, digits=1)) minutes")
end

main()
