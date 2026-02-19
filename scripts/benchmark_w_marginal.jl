#!/usr/bin/env julia

"""
Benchmark to test the hypothesis that the w-marginal computation is the bottleneck
due to repeated dense allocations when summing low-rank precision matrices.

There are two suspected allocation sources:
1. Sequential hcat in LowRankMatrix + LowRankMatrix (O(n_feat × n_obs²) copies)
2. Dense materialization when LowRankMatrix meets a dense/diagonal prior precision

Run:
    julia --project scripts/benchmark_w_marginal.jl
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Statistics
using LinearAlgebra
using LowRankMatrices
using ProbabilisticEnsembling
using Random
using BenchmarkTools
using BayesBase

# ─── Part 1: Benchmark raw LowRankMatrix accumulation ─────────────────────────

function bench_lowrank_accumulation(n_feat::Int, n_obs::Int)
    rng = MersenneTwister(42)
    # Simulate n_obs rank-1 LowRankMatrix precision matrices
    lrs = [let x = randn(rng, n_feat); LowRankMatrix(x, x) end for _ in 1:n_obs]

    # Sequential addition (what foldl does during message product)
    function sequential_add(lrs)
        acc = lrs[1]
        for i in 2:length(lrs)
            acc = acc + lrs[i]
        end
        return acc
    end

    # Warmup
    sequential_add(lrs)

    # Benchmark
    allocs = @allocated sequential_add(lrs)
    t = @elapsed sequential_add(lrs)
    result = sequential_add(lrs)

    return (; time=t, allocs=allocs, result_rank=rank(result),
              U_size=size(result.U), V_size=size(result.V))
end

# ─── Part 2: Benchmark the full message product (MvNormalWeightedMeanPrecision) ─

function bench_message_product_old(n_feat::Int, n_obs::Int)
    rng = MersenneTwister(42)

    # OLD: MvNormalWeightedMeanPrecision with LowRankMatrix precision
    messages = map(1:n_obs) do _
        x = randn(rng, n_feat)
        γ = abs(randn(rng)) + 0.1
        Dθ = γ * LowRankMatrix(x, x)
        zθ = γ * x * randn(rng)
        MvNormalWeightedMeanPrecision(zθ, Dθ)
    end

    function sequential_prod(messages)
        acc = messages[1]
        for i in 2:length(messages)
            acc = prod(PreserveTypeProd(Distribution), acc, messages[i])
        end
        return acc
    end

    sequential_prod(messages)
    allocs = @allocated sequential_prod(messages)
    t = @elapsed sequential_prod(messages)
    result = sequential_prod(messages)
    prec_type = typeof(invcov(result))
    return (; time=t, allocs=allocs, precision_type=prec_type, result=result)
end

function bench_message_product_new(n_feat::Int, n_obs::Int)
    rng = MersenneTwister(42)

    # NEW: LowRankNormalWeightedMeanPrecision
    messages = map(1:n_obs) do _
        x = randn(rng, n_feat)
        γ = abs(randn(rng)) + 0.1
        zθ = γ * x * randn(rng)
        LowRankNormalWeightedMeanPrecision(zθ, x, γ)
    end

    function sequential_prod(messages)
        acc = messages[1]
        for i in 2:length(messages)
            acc = prod(PreserveTypeProd(Distribution), acc, messages[i])
        end
        return acc
    end

    sequential_prod(messages)
    allocs = @allocated sequential_prod(messages)
    t = @elapsed sequential_prod(messages)
    result = sequential_prod(messages)
    prec_type = typeof(invcov(result))
    return (; time=t, allocs=allocs, precision_type=prec_type, result=result)
end

# ─── Part 3: Benchmark what happens when prior (dense) meets accumulated low-rank ─

function bench_prior_meets_lowrank(n_feat::Int, n_obs::Int)
    rng = MersenneTwister(42)

    # Build accumulated LowRankMatrix (simulating n_obs messages already folded)
    xs = [randn(rng, n_feat) for _ in 1:n_obs]
    acc_lr = LowRankMatrix(xs[1], xs[1])
    for i in 2:n_obs
        acc_lr = acc_lr + LowRankMatrix(xs[i], xs[i])
    end

    # Prior precision: dense diagonal-like matrix
    prior_prec = Matrix(0.1 * I, n_feat, n_feat)

    # The addition that triggers dense materialization
    dense_result = @allocated (acc_lr + prior_prec)
    t = @elapsed (acc_lr + prior_prec)

    return (; time=t, allocs=dense_result)
end

# ─── Part 4: Full RxInfer inference benchmark ─────────────────────────────────

@model function dynamic_mv_ensemble_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors)
    local w, z, γ, τ

    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
    end

    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            γ[i, j] ~ GammaShapeScale(1.0, 1.0)
            z[i, j] ~ Log(γ[i, j])
            y[j] ~ MvNormalMeanScalePrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function dynamic_ensemble_constraints()
    q(w, z, γ, τ) = q(w)q(z)q(γ)q(τ)
    q(z)::ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ)::ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

@initialization function dynamic_ensemble_init(w_init)
    q(w) = w_init
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
end

function bench_full_infer(n_feat::Int, n_obs::Int; n_forecasters::Int=2, d::Int=7)
    rng = MersenneTwister(42)
    features = [vcat(1.0, randn(rng, n_feat - 1)) for _ in 1:n_obs]
    predictions = [randn(rng, d) for _ in 1:n_forecasters, _ in 1:n_obs]
    y_train = [randn(rng, d) for _ in 1:n_obs]

    w_priors = [MvNormalMeanScalePrecision(zeros(n_feat), 0.1) for _ in 1:n_forecasters]
    τ_priors = [GammaShapeScale(1.0, 1e12) for _ in 1:n_forecasters]

    function run_infer()
        infer(
            model=dynamic_mv_ensemble_model(
                n_forecasters=n_forecasters,
                n_obs=n_obs,
                w_priors=w_priors,
                τ_priors=τ_priors,
            ),
            data=(y=y_train, features=features, predictions=predictions),
            constraints=dynamic_ensemble_constraints(),
            initialization=dynamic_ensemble_init(w_priors),
            iterations=1,
            free_energy=false,
            showprogress=false,
        )
    end

    # Warmup
    run_infer()

    allocs = @allocated run_infer()
    t = @elapsed run_infer()

    return (; time=t, allocs=allocs)
end

# ─── Part 5: Compare dense-only baseline (no LowRank at all) ─────────────────

function bench_dense_accumulation(n_feat::Int, n_obs::Int)
    rng = MersenneTwister(42)
    # Same as low-rank but using dense matrices
    xs = [randn(rng, n_feat) for _ in 1:n_obs]

    function sequential_dense_add(xs)
        acc = xs[1] * xs[1]'
        for i in 2:length(xs)
            acc = acc + xs[i] * xs[i]'
        end
        return acc
    end

    # Warmup
    sequential_dense_add(xs)

    allocs = @allocated sequential_dense_add(xs)
    t = @elapsed sequential_dense_add(xs)

    return (; time=t, allocs=allocs)
end

# What the optimal approach would be: accumulate as sum of outer products directly
function bench_optimal_accumulation(n_feat::Int, n_obs::Int)
    rng = MersenneTwister(42)
    xs = [randn(rng, n_feat) for _ in 1:n_obs]
    gammas = [abs(randn(rng)) + 0.1 for _ in 1:n_obs]

    function inplace_accumulate(xs, gammas)
        n = length(xs[1])
        Λ = zeros(n, n)
        xi = zeros(n)
        for i in eachindex(xs)
            x = xs[i]
            γ = gammas[i]
            # Λ += γ * x * x' (rank-1 update via BLAS)
            LinearAlgebra.BLAS.ger!(γ, x, x, Λ)
            # xi += γ * x * y_scalar (simplified)
            xi .+= γ .* x
        end
        return Λ, xi
    end

    # Warmup
    inplace_accumulate(xs, gammas)

    allocs = @allocated inplace_accumulate(xs, gammas)
    t = @elapsed inplace_accumulate(xs, gammas)

    return (; time=t, allocs=allocs)
end

# ─── Run all benchmarks ──────────────────────────────────────────────────────

function main()
    println("=" ^ 80)
    println("HYPOTHESIS TEST: Is the w-marginal LowRank accumulation the bottleneck?")
    println("=" ^ 80)

    n_feat = 15  # typical feature dimension (1 + 7 + 7cos + 7sin ≈ 15 for ETTh1 with 7 features)

    println("\n─── Test 1: Raw LowRankMatrix sequential accumulation ───")
    println("(This is what happens inside the message product on w)")
    for n_obs in [10, 50, 100, 200, 500]
        r = bench_lowrank_accumulation(n_feat, n_obs)
        println("  n_obs=$n_obs: time=$(round(r.time*1000, digits=2))ms, " *
                "allocs=$(round(r.allocs/1024, digits=1))KB, " *
                "result_rank=$(r.result_rank), U=$(r.U_size)")
    end

    println("\n─── Test 2: OLD vs NEW message product ───")
    println("(Old: LowRankMatrix hcat, New: LowRankNormalWMP with BLAS ger!)")
    for n_obs in [10, 50, 100, 200, 500]
        r_old = bench_message_product_old(n_feat, n_obs)
        r_new = bench_message_product_new(n_feat, n_obs)
        speedup = r_old.time / max(r_new.time, 1e-9)
        println("  n_obs=$n_obs:")
        println("    OLD: time=$(round(r_old.time*1000, digits=2))ms, allocs=$(round(r_old.allocs/1024, digits=1))KB, type=$(r_old.precision_type)")
        println("    NEW: time=$(round(r_new.time*1000, digits=2))ms, allocs=$(round(r_new.allocs/1024, digits=1))KB, type=$(r_new.precision_type)")
        println("    Speedup: $(round(speedup, digits=1))x, Alloc reduction: $(round(r_old.allocs / max(r_new.allocs, 1), digits=1))x")
    end

    println("\n─── Test 3: Correctness check (old vs new precision matrices) ───")
    for n_obs in [10, 50, 200]
        r_old = bench_message_product_old(n_feat, n_obs)
        r_new = bench_message_product_new(n_feat, n_obs)
        Λ_old = Matrix(invcov(r_old.result))
        Λ_new = Matrix(invcov(r_new.result))
        xi_old = weightedmean(r_old.result)
        xi_new = weightedmean(r_new.result)
        prec_err = maximum(abs.(Λ_old .- Λ_new))
        xi_err = maximum(abs.(xi_old .- xi_new))
        println("  n_obs=$n_obs: max|Λ_old - Λ_new| = $(prec_err), max|ξ_old - ξ_new| = $(xi_err)")
        if prec_err > 1e-10 || xi_err > 1e-10
            println("    ⚠ MISMATCH DETECTED")
        else
            println("    ✓ Numerically identical")
        end
    end

    println("\n─── Test 4: Comparison of accumulation strategies ───")
    println("(Old hcat vs New ger! vs Optimal in-place)")
    for n_obs in [50, 200, 500]
        r_old = bench_message_product_old(n_feat, n_obs)
        r_new = bench_message_product_new(n_feat, n_obs)
        r_opt = bench_optimal_accumulation(n_feat, n_obs)
        println("  n_obs=$n_obs:")
        println("    Old hcat:     time=$(round(r_old.time*1000, digits=2))ms, allocs=$(round(r_old.allocs/1024, digits=1))KB")
        println("    New ger!:     time=$(round(r_new.time*1000, digits=2))ms, allocs=$(round(r_new.allocs/1024, digits=1))KB")
        println("    BLAS optimal: time=$(round(r_opt.time*1000, digits=2))ms, allocs=$(round(r_opt.allocs/1024, digits=1))KB")
    end

    println("\n─── Test 5: Full RxInfer inference (includes all message passing) ───")
    println("(Scaling with n_obs while n_feat=$n_feat, n_forecasters=2)")
    for n_obs in [10, 50, 100, 200]
        r = bench_full_infer(n_feat, n_obs; n_forecasters=2, d=7)
        println("  n_obs=$n_obs: time=$(round(r.time, digits=3))s, " *
                "allocs=$(round(r.allocs/1024/1024, digits=1))MB")
    end
end

main()
