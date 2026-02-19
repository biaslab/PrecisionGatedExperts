#!/usr/bin/env julia

"""
Quick benchmark: one RxInfer iteration on electricity dataset.
Tests the in-place DenseNormalWMP accumulation vs old LowRankMatrix hcat.
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Statistics
using LinearAlgebra
using ProbabilisticEnsembling
using Random

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

function run_bench(dataset_path::String; n_obs::Union{Int,Nothing}=nothing, n_forecasters::Int=4, seq_len::Int=96, horizon::Int=96)
    @info "Loading dataset" path=dataset_path
    Xmat, _ = load_ett(dataset_path)
    X3, Y2 = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)
    Xtr, _, Xval, Yval, _, _ = train_val_test_split(X3, Y2)

    d = size(Yval, 1)
    scaler = fit_scaler(Xtr)
    Xval_s = scale_inputs(scaler, Xval)
    Yval_sc = scale_targets(scaler, Yval)

    n_obs_available = size(Xval, 3)
    n_obs_actual = something(n_obs, n_obs_available)
    n_obs_actual = min(n_obs_actual, n_obs_available)

    rng = Random.MersenneTwister(42)
    features = [let x = Float64.(Xval_s[:, end, j]); vcat(1.0, x, map(cos, x), map(sin, x)) end for j in 1:n_obs_actual]
    n_feat = length(features[1])

    predictions = [randn(rng, d) for _ in 1:n_forecasters, _ in 1:n_obs_actual]
    y_train = [Vector{Float64}(Yval_sc[:, j]) for j in 1:n_obs_actual]

    w_priors = [MvNormalMeanScalePrecision(zeros(n_feat), 0.1) for _ in 1:n_forecasters]
    τ_priors = [GammaShapeScale(1.0, 1e12) for _ in 1:n_forecasters]

    @info "Configuration" n_obs=n_obs_actual n_forecasters n_feat d

    function run_infer()
        infer(
            model=dynamic_mv_ensemble_model(
                n_forecasters=n_forecasters,
                n_obs=n_obs_actual,
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

    # Warmup with small n_obs
    n_warm = min(3, n_obs_actual)
    feat_w = features[1:n_warm]
    pred_w = predictions[:, 1:n_warm]
    y_w = y_train[1:n_warm]
    wp_w = [MvNormalMeanScalePrecision(zeros(n_feat), 0.1) for _ in 1:n_forecasters]
    tp_w = [GammaShapeScale(1.0, 1e12) for _ in 1:n_forecasters]
    @info "Warmup..."
    infer(
        model=dynamic_mv_ensemble_model(n_forecasters=n_forecasters, n_obs=n_warm, w_priors=wp_w, τ_priors=tp_w),
        data=(y=y_w, features=feat_w, predictions=pred_w),
        constraints=dynamic_ensemble_constraints(),
        initialization=dynamic_ensemble_init(wp_w),
        iterations=1, free_energy=false, showprogress=false,
    )

    @info "Running benchmark..."
    GC.gc()
    allocs = @allocated run_infer()
    GC.gc()
    t = @elapsed run_infer()

    @info "Results" time_s=round(t, digits=3) allocs_MB=round(allocs/1024/1024, digits=1)
    return (; time=t, allocs=allocs)
end

# Run on electricity
println("=" ^ 70)
println("Electricity dataset benchmark - 1 iteration")
println("=" ^ 70)

for n_obs in [50, 200, 500]
    r = run_bench("data/electricity.csv"; n_obs=n_obs, n_forecasters=2)
    println("  n_obs=$n_obs: time=$(round(r.time, digits=3))s, allocs=$(round(r.allocs/1024/1024, digits=1))MB")
end
