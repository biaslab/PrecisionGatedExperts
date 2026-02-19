#!/usr/bin/env julia

"""
Profile a single RxInfer inference iteration to understand where time is spent.

Usage (from the Julia REPL):
    julia> ARGS = ["data/ETTh1.csv"]
    julia> include("scripts/profile_inference.jl")
    # Then use the helpers:
    julia> @profview profile_infer()                      # flame graph
    julia> @profview profile_infer(free_energy=true)      # with free energy

    # Change settings and re-profile:
    julia> setup_profile("data/ETTh1.csv"; n_features=5)  # low features
    julia> @profview profile_infer()

    julia> setup_profile("data/ETTh1.csv"; n_features=100) # high features
    julia> @profview profile_infer()
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Statistics
using LinearAlgebra
using ProbabilisticEnsembling
using Profile
using ProfileView
using Random

# -----------------------------------------------------------------------------
# RxInfer model (same as in dynamic_neural_ensemble_rxinfer.jl)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Feature helpers
# -----------------------------------------------------------------------------

function make_features_from_data(X_scaled::AbstractArray{<:Real,3})
    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        x_last = Float64.(X_scaled[:, end, j])
        feats[j] = vcat(1.0, x_last, map(cos, x_last), map(sin, x_last))
    end
    return feats
end

function make_synthetic_features(n_obs::Int, n_features::Int; rng=Random.default_rng())
    return [vcat(1.0, randn(rng, n_features - 1)) for _ in 1:n_obs]
end

function make_synthetic_predictions(n_forecasters::Int, d::Int, n_obs::Int; rng=Random.default_rng())
    preds = Array{Vector{Float64}}(undef, n_forecasters, n_obs)
    for i in 1:n_forecasters
        for j in 1:n_obs
            preds[i, j] = randn(rng, d)
        end
    end
    return preds
end

# -----------------------------------------------------------------------------
# Global state for interactive profiling
# -----------------------------------------------------------------------------

const PROFILE_STATE = Ref{NamedTuple}()

"""
    setup_profile(dataset_path; n_obs=nothing, n_forecasters=4, n_features=nothing, seq_len=96, horizon=96)

Load data, build synthetic predictions, and warmup. Call once, then use `profile_infer()`.
"""
function setup_profile(dataset_path::String;
    n_obs::Union{Int,Nothing}=nothing,
    n_forecasters::Int=4,
    n_features::Union{Int,Nothing}=nothing,
    seq_len::Int=96,
    horizon::Int=96,
)
    @info "Loading dataset" path=dataset_path seq_len horizon

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

    natural_features = make_features_from_data(Xval_s)
    rng = Random.MersenneTwister(42)

    if n_features === nothing
        n_feat = length(natural_features[1])
        features = natural_features[1:n_obs_actual]
    else
        n_feat = n_features
        features = make_synthetic_features(n_obs_actual, n_feat; rng=rng)
    end

    predictions = make_synthetic_predictions(n_forecasters, d, n_obs_actual; rng=rng)
    y_train = [Vector{Float64}(Yval_sc[:, j]) for j in 1:n_obs_actual]

    w_priors = [MvNormalMeanScalePrecision(zeros(n_feat), 0.1) for _ in 1:n_forecasters]
    τ_priors = [GammaShapeScale(1.0, 1e12) for _ in 1:n_forecasters]

    kwargs = (
        n_forecasters=n_forecasters, n_obs=n_obs_actual, n_features=n_feat,
        d=d, features=features, predictions=predictions,
        y_train=y_train, w_priors=w_priors, τ_priors=τ_priors,
    )

    @info "Configuration" n_obs=n_obs_actual n_forecasters n_features=n_feat output_dim=d

    # Warmup
    @info "Warmup ..."
    warmup_kwargs = (
        n_forecasters=n_forecasters, n_obs=min(2, n_obs_actual), n_features=n_feat,
        d=d, features=features[1:min(2, n_obs_actual)], predictions=predictions[:, 1:min(2, n_obs_actual)],
        y_train=y_train[1:min(2, n_obs_actual)],
        w_priors=[MvNormalMeanScalePrecision(zeros(n_feat), 0.1) for _ in 1:n_forecasters],
        τ_priors=[GammaShapeScale(1.0, 1e12) for _ in 1:n_forecasters],
    )
    _run_infer(; warmup_kwargs..., free_energy=false)
    @info "Warmup done. Now use: @profview profile_infer()"

    PROFILE_STATE[] = kwargs
    return nothing
end

function _run_infer(;
    n_forecasters, n_obs, n_features, d, features, predictions, y_train, w_priors, τ_priors,
    free_energy=false,
)
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
        free_energy=free_energy,
        showprogress=false,
    )
    return nothing
end

"""
    profile_infer(; free_energy=false)

Run one inference iteration. Use with `@profview`:

    @profview profile_infer()
    @profview profile_infer(free_energy=true)
"""
function profile_infer(; free_energy::Bool=false)
    if !isassigned(PROFILE_STATE)
        error("Call setup_profile(\"data/ETTh1.csv\") first")
    end
    _run_infer(; PROFILE_STATE[]..., free_energy=free_energy)
end

# -----------------------------------------------------------------------------
# Auto-setup when included with ARGS
# -----------------------------------------------------------------------------

function _parse_args(args)
    ds = nothing
    n_obs = nothing
    n_forecasters = 4
    n_features = nothing
    seq_len = 96
    horizon = 96

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--n-obs"
            n_obs = parse(Int, args[i+1]); i += 2
        elseif a == "--n-forecasters"
            n_forecasters = parse(Int, args[i+1]); i += 2
        elseif a == "--n-features"
            n_features = parse(Int, args[i+1]); i += 2
        elseif a == "--seq-len"
            seq_len = parse(Int, args[i+1]); i += 2
        elseif a == "--horizon"
            horizon = parse(Int, args[i+1]); i += 2
        elseif !startswith(a, "-")
            ds = a; i += 1
        else
            error("Unknown argument: $a")
        end
    end

    return ds, n_obs, n_forecasters, n_features, seq_len, horizon
end

# Run setup automatically if ARGS has a dataset path
if !isempty(ARGS)
    ds, n_obs, n_forecasters, n_features, seq_len, horizon = _parse_args(ARGS)
    if ds !== nothing
        setup_profile(ds; n_obs, n_forecasters, n_features, seq_len, horizon)
    end
end
