#!/usr/bin/env julia

"""
Dynamic Probabilistic Ensemble (RxInfer) over neural forecasters saved as JLD2.

Usage:
    julia scripts/dynamic_neural_ensemble_rxinfer.jl <model1.jld2> <model2.jld2> [more...]
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Statistics
using LinearAlgebra
using JLD2
using Lux
using Reactant
using ProbabilisticEnsembling
using Plots

# -----------------------------------------------------------------------------
# Model definitions (mirror training script)
# -----------------------------------------------------------------------------

struct TimeSeriesLSTM{L,H} <: Lux.AbstractLuxContainerLayer{(:lstm_cell, :head)}
    lstm_cell::L
    head::H
end

function TimeSeriesLSTM(in_dims::Int, hidden_dims::Int, out_dims::Int)
    return TimeSeriesLSTM(
        LSTMCell(in_dims => hidden_dims),
        Chain(Dense(hidden_dims => hidden_dims, relu), Dense(hidden_dims => out_dims))
    )
end

function (m::TimeSeriesLSTM)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
    (y, carry), st_lstm = m.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    for x_t in x_rest
        (y, carry), st_lstm = m.lstm_cell((x_t, carry), ps.lstm_cell, st_lstm)
    end
    y, st_head = m.head(y, ps.head, st.head)
    st = merge(st, (lstm_cell=st_lstm, head=st_head))
    return y, st
end

struct TimeSeriesCNN{C1,C2,H} <: Lux.AbstractLuxContainerLayer{(:conv1, :conv2, :head)}
    conv1::C1
    conv2::C2
    head::H
end

function TimeSeriesCNN(in_dims::Int, out_dims::Int; channels::Int=64, k::Int=7, stride::Int=2)
    return TimeSeriesCNN(
        Conv((k,), in_dims => channels, relu; pad=(1,), stride=(stride,)),
        Conv((k,), channels => channels, relu; pad=(1,), stride=(stride,)),
        Chain(Dense(channels => channels, relu), Dense(channels => out_dims))
    )
end

function (m::TimeSeriesCNN)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x = permutedims(x, (2, 1, 3))
    x, st1 = m.conv1(x, ps.conv1, st.conv1)
    x, st2 = m.conv2(x, ps.conv2, st.conv2)
    x = mean(x; dims=1)                   # (1, C, N)
    x = reshape(x, size(x, 2), size(x, 3)) # (C, N)
    y, st_head = m.head(x, ps.head, st.head)
    st = merge(st, (conv1=st1, conv2=st2, head=st_head))
    return y, st
end

# -----------------------------------------------------------------------------
# RxInfer dynamic model (multivariate outputs)
# -----------------------------------------------------------------------------

@model function dynamic_mv_ensemble_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors)
    local w, z, γ, τ


    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
    end

    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])

            γ[i, j] ~ GammaShapeScale(1.0, 1.0)
            z[i, j] ~ Log(γ[i, j])

            y[j] ~ MvNormalMeanScalePrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function dynamic_ensemble_constraints()
    q(w, z, γ, τ) = q(w)q(z)q(γ)q(τ)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ) :: ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

#const N_FEATURES = Ref(0)

@initialization function dynamic_ensemble_init(n_features)
    q(w) = MvNormalMeanScalePrecision(zeros(n_features), 0.1)
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

reactant_device() = (
    try
        Reactant.default_device()
    catch
        Lux.cpu_device()
    end
)
cpu_device() = Lux.cpu_device()

function build_model(model_type::Symbol, config)
    if model_type == :TimeSeriesLSTM
        return TimeSeriesLSTM(config.input_dim, config.hidden_dim, config.out_dim)
    elseif model_type == :TimeSeriesCNN
        channels = get(config, :channels, 64)
        return TimeSeriesCNN(config.input_dim, config.out_dim; channels=channels)
    else
        error("Unknown model_type=$(model_type)")
    end
end

function load_jld2_model(path::AbstractString)
    @assert isfile(path) "Model file not found: $(path)"
    return JLD2.jldopen(path, "r") do f
        (
            model_type=read(f, "model_type"),
            parameters=read(f, "parameters"),
            states=read(f, "states"),
            config=read(f, "config"),
            meta=read(f, "meta"),
        )
    end
end

function same_scaler(s1, s2; atol=1f-6)
    length(s1.μ) == length(s2.μ) || return false
    length(s1.σ) == length(s2.σ) || return false
    return maximum(abs.(s1.μ .- s2.μ)) ≤ atol && maximum(abs.(s1.σ .- s2.σ)) ≤ atol
end

function predict_unscaled(model, ps, st, X_scaled; dev=reactant_device())
    Xd = dev(Float32.(X_scaled))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)
    model_compiled = if dev isa ReactantDevice
        @compile model(Xd, ps_d, st_test)
    else
        model
    end
    y_sc, _ = model_compiled(Xd, ps_d, st_test)
    y_sc = cpu_device()(y_sc)
    return Array(y_sc)
end

function to_vecs(Y::AbstractMatrix)
    return [Vector{Float64}(Y[:, j]) for j in 1:size(Y, 2)]
end

function mse_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d = size(y, 1)
    return mean(sum((ŷ .- y).^2; dims=1)) / d
end

function mae_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d = size(y, 1)
    return mean(sum(abs.(ŷ .- y); dims=1)) / d
end

function rmse_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    return sqrt(mse_mv(ŷ, y))
end

function r2_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d = size(y, 1)
    r2s = Float64[]
    for i in 1:d
        push!(r2s, r2(ŷ[i, :], y[i, :]))
    end
    return mean(r2s)
end

function mape_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d = size(y, 1)
    m = Float64[]
    for i in 1:d
        push!(m, mape(ŷ[i, :], y[i, :]))
    end
    return mean(m)
end

function smape_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d = size(y, 1)
    m = Float64[]
    for i in 1:d
        push!(m, smape(ŷ[i, :], y[i, :]))
    end
    return mean(m)
end

function make_features(X_scaled)
    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        x_last_cos = map(cos,X_scaled[:, end, j])
        x_last_sin = map(sin,X_scaled[:, end, j])
        x_last = Float64.(X_scaled[:, end, j])
        feats[j] = vcat(1.0, x_last,x_last_cos,x_last_sin)
    end
    return feats
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

#function main()
    if length(ARGS) < 2
        println("Usage: julia scripts/dynamic_neural_ensemble_rxinfer.jl <model1.jld2> <model2.jld2> [more...]")
        return
    end

    model_paths = ["/Users/ruiite/projects/prob_ensem_forecast/probabilistic_ensemble_forecasting/models/ETTh1_h96_CNN_enzyme.jld2",
        "/Users/ruiite/projects/prob_ensem_forecast/probabilistic_ensemble_forecasting/models/ETTh1_h96_LSTM_enzyme.jld2"]

    models = map(load_jld2_model, model_paths)

    base_meta = models[1].meta
    for m in models[2:end]
        if m.meta.dataset != base_meta.dataset || m.meta.seq_len != base_meta.seq_len || m.meta.horizon != base_meta.horizon
            error("All models must share dataset, seq_len, and horizon. Got $(m.meta.dataset), seq_len=$(m.meta.seq_len), horizon=$(m.meta.horizon)")
        end
        if !same_scaler(m.meta.scaler, base_meta.scaler)
            error("All models must share the same scaler (train split).")
        end
    end

    @info "Loading dataset" dataset=base_meta.dataset seq_len=base_meta.seq_len horizon=base_meta.horizon

    data_dir = joinpath(@__DIR__, "..", "data")
    ds_path = joinpath(data_dir, String(base_meta.dataset))
    Xmat, _ = load_ett(ds_path)

    X3, Y2 = make_sequences(Xmat; seq_len=Int(base_meta.seq_len), horizon=Int(base_meta.horizon))
    split = base_meta.split
    Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xtr_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)


    n_forecasters = length(models)
    n_train = size(Xval, 3)
    n_test = size(Xte, 3)
    d = size(Yval, 1)

    predictions_train = Array{Float64}(undef, n_forecasters, d, n_train)
    predictions_test = Array{Float64}(undef, n_forecasters, d, n_test)

    @info "Running forecasters" n_forecasters n_train n_test output_dim=d

    for (i, m) in enumerate(models)
        model = build_model(m.model_type, m.config)
        yhat_tr_sc = predict_unscaled(model, m.parameters, m.states, Xtr_s)
        yhat_te_sc = predict_unscaled(model, m.parameters, m.states, Xte_s)

        yhat_tr = inverse_targets(scaler, yhat_tr_sc)
        yhat_te = inverse_targets(scaler, yhat_te_sc)

        predictions_train[i, :, :] = Float64.(yhat_tr)
        predictions_test[i, :, :] = Float64.(yhat_te)
        @info "Forecaster ready" index=i model_type=m.model_type path=model_paths[i]
    end

    y_train = to_vecs(Float64.(Yval))
    y_test = to_vecs(Float64.(Yte))

    predictions_train_vec = Array{Vector{Float64}}(undef, n_forecasters, n_train)
    predictions_test_vec = Array{Vector{Float64}}(undef, n_forecasters, n_test)
    for i in 1:n_forecasters
        for j in 1:n_train
            predictions_train_vec[i, j] = Vector{Float64}(predictions_train[i, :, j])
        end
        for j in 1:n_test
            predictions_test_vec[i, j] = Vector{Float64}(predictions_test[i, :, j])
        end
    end

    features_train = make_features(Xtr_s)
    features_test = make_features(Xte_s)
    n_features = length(features_train[1])
    #N_FEATURES[] = n_features

    @info "Step 1: Training dynamic ensemble using RxInfer" n_features=n_features
    w_priors_init = [MvNormalMeanScalePrecision(zeros(n_features), 0.1) for _ in 1:n_forecasters]
    #w_priors_init = [MvNormalMeanPrecision(zeros(n_features), 0.1 * diagm(ones(n_features))) for _ in 1:n_forecasters]
    τ_priors_init = [ GammaShapeScale(1.0, 1e12)  for _ in 1:n_forecasters ]

    dynamic_result = infer(
        model = dynamic_mv_ensemble_model(
            n_forecasters = n_forecasters,
            n_obs = n_train,
            w_priors = w_priors_init,
            τ_priors = τ_priors_init
        ),
        data = (y = y_train, features = features_train, predictions = predictions_train_vec),
        constraints = dynamic_ensemble_constraints(),
        initialization = dynamic_ensemble_init(n_features),
        iterations = 30,
        free_energy = false,
        showprogress = true,
    )

    w_posteriors = dynamic_result.posteriors[:w][end]
    τ_posteriors = dynamic_result.posteriors[:τ][end]

    @info "Step 2: Dynamic ensemble prediction on test"
    # ensemble_mean = Array{Float64}(undef, d, n_test)
    # ensemble_std = Array{Float64}(undef, n_test)


    # for j in 1:n_test
    #     γ_values = zeros(n_forecasters)
    #     for i in 1:n_forecasters
    #         w_mean = mean(w_posteriors[i])
    #         z_ij = dot(w_mean, features_test[j])
    #         γ_values[i] = exp(z_ij)
    #     end
    #     γ_values_all[:, j] = γ_values
    #     total_precision = sum(γ_values)
    #     ensemble_mean[:, j] = sum(γ_values[i] .* predictions_test[i, :, j] for i in 1:n_forecasters) / total_precision
    #     ensemble_std[j] = sqrt(1.0 / total_precision)
    # end
    y_missing = [missing for _ in 1:n_test]

    dynamic_predict = infer(
        model = dynamic_mv_ensemble_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            w_priors = w_posteriors,
            τ_priors = τ_posteriors
        ),
        data = (y = y_missing, features = features_test, predictions = predictions_test_vec),
        constraints = dynamic_ensemble_constraints(),
        initialization = dynamic_ensemble_init(n_features),
        iterations = 10
    )

    dynamic_predictions = dynamic_predict.predictions[:y][end]
    ensemble_mean = hcat(map(mean, dynamic_predictions)...)
    ensemble_std = map((a)->a[1,1],map(std, dynamic_predictions))


    # Extract γ posteriors for weight uncertainty visualization
    γ_dynamic_posteriors = dynamic_predict.posteriors[:γ][end]  # [n_forecasters, n_test] matrix of distributions

    γ_values_all = mean.(γ_dynamic_posteriors)

    y_test_mat = Float64.(Yte)
    ensemble_metrics = (
        mse = mse_mv(ensemble_mean, y_test_mat),
        mae = mae_mv(ensemble_mean, y_test_mat),
        rmse = rmse_mv(ensemble_mean, y_test_mat),
        r2 = r2_mv(ensemble_mean, y_test_mat),
        mape = mape_mv(ensemble_mean, y_test_mat),
        smape = smape_mv(ensemble_mean, y_test_mat),
        mean_std = mean(ensemble_std),
    )

    @info "Step 3: Performance comparison on test"
    individual = []
    for i in 1:n_forecasters
        yhat = predictions_test[i, :, :]
        push!(individual, (
            path=model_paths[i],
            mse=mse_mv(yhat, y_test_mat),
            mae=mae_mv(yhat, y_test_mat),
            rmse=rmse_mv(yhat, y_test_mat),
            r2=r2_mv(yhat, y_test_mat),
            mape=mape_mv(yhat, y_test_mat),
            smape=smape_mv(yhat, y_test_mat),
        ))
    end

    simple_avg = vec(mean(predictions_test; dims=1)) |> x -> reshape(x, d, n_test)
    simple_metrics = (
        mse = mse_mv(simple_avg, y_test_mat),
        mae = mae_mv(simple_avg, y_test_mat),
        rmse = rmse_mv(simple_avg, y_test_mat),
        r2 = r2_mv(simple_avg, y_test_mat),
        mape = mape_mv(simple_avg, y_test_mat),
        smape = smape_mv(simple_avg, y_test_mat),
    )

    @info "Dynamic ensemble metrics" ensemble_metrics...
    @info "Simple average metrics" simple_metrics...
    for (i, m) in enumerate(individual)
        @info "Forecaster metrics" index=i path=m.path mse=m.mse mae=m.mae rmse=m.rmse r2=m.r2 mape=m.mape smape=m.smape
    end

    # -------------------------------------------------------------------------
    # Visualization (mirrors dynamic_ensemble_rxinfer.jl)
    # -------------------------------------------------------------------------
    x_test = 1:n_test
    y_test_mat_T = permutedims(y_test_mat, (2, 1))              # n_test x d
    ensemble_mean_T = permutedims(ensemble_mean, (2, 1))        # n_test x d
    simple_avg_T = permutedims(simple_avg, (2, 1))              # n_test x d

    p1 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Dynamic Ensemble vs Individual (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright
    )
    plot!(p1, x_test, ensemble_mean_T[:, 1],
        ribbon=2 .* ensemble_std,
        label="Dynamic ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )

    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan]
    for i in 1:n_forecasters
        plot!(p1, x_test, predictions_test[i, 1, :],
            label="F$(i)", ls=:dash, alpha=0.6, color=colors[mod1(i, length(colors))]
        )
    end

    p2 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Simple Avg vs Dynamic (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright
    )
    plot!(p2, x_test, simple_avg_T[:, 1],
        label="Simple Avg", lw=2, color=:orange, ls=:dash
    )
    plot!(p2, x_test, ensemble_mean_T[:, 1],
        label="Dynamic", lw=2, color=:blue
    )

    p3 = plot(title="Dynamic Precision Weights",
        xlabel="t", ylabel="γ(x) = exp(w'x)",
        legend=:topright
    )
    for i in 1:n_forecasters
        plot!(p3, x_test, γ_values_all[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    p4 = plot(title="Uncertainty (Dynamic)",
        xlabel="t", ylabel="Standard Deviation (σ)",
        legend=:topright
    )
    plot!(p4, x_test, ensemble_std, label="Dynamic σ", lw=2, color=:blue)

    all_mses = vcat([m.mse for m in individual], [simple_metrics.mse, ensemble_metrics.mse])
    all_labels = vcat(["F$(i)" for i in 1:n_forecasters], ["Simple Avg", "Dynamic"])
    bar_colors = vcat(fill(:gray, n_forecasters), [:orange, :blue])
    p5 = bar(1:length(all_mses), all_mses,
        title="MSE Comparison (Test Set, avg over dims)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45
    )

    normalized_weights = γ_values_all ./ sum(γ_values_all; dims=1)
    p6 = plot(title="Normalized Dynamic Weights",
        xlabel="t", ylabel="Weight (normalized)",
        legend=:outerright
    )
    for i in 1:n_forecasters
        plot!(p6, x_test, normalized_weights[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    plt = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1200, 1200))
    savefig(plt, "dynamic_neural_ensemble_rxinfer.png")
    @info "Saved visualization" file="dynamic_neural_ensemble_rxinfer.png"

    @info "Done"
#end

# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end
