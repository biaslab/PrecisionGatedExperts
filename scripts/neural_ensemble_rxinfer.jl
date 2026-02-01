#!/usr/bin/env julia

"""
Probabilistic Ensemble (RxInfer) over neural forecasters saved as JLD2.

Usage:
    julia scripts/neural_ensemble_rxinfer.jl <model1.jld2> <model2.jld2> [more...]
"""

using RxInfer
using Distributions
using Statistics
using LinearAlgebra
using JLD2
using Lux
using Reactant
using ProbabilisticEnsembling
using ExponentialFamilyProjection  # ensure scaler types deserialize
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
# RxInfer model
# -----------------------------------------------------------------------------

@model function mv_ensemble_precision_model(n_forecasters, n_samples, means, y, priors)
    local γ

    for i in 1:n_forecasters
        γ[i] ~ priors[i]
    end

    for i in 1:n_forecasters
        for j in 1:n_samples
            idx = (i - 1) * n_samples + j
            y[j] ~ MvNormalMeanScalePrecision(means[idx], γ[i])
        end
    end
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
    d, n = size(y)
    r2s = Float64[]
    for i in 1:d
        push!(r2s, r2(ŷ[i, :], y[i, :]))
    end
    return mean(r2s)
end

function mape_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d, n = size(y)
    m = Float64[]
    for i in 1:d
        push!(m, mape(ŷ[i, :], y[i, :]))
    end
    return mean(m)
end

function smape_mv(ŷ::AbstractMatrix, y::AbstractMatrix)
    d, n = size(y)
    m = Float64[]
    for i in 1:d
        push!(m, smape(ŷ[i, :], y[i, :]))
    end
    return mean(m)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    if length(ARGS) < 2
        println("Usage: julia scripts/neural_ensemble_rxinfer.jl <model1.jld2> <model2.jld2> [more...]")
        return
    end

    model_paths = ARGS
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
    Xtr_s = scale_inputs(scaler, Xtr)
    Xte_s = scale_inputs(scaler, Xte)

    n_forecasters = length(models)
    n_train = size(Xtr, 3)
    n_test = size(Xte, 3)
    d = size(Ytr, 1)

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

    y_train = to_vecs(Float64.(Ytr))
    y_test = to_vecs(Float64.(Yte))

    means_train = Vector{Vector{Float64}}(undef, n_forecasters * n_train)
    for i in 1:n_forecasters
        for j in 1:n_train
            means_train[(i - 1) * n_train + j] = Vector{Float64}(predictions_train[i, :, j])
        end
    end

    means_test = Vector{Vector{Float64}}(undef, n_forecasters * n_test)
    for i in 1:n_forecasters
        for j in 1:n_test
            means_test[(i - 1) * n_test + j] = Vector{Float64}(predictions_test[i, :, j])
        end
    end

    @info "Step 1: Learning precision parameters on train"
    starting_priors = [GammaShapeRate(1.0, 1e-12) for _ in 1:n_forecasters]
    result = infer(
        model = mv_ensemble_precision_model(n_forecasters=n_forecasters, n_samples=n_train, priors=starting_priors),
        data = (y = y_train, means = means_train),
        iterations = 20
    )

    γ_posteriors = result.posteriors[:γ][end]
    γ_means = map(mean, γ_posteriors)
    weights = γ_means ./ sum(γ_means)

    @info "Learned weights (normalized precisions)"
    for i in 1:n_forecasters
        mse_i = mse_mv(predictions_train[i, :, :], Ytr)
        @info "Forecaster" index=i path=model_paths[i] E_γ=round(γ_means[i], digits=4) train_MSE=round(mse_i, digits=6)
    end

    @info "Step 2: Ensemble prediction on test"
    prediction_array = [missing for _ in 1:n_test]
    infer_predict = infer(
        model = mv_ensemble_precision_model(n_forecasters=n_forecasters, n_samples=n_test, priors=γ_posteriors),
        data = (y = prediction_array, means = means_test),
        iterations = 1
    )

    ensemble_predictions = infer_predict.predictions[:y][end]
    ensemble_mean = [mean(p) for p in ensemble_predictions]
    ensemble_cov = [cov(p) for p in ensemble_predictions]
    ensemble_std = [sqrt(mean(diag(c))) for c in ensemble_cov]

    ensemble_mean_mat = hcat(ensemble_mean...)
    y_test_mat = Float64.(Yte)

    ensemble_metrics = (
        mse = mse_mv(ensemble_mean_mat, y_test_mat),
        mae = mae_mv(ensemble_mean_mat, y_test_mat),
        rmse = rmse_mv(ensemble_mean_mat, y_test_mat),
        r2 = r2_mv(ensemble_mean_mat, y_test_mat),
        mape = mape_mv(ensemble_mean_mat, y_test_mat),
        smape = smape_mv(ensemble_mean_mat, y_test_mat),
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

    @info "Ensemble metrics" ensemble_metrics...
    @info "Simple average metrics" simple_metrics...
    for (i, m) in enumerate(individual)
        @info "Forecaster metrics" index=i path=m.path mse=m.mse mae=m.mae rmse=m.rmse r2=m.r2 mape=m.mape smape=m.smape
    end

    # -------------------------------------------------------------------------
    # Visualization (same layout as ensemble_multivariate_rxinfer.jl)
    # -------------------------------------------------------------------------
    x_test = 1:n_test
    y_test_mat_T = permutedims(y_test_mat, (2, 1))                # n_test x d
    ensemble_mean_mat_T = permutedims(ensemble_mean_mat, (2, 1))  # n_test x d
    simple_avg_mat_T = permutedims(simple_avg, (2, 1))            # n_test x d

    p1 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Ensemble vs Individual Predictions (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright, size=(600, 400)
    )
    plot!(p1, x_test, ensemble_mean_mat_T[:, 1],
        ribbon=2 .* ensemble_std,
        label="Ensemble ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )
    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan]
    for i in 1:n_forecasters
        plot!(p1, x_test, predictions_test[i, 1, :],
            label="F$(i)",
            ls=:dash, alpha=0.6, color=colors[mod1(i, length(colors))]
        )
    end

    p2 = bar(1:n_forecasters, weights,
        title="Learned Ensemble Weights",
        xlabel="Forecaster", ylabel="Weight",
        legend=false, color=:steelblue,
        size=(600, 400)
    )

    all_mses = vcat([m.mse for m in individual], [simple_metrics.mse, ensemble_metrics.mse])
    all_labels = vcat(["F$(i)" for i in 1:n_forecasters], ["Simple Avg", "Ensemble"])
    bar_colors = vcat(fill(:gray, n_forecasters), [:orange, :green])
    p3 = bar(1:length(all_mses), all_mses,
        title="MSE Comparison (Test Set, avg over dims)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45, size=(600, 400)
    )

    p4 = plot(title="Prediction Errors vs True Values (Dim 1)",
        xlabel="t", ylabel="Error (pred - true)",
        legend=:topright, size=(600, 400)
    )
    plot!(p4, x_test, ensemble_mean_mat_T[:, 1] .- y_test_mat_T[:, 1], label="Ensemble", lw=2, color=:blue)
    plot!(p4, x_test, simple_avg_mat_T[:, 1] .- y_test_mat_T[:, 1], label="Simple Avg", lw=1, color=:orange, ls=:dash)
    best_idx = argmin([m.mse for m in individual])
    plot!(p4, x_test, predictions_test[best_idx, 1, :] .- y_test_mat_T[:, 1],
        label="Best Individual", lw=1, color=:green, ls=:dot)
    hline!(p4, [0], color=:black, ls=:dash, label="", alpha=0.5)

    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
    savefig(plt, "neural_ensemble_rxinfer.png")
    @info "Saved visualization" file="neural_ensemble_rxinfer.png"

    @info "Done" weights=weights
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
