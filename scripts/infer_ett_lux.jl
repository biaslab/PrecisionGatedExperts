#!/usr/bin/env julia

# Inference script for ETTh LSTM/CNN models trained with scripts/train_ett_lux.jl
using Pkg
using BSON: @load
using Random, Statistics
using Lux

include(joinpath(@__DIR__, "utils.jl"))
using .Utils: load_ett, make_sequences, train_val_test_split,
    scale_inputs, scale_targets, inverse_targets,
    mse, mae, rmse, r2, mape

function lstm_forward_batch(lstm_cell, head, ps_cell, st_cell, ps_head, st_head, xb)
    stc = st_cell
    h = nothing
    @inbounds for t in 1:size(xb, 2)
        xt = @view xb[:, t, :]
        h, stc = Lux.apply(lstm_cell, xt, ps_cell, stc)
    end
    if h isa Tuple
        h = h[1]
    end
    y, sth = Lux.apply(head, h, ps_head, st_head)
    return y, (stc, sth)
end

function cnn_forward_batch(layers, params, states, xb)
    (conv1, conv2, fc1, fc2) = layers
    (ps1, ps2, ps3, ps4) = params
    (st1, st2, st3, st4) = states
    # Permute input to (seq, channels, N)
    x = permutedims(xb, (2, 1, 3))
    x, st1 = Lux.apply(conv1, x, ps1, st1)
    x, st2 = Lux.apply(conv2, x, ps2, st2)
    # Global average over time
    x = mean(x; dims=1)
    x = reshape(x, size(x, 2), size(x, 3))
    x, st3 = Lux.apply(fc1, x, ps3, st3)
    y, st4 = Lux.apply(fc2, x, ps4, st4)
    return y, (st1, st2, st3, st4)
end

function reorder_features!(Xmat::Matrix{Float32}, current_feats::Vector{String}, desired_feats::Vector{String})
    if current_feats == desired_feats
        return Xmat, current_feats
    end
    mapidx = Dict(name => i for (i, name) in enumerate(current_feats))
    idxs = [get(mapidx, n, nothing) for n in desired_feats]
    if any(i -> i === nothing, idxs)
        missing = [desired_feats[i] for i in eachindex(idxs) if idxs[i] === nothing]
        error("Missing required features in data: $(missing)")
    end
    Xr = Xmat[idxs, :]
    return Xr, desired_feats
end

function main()
    if length(ARGS) == 0 && !haskey(ENV, "MODEL")
        println("Usage: julia scripts/infer_ett_lux.jl <path/to/model.bson> [optional data_dir]")
        println("Or set MODEL=/path/to/model.bson and DATA=/path/to/data")
        return
    end
    model_path = length(ARGS) >= 1 ? ARGS[1] : ENV["MODEL"]
    data_override = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "DATA", "")
    model_path = joinpath(@__DIR__, "../models/ETTh1_h192_CNN.bson")
    data_override = joinpath(@__DIR__, "../data/ETTh1.csv")
    model = nothing
    ps = nothing
    st = nothing
    meta = nothing
    @load model_path model ps st meta

    dataset_name = hasproperty(meta, :dataset) ? meta.dataset : nothing
    horizon = hasproperty(meta, :horizon) ? meta.horizon : 1
    seq_len = hasproperty(meta, :seq_len) ? meta.seq_len : 96
    features = hasproperty(meta, :features) ? collect(meta.features) : nothing
    scaler = hasproperty(meta, :scaler) ? meta.scaler : nothing
    @assert scaler !== nothing "Saved scaler not found in model file"

    data_dir = if !isempty(data_override)
        data_override
    elseif dataset_name !== nothing
        # Resolve relative to repo layout
        joinpath(@__DIR__, "..", "data", String(dataset_name))
    else
        error("Please provide data directory as second arg or via DATA env var")
    end

    Xmat, feat_cols = load_ett(data_dir)
    if features !== nothing
        Xmat, _ = reorder_features!(Xmat, String.(feat_cols), String.(features))
    end

    X3, Y2 = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)
    Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(0.6, 0.2, 0.2))
    # Scale inputs and targets with saved scaler
    Xte_s = scale_inputs(scaler, Xte)
    Yte_s = scale_targets(scaler, Yte)

    if model == :LSTM_manual
        lstm_art = ps
        function predict_lstm(X)
            ŷ, _ = lstm_forward_batch(lstm_art.lstm_cell, lstm_art.head, lstm_art.ps_cell, lstm_art.st_cell, lstm_art.ps_head, lstm_art.st_head, X)
            ŷ
        end
        ŷ_sc = predict_lstm(Xte_s)
        ŷ = inverse_targets(scaler, Array(ŷ_sc))
        y = inverse_targets(scaler, Array(Yte_s))
        println("LSTM test metrics:")
        println((mse=mse(ŷ, y), mae=mae(ŷ, y), rmse=rmse(ŷ, y), r2=r2(ŷ, y), mape=mape(ŷ, y)))
    elseif model == :CNN_manual
        cnn_art = ps
        function predict_cnn(X)
            ŷ, _ = cnn_forward_batch(cnn_art.layers, cnn_art.ps, cnn_art.st, X)
            ŷ
        end
        ŷ_sc = predict_cnn(Xte_s)
        ŷ = inverse_targets(scaler, Array(ŷ_sc))
        y = inverse_targets(scaler, Array(Yte_s))
        println("CNN test metrics:")
        println((mse=mse(ŷ, y), mae=mae(ŷ, y), rmse=rmse(ŷ, y), r2=r2(ŷ, y), mape=mape(ŷ, y)))
    else
        error("Unknown model tag in BSON: $(model). Expected :LSTM_manual or :CNN_manual")
    end
end

main()
