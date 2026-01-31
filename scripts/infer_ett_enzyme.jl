#!/usr/bin/env julia

# Inference for LSTM/CNN Lux models saved by scripts/train_ett_lstm_enzyme.jl (JLD2)

using ADTypes
using Lux
using JLD2
using MLUtils
using Reactant
using Random
using Statistics
using ProbabilisticEnsembling
using ExponentialFamilyProjection  # ensure scaler types deserialize

# -----------------------------------------------------------------------------

# Model Definitions (mirror training script)

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
    x = mean(x; dims=1)              # (1, C, N)
    x = reshape(x, size(x, 2), size(x, 3)) # (C, N)
    y, st_head = m.head(x, ps.head, st.head)
    st = merge(st, (conv1=st1, conv2=st2, head=st_head))
    return y, st
end

# -----------------------------------------------------------------------------

# Helpers

# -----------------------------------------------------------------------------

reactant_device() = (
    try
        Reactant.default_device()
    catch
        cpu_device()
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

function eval_on_test(model, ps, st, meta)
    # Rebuild dataset and splits per meta
    data_dir = joinpath(@__DIR__, "..", "data")
    ds_path = joinpath(data_dir, String(meta.dataset))
    Xmat, _ = load_ett(ds_path)

    X3, Y2 = make_sequences(Xmat; seq_len=Int(meta.seq_len), horizon=Int(meta.horizon))
    Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(meta.split.train, meta.split.val, meta.split.test))

    scaler = meta.scaler
    # Scale inputs/targets and evaluate on test split
    Xte_s = scale_inputs(scaler, Xte)
    Yte_s = scale_targets(scaler, Yte)

    dev = reactant_device()
    cdev = cpu_device()
    Xd = dev(Float32.(Xte_s))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)

    # Compile for inference on Reactant
    model_compiled = if dev isa ReactantDevice
        @compile model(Xd, ps_d, st_test)
    else
        model
    end

    ŷ_sc, _ = model_compiled(Xd, ps_d, st_test)
    ŷ_sc = cdev(ŷ_sc)

    ŷ = inverse_targets(scaler, Array(ŷ_sc))
    y = inverse_targets(scaler, Array(Yte_s))

    return (
        mse=mse(ŷ, y),
        mae=mae(ŷ, y),
        rmse=rmse(ŷ, y),
        r2=r2(ŷ, y),
        mape=mape(ŷ, y),
        smape=smape(ŷ, y),
        n_test=size(Xte, 3),
        n_features=size(Xte, 1),
    )
end

function main()
    if length(ARGS) < 1
        println("Usage: julia scripts/infer_ett_enzyme.jl <model_path.jld2>")
        return
    end
    model_path = ARGS[1]
    @info "Loading JLD2 model" path = model_path

    @assert isfile(model_path) "Model file not found: $(model_path)"
    data = JLD2.jldopen(model_path, "r") do f
        (
            model_type=read(f, "model_type"),
            parameters=read(f, "parameters"),
            states=read(f, "states"),
            config=read(f, "config"),
            meta=read(f, "meta"),
        )
    end

    model = build_model(data.model_type, data.config)
    metrics = eval_on_test(model, data.parameters, data.states, data.meta)

    @info "Test metrics" model = data.model_type dataset = data.meta.dataset horizon = data.meta.horizon metrics...
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
