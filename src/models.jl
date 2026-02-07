using Lux

export TimeSeriesLSTM, TimeSeriesCNN, TimeSeriesMLP, build_model

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
    x = mean(x; dims=1)
    x = reshape(x, size(x, 2), size(x, 3))
    y, st_head = m.head(x, ps.head, st.head)
    st = merge(st, (conv1=st1, conv2=st2, head=st_head))
    return y, st
end

struct TimeSeriesMLP{H} <: Lux.AbstractLuxContainerLayer{(:head,)}
    head::H
end

function TimeSeriesMLP(
    in_dims::Int,
    seq_len::Int,
    out_dims::Int;
    hidden_dims::Int=64,
    depth::Int=2
)
    depth >= 1 || error("depth must be >= 1")
    layers = Any[Dense(in_dims * seq_len => hidden_dims, relu)]
    for _ in 2:depth
        push!(layers, Dense(hidden_dims => hidden_dims, relu))
    end
    push!(layers, Dense(hidden_dims => out_dims))
    return TimeSeriesMLP(Chain(layers...))
end

function (m::TimeSeriesMLP)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x2 = reshape(x, :, size(x, 3))
    y, st_head = m.head(x2, ps.head, st.head)
    st = merge(st, (head=st_head,))
    return y, st
end

function build_model(model_type::Symbol, config)
    if model_type == :TimeSeriesLSTM
        return TimeSeriesLSTM(config.input_dim, config.hidden_dim, config.out_dim)
    elseif model_type == :TimeSeriesCNN
        channels = get(config, :channels, 64)
        return TimeSeriesCNN(config.input_dim, config.out_dim; channels=channels)
    elseif model_type == :TimeSeriesMLP
        hidden_dims = get(config, :hidden_dim, 64)
        depth = get(config, :depth, 2)
        return TimeSeriesMLP(config.input_dim, config.seq_len, config.out_dim; hidden_dims=hidden_dims, depth=depth)
    else
        error("Unknown model_type=$(model_type)")
    end
end
