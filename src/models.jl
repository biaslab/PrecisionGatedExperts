using Lux

export TimeSeriesLSTM,
    TimeSeriesLSTMOneStep,
    TimeSeriesCNN,
    TimeSeriesMLP,
    TimeSeriesNLinear,
    TimeSeriesNConv,
    TimeSeriesDLinear,
    build_model

struct TimeSeriesLSTM{L,H} <: Lux.AbstractLuxContainerLayer{(:lstm_cell, :head)}
    lstm_cell::L
    head::H
end

function TimeSeriesLSTM(in_dims::Int, hidden_dims::Int, out_dims::Int)
    return TimeSeriesLSTM(
        LSTMCell(in_dims => hidden_dims),
        Chain(Dense(hidden_dims => hidden_dims, relu), Dense(hidden_dims => out_dims)),
    )
end

function (m::TimeSeriesLSTM)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
    (y, carry), st_lstm = m.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    for x_t in x_rest
        (y, carry), st_lstm = m.lstm_cell((x_t, carry), ps.lstm_cell, st_lstm)
    end
    y, st_head = m.head(y, ps.head, st.head)
    st = merge(st, (lstm_cell = st_lstm, head = st_head))
    return y, st
end

struct TimeSeriesLSTMOneStep{L,H} <: Lux.AbstractLuxContainerLayer{(:lstm_cell, :head)}
    lstm_cell::L
    head::H
    chunk_len::Int
    n_steps::Int
end

function TimeSeriesLSTMOneStep(
    in_dims::Int,
    seq_len::Int,
    hidden_dims::Int,
    out_dims::Int;
    n_steps::Int = 1,
)
    n_steps >= 1 || error("n_steps must be >= 1")
    (seq_len % n_steps == 0) ||
        error("seq_len=$(seq_len) must be divisible by n_steps=$(n_steps)")
    chunk_len = div(seq_len, n_steps)

    return TimeSeriesLSTMOneStep(
        LSTMCell((in_dims * chunk_len) => hidden_dims),
        Chain(Dense(hidden_dims => hidden_dims, relu), Dense(hidden_dims => out_dims)),
        chunk_len,
        n_steps,
    )
end

function (m::TimeSeriesLSTMOneStep)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    bsz = size(x, 3)
    if m.n_steps == 1
        x2 = reshape(x, :, bsz)
        (y, _), st_lstm = m.lstm_cell(x2, ps.lstm_cell, st.lstm_cell)
    else
        x4 = reshape(x, size(x, 1), m.chunk_len, m.n_steps, bsz)
        x1 = reshape(@view(x4[:, :, 1, :]), :, bsz)
        (y, carry), st_lstm = m.lstm_cell(x1, ps.lstm_cell, st.lstm_cell)
        for s = 2:m.n_steps
            xs = reshape(@view(x4[:, :, s, :]), :, bsz)
            (y, carry), st_lstm = m.lstm_cell((xs, carry), ps.lstm_cell, st_lstm)
        end
    end
    y, st_head = m.head(y, ps.head, st.head)
    st = merge(st, (lstm_cell = st_lstm, head = st_head))
    return y, st
end

struct TimeSeriesCNN{C1,C2,H} <: Lux.AbstractLuxContainerLayer{(:conv1, :conv2, :head)}
    conv1::C1
    conv2::C2
    head::H
end

function TimeSeriesCNN(
    in_dims::Int,
    out_dims::Int;
    channels::Int = 64,
    k::Int = 7,
    stride::Int = 2,
)
    return TimeSeriesCNN(
        Conv((k,), in_dims => channels, relu; pad = (1,), stride = (stride,)),
        Conv((k,), channels => channels, relu; pad = (1,), stride = (stride,)),
        Chain(Dense(channels => channels, relu), Dense(channels => out_dims)),
    )
end

function (m::TimeSeriesCNN)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x = permutedims(x, (2, 1, 3))
    x, st1 = m.conv1(x, ps.conv1, st.conv1)
    x, st2 = m.conv2(x, ps.conv2, st.conv2)
    x = mean(x; dims = 1)
    x = reshape(x, size(x, 2), size(x, 3))
    y, st_head = m.head(x, ps.head, st.head)
    st = merge(st, (conv1 = st1, conv2 = st2, head = st_head))
    return y, st
end

struct TimeSeriesMLP{H} <: Lux.AbstractLuxContainerLayer{(:head,)}
    head::H
end

function TimeSeriesMLP(
    in_dims::Int,
    seq_len::Int,
    out_dims::Int;
    hidden_dims::Int = 64,
    depth::Int = 2,
)
    depth >= 1 || error("depth must be >= 1")
    layers = Any[Dense(in_dims * seq_len => hidden_dims, relu)]
    for _ = 2:depth
        push!(layers, Dense(hidden_dims => hidden_dims, relu))
    end
    push!(layers, Dense(hidden_dims => out_dims))
    return TimeSeriesMLP(Chain(layers...))
end

function (m::TimeSeriesMLP)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x2 = reshape(x, :, size(x, 3))
    y, st_head = m.head(x2, ps.head, st.head)
    st = merge(st, (head = st_head,))
    return y, st
end

struct TimeSeriesNLinear{H} <: Lux.AbstractLuxContainerLayer{(:head,)}
    head::H
    in_dims::Int
    seq_len::Int
    out_dims::Int
end

function TimeSeriesNLinear(in_dims::Int, seq_len::Int, out_dims::Int; bias::Bool = true)
    in_dims == out_dims || error(
        "TimeSeriesNLinear is channel-independent and requires out_dims == in_dims. Got in_dims=$(in_dims), out_dims=$(out_dims).",
    )
    return TimeSeriesNLinear(
        Dense(seq_len => 1; use_bias = bias),
        in_dims,
        seq_len,
        out_dims,
    )
end

function (m::TimeSeriesNLinear)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    size(x, 1) == m.in_dims || error("Expected in_dims=$(m.in_dims), got $(size(x, 1)).")
    size(x, 2) == m.seq_len || error("Expected seq_len=$(m.seq_len), got $(size(x, 2)).")

    f = size(x, 1)
    bsz = size(x, 3)
    x_last = @view x[:, m.seq_len, :]                # (F, B)
    x_norm = x .- reshape(x_last, f, 1, bsz)         # (F, T, B)

    # Apply shared Dense(seq_len=>1) independently per channel.
    x2 = reshape(permutedims(x_norm, (2, 1, 3)), m.seq_len, :)
    y2, st_head = m.head(x2, ps.head, st.head)       # (1, F*B)
    y = reshape(y2, 1, f, bsz)
    y = reshape(permutedims(y, (2, 1, 3)), f, bsz)   # (F, B)
    y .+= x_last

    st = merge(st, (head = st_head,))
    return y, st
end

struct TimeSeriesNConv{C} <: Lux.AbstractLuxContainerLayer{(:conv,)}
    conv::C
    in_dims::Int
    seq_len::Int
    out_dims::Int
    kernel_size::Int
end

function TimeSeriesNConv(in_dims::Int, seq_len::Int, out_dims::Int; kernel_size::Int = 25)
    in_dims == out_dims || error(
        "TimeSeriesNConv is channel-independent and requires out_dims == in_dims. Got in_dims=$(in_dims), out_dims=$(out_dims).",
    )
    kernel_size >= 1 || error("kernel_size must be >= 1")
    isodd(kernel_size) || error("kernel_size must be odd, got $(kernel_size)")
    pad = (kernel_size ÷ 2,)
    # Depthwise temporal convolution: one independent kernel per channel (no channel mixing).
    return TimeSeriesNConv(
        Conv((kernel_size,), in_dims => out_dims; pad = pad, groups = in_dims),
        in_dims,
        seq_len,
        out_dims,
        kernel_size,
    )
end

function (m::TimeSeriesNConv)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    size(x, 1) == m.in_dims || error("Expected in_dims=$(m.in_dims), got $(size(x, 1)).")
    size(x, 2) == m.seq_len || error("Expected seq_len=$(m.seq_len), got $(size(x, 2)).")

    f = size(x, 1)
    bsz = size(x, 3)
    x_last = @view x[:, m.seq_len, :]                      # (F, B)
    x_norm = x .- reshape(x_last, f, 1, bsz)               # (F, T, B)

    # Channel-independent (depthwise) temporal conv over (T, F, B).
    x2 = permutedims(x_norm, (2, 1, 3))                      # (T, F, B)
    y2, st_conv = m.conv(x2, ps.conv, st.conv)               # (T, F, B)
    y = @view(y2[m.seq_len, :, :])                            # (F, B)
    y .+= x_last

    st = merge(st, (conv = st_conv,))
    return y, st
end

struct TimeSeriesDLinear{HT,HS} <:
       Lux.AbstractLuxContainerLayer{(:trend_head, :seasonal_head)}
    trend_head::HT
    seasonal_head::HS
    in_dims::Int
    seq_len::Int
    out_dims::Int
    kernel_size::Int
end

function TimeSeriesDLinear(
    in_dims::Int,
    seq_len::Int,
    out_dims::Int;
    kernel_size::Int = 25,
    bias::Bool = true,
)
    in_dims == out_dims || error(
        "TimeSeriesDLinear is channel-independent and requires out_dims == in_dims. Got in_dims=$(in_dims), out_dims=$(out_dims).",
    )
    kernel_size >= 1 || error("kernel_size must be >= 1")
    isodd(kernel_size) || error("kernel_size must be odd, got $(kernel_size)")
    return TimeSeriesDLinear(
        Dense(seq_len => 1; use_bias = bias),
        Dense(seq_len => 1; use_bias = bias),
        in_dims,
        seq_len,
        out_dims,
        kernel_size,
    )
end

function moving_average_3d(x::AbstractArray{T,3}, kernel_size::Int) where {T}
    f, tlen, bsz = size(x)
    h = (kernel_size - 1) ÷ 2
    trend = similar(x)

    @inbounds for t = 1:tlen
        trend[:, t, :] .= zero(T)
        for o = (-h):h
            src_t = clamp(t + o, 1, tlen)
            trend[:, t, :] .+= x[:, src_t, :]
        end
        trend[:, t, :] ./= T(kernel_size)
    end

    return trend
end

function (m::TimeSeriesDLinear)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    size(x, 1) == m.in_dims || error("Expected in_dims=$(m.in_dims), got $(size(x, 1)).")
    size(x, 2) == m.seq_len || error("Expected seq_len=$(m.seq_len), got $(size(x, 2)).")

    f = size(x, 1)
    bsz = size(x, 3)
    trend = moving_average_3d(x, m.kernel_size)
    seasonal = x .- trend

    trend2 = reshape(permutedims(trend, (2, 1, 3)), m.seq_len, :)
    seasonal2 = reshape(permutedims(seasonal, (2, 1, 3)), m.seq_len, :)

    y_trend2, st_trend = m.trend_head(trend2, ps.trend_head, st.trend_head)
    y_seasonal2, st_seasonal =
        m.seasonal_head(seasonal2, ps.seasonal_head, st.seasonal_head)

    y_trend = reshape(permutedims(reshape(y_trend2, 1, f, bsz), (2, 1, 3)), f, bsz)
    y_seasonal = reshape(permutedims(reshape(y_seasonal2, 1, f, bsz), (2, 1, 3)), f, bsz)
    y = y_trend .+ y_seasonal

    st = merge(st, (trend_head = st_trend, seasonal_head = st_seasonal))
    return y, st
end

function build_model(model_type::Symbol, config)
    if model_type == :TimeSeriesLSTM
        return TimeSeriesLSTM(config.input_dim, config.hidden_dim, config.out_dim)
    elseif model_type == :TimeSeriesLSTMOneStep
        n_steps = get(config, :n_steps, 1)
        return TimeSeriesLSTMOneStep(
            config.input_dim,
            config.seq_len,
            config.hidden_dim,
            config.out_dim;
            n_steps = n_steps,
        )
    elseif model_type == :TimeSeriesCNN
        channels = get(config, :channels, 64)
        return TimeSeriesCNN(config.input_dim, config.out_dim; channels = channels)
    elseif model_type == :TimeSeriesMLP
        hidden_dims = get(config, :hidden_dim, 64)
        depth = get(config, :depth, 2)
        return TimeSeriesMLP(
            config.input_dim,
            config.seq_len,
            config.out_dim;
            hidden_dims = hidden_dims,
            depth = depth,
        )
    elseif model_type == :TimeSeriesNLinear
        bias = get(config, :bias, true)
        return TimeSeriesNLinear(
            config.input_dim,
            config.seq_len,
            config.out_dim;
            bias = bias,
        )
    elseif model_type == :TimeSeriesNConv
        kernel_size = get(config, :kernel_size, 25)
        return TimeSeriesNConv(
            config.input_dim,
            config.seq_len,
            config.out_dim;
            kernel_size = kernel_size,
        )
    elseif model_type == :TimeSeriesDLinear
        kernel_size = get(config, :kernel_size, 25)
        bias = get(config, :bias, true)
        return TimeSeriesDLinear(
            config.input_dim,
            config.seq_len,
            config.out_dim;
            kernel_size = kernel_size,
            bias = bias,
        )
    else
        error("Unknown model_type=$(model_type)")
    end
end
