using Lux

export TimeSeriesLSTM,
    TimeSeriesLSTMOneStep,
    TimeSeriesCNN,
    TimeSeriesMLP,
    TimeSeriesNLinear,
    TimeSeriesNConv,
    TimeSeriesDLinear,
    TimeSeriesPatchTST,
    TimeSeriesTimeMixerPP,
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

function feature_layer_norm(x::AbstractArray{T,3}; eps::T = T(1.0f-5)) where {T<:Real}
    μ = mean(x; dims = 1)
    xc = x .- μ
    σ2 = mean(xc .* xc; dims = 1)
    return xc ./ sqrt.(σ2 .+ eps)
end

function downsample_mean(x::AbstractArray{T,3}, factor::Int) where {T}
    factor >= 1 || error("downsample factor must be >= 1")
    factor == 1 && return x

    f, tlen, bsz = size(x)
    tnew = max(1, fld(tlen, factor))
    y = Array{T}(undef, f, tnew, bsz)

    @inbounds for i = 1:tnew
        s = (i - 1) * factor + 1
        e = min(i * factor, tlen)
        yi = dropdims(mean(@view(x[:, s:e, :]); dims = 2); dims = 2)
        y[:, i, :] .= yi
    end

    return y
end

function upsample_repeat_to(x::AbstractArray{T,3}, target_len::Int, factor::Int) where {T}
    factor >= 1 || error("upsample factor must be >= 1")
    x_rep = factor == 1 ? x : repeat(x; inner = (1, factor, 1))
    tcur = size(x_rep, 2)
    if tcur == target_len
        return x_rep
    elseif tcur > target_len
        return @view x_rep[:, 1:target_len, :]
    else
        pad = target_len - tcur
        last = @view x_rep[:, tcur:tcur, :]
        return cat(x_rep, repeat(last; inner = (1, pad, 1)); dims = 2)
    end
end

function sinusoidal_positional_encoding(d_model::Int, n_tokens::Int, ::Type{T}) where {T<:Real}
    pe = Array{T}(undef, d_model, n_tokens, 1)
    @inbounds for p = 1:n_tokens
        pos = T(p - 1)
        for i = 1:d_model
            k = div(i - 1, 2)
            denom = T(10000.0)^(T(2 * k) / T(d_model))
            angle = pos / denom
            pe[i, p, 1] = isodd(i) ? sin(angle) : cos(angle)
        end
    end
    return pe
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

struct PatchTSTEncoderBlock{A,F1,F2} <:
       Lux.AbstractLuxContainerLayer{(:self_attn, :ffn1, :ffn2)}
    self_attn::A
    ffn1::F1
    ffn2::F2
end

function PatchTSTEncoderBlock(d_model::Int; n_heads::Int = 4, ff_dim::Int = 128)
    d_model >= 1 || error("d_model must be >= 1")
    n_heads >= 1 || error("n_heads must be >= 1")
    ff_dim >= 1 || error("ff_dim must be >= 1")
    return PatchTSTEncoderBlock(
        MultiHeadAttention(
            d_model;
            nheads = n_heads,
            attention_dropout_probability = 0.0f0,
        ),
        Dense(d_model => ff_dim, relu),
        Dense(ff_dim => d_model),
    )
end

function (m::PatchTSTEncoderBlock)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T<:Real}
    x_ln1 = feature_layer_norm(x)
    attn_out, st_attn = m.self_attn((x_ln1, x_ln1, x_ln1), ps.self_attn, st.self_attn)
    x_res1 = x .+ attn_out[1]

    x_ln2 = feature_layer_norm(x_res1)
    ff1, st_ff1 = m.ffn1(x_ln2, ps.ffn1, st.ffn1)
    ff2, st_ff2 = m.ffn2(ff1, ps.ffn2, st.ffn2)
    y = x_res1 .+ ff2

    st = merge(st, (self_attn = st_attn, ffn1 = st_ff1, ffn2 = st_ff2))
    return y, st
end

struct TimeSeriesPatchTST{E,B,H} <:
       Lux.AbstractLuxContainerLayer{(:patch_embed, :encoder, :head)}
    patch_embed::E
    encoder::B
    head::H
    in_dims::Int
    seq_len::Int
    out_dims::Int
    patch_len::Int
    stride::Int
    d_model::Int
    n_patches::Int
    n_layers::Int
end

function TimeSeriesPatchTST(
    in_dims::Int,
    seq_len::Int,
    out_dims::Int;
    patch_len::Int = 16,
    stride::Int = 8,
    d_model::Int = 64,
    n_heads::Int = 4,
    n_layers::Int = 2,
    ff_dim::Int = 128,
)
    in_dims == out_dims || error(
        "TimeSeriesPatchTST uses channel-independent heads and requires out_dims == in_dims. Got in_dims=$(in_dims), out_dims=$(out_dims).",
    )
    patch_len >= 1 || error("patch_len must be >= 1")
    stride >= 1 || error("stride must be >= 1")
    n_heads >= 1 || error("n_heads must be >= 1")
    n_layers >= 1 || error("n_layers must be >= 1")
    (d_model % n_heads == 0) ||
        error("d_model=$(d_model) must be divisible by n_heads=$(n_heads)")
    patch_len <= seq_len || error("patch_len=$(patch_len) must be <= seq_len=$(seq_len)")

    n_patches = Int(fld(seq_len - patch_len, stride)) + 1
    n_patches >= 1 || error("No patches generated. Check patch_len and stride.")

    blocks = ntuple(
        _ -> PatchTSTEncoderBlock(d_model; n_heads = n_heads, ff_dim = ff_dim),
        n_layers,
    )

    return TimeSeriesPatchTST(
        Dense(patch_len => d_model),
        Chain(blocks...),
        Chain(Dense(d_model * n_patches => ff_dim, relu), Dense(ff_dim => 1)),
        in_dims,
        seq_len,
        out_dims,
        patch_len,
        stride,
        d_model,
        n_patches,
        n_layers,
    )
end

function (m::TimeSeriesPatchTST)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    size(x, 1) == m.in_dims || error("Expected in_dims=$(m.in_dims), got $(size(x, 1)).")
    size(x, 2) == m.seq_len || error("Expected seq_len=$(m.seq_len), got $(size(x, 2)).")

    f = size(x, 1)
    bsz = size(x, 3)

    patches = Array{T}(undef, m.patch_len, m.n_patches, f * bsz)
    @inbounds for p = 1:m.n_patches
        s = 1 + (p - 1) * m.stride
        e = s + m.patch_len - 1
        xt = permutedims(@view(x[:, s:e, :]), (2, 1, 3))
        patches[:, p, :] .= reshape(xt, m.patch_len, :)
    end

    patch2 = reshape(patches, m.patch_len, :)
    emb2, st_patch = m.patch_embed(patch2, ps.patch_embed, st.patch_embed)
    tokens = reshape(emb2, m.d_model, m.n_patches, :)
    pos = sinusoidal_positional_encoding(m.d_model, m.n_patches, T)
    tokens = tokens .+ pos
    enc, st_enc = m.encoder(tokens, ps.encoder, st.encoder)

    # Channel-independent forecasting head from patch tokens.
    flat = reshape(enc, m.d_model * m.n_patches, :)
    y_ch, st_head = m.head(flat, ps.head, st.head)
    y = reshape(y_ch, f, bsz)

    st = merge(st, (patch_embed = st_patch, encoder = st_enc, head = st_head))
    return y, st
end

struct TimeSeriesTimeMixerPP{S1,S2,S3,T1,T2,T3,C,H} <: Lux.AbstractLuxContainerLayer{(
    :seasonal_mixer_1,
    :seasonal_mixer_2,
    :seasonal_mixer_3,
    :trend_mixer_1,
    :trend_mixer_2,
    :trend_mixer_3,
    :channel_mixer,
    :head,
)}
    seasonal_mixer_1::S1
    seasonal_mixer_2::S2
    seasonal_mixer_3::S3
    trend_mixer_1::T1
    trend_mixer_2::T2
    trend_mixer_3::T3
    channel_mixer::C
    head::H
    in_dims::Int
    seq_len::Int
    out_dims::Int
    hidden_dims::Int
    kernel_sizes::NTuple{3,Int}
end

function TimeSeriesTimeMixerPP(
    in_dims::Int,
    seq_len::Int,
    out_dims::Int;
    hidden_dims::Int = 128,
    kernel_sizes::NTuple{3,Int} = (5, 13, 25),
)
    for k in kernel_sizes
        k >= 1 || error("kernel_size must be >= 1, got $(k)")
        isodd(k) || error("kernel_size must be odd, got $(k)")
    end
    t1 = seq_len
    t2 = max(1, fld(seq_len, 2))
    t3 = max(1, fld(seq_len, 4))
    channel_hidden = max(8, hidden_dims ÷ 2)

    return TimeSeriesTimeMixerPP(
        Chain(Dense(t1 => hidden_dims, relu), Dense(hidden_dims => t1)),
        Chain(Dense(t2 => hidden_dims, relu), Dense(hidden_dims => t2)),
        Chain(Dense(t3 => hidden_dims, relu), Dense(hidden_dims => t3)),
        Chain(Dense(t1 => hidden_dims, relu), Dense(hidden_dims => t1)),
        Chain(Dense(t2 => hidden_dims, relu), Dense(hidden_dims => t2)),
        Chain(Dense(t3 => hidden_dims, relu), Dense(hidden_dims => t3)),
        Chain(Dense(in_dims => channel_hidden, relu), Dense(channel_hidden => in_dims)),
        Chain(
            Dense(in_dims * seq_len => hidden_dims, relu),
            Dense(hidden_dims => out_dims),
        ),
        in_dims,
        seq_len,
        out_dims,
        hidden_dims,
        kernel_sizes,
    )
end

function (m::TimeSeriesTimeMixerPP)(
    x::AbstractArray{T,3},
    ps::NamedTuple,
    st::NamedTuple,
) where {T}
    size(x, 1) == m.in_dims || error("Expected in_dims=$(m.in_dims), got $(size(x, 1)).")
    size(x, 2) == m.seq_len || error("Expected seq_len=$(m.seq_len), got $(size(x, 2)).")

    f = size(x, 1)
    bsz = size(x, 3)

    x1 = x
    x2 = downsample_mean(x, 2)
    x3 = downsample_mean(x, 4)

    k1, k2, k3 = m.kernel_sizes
    tr1 = moving_average_3d(x1, min(k1, isodd(size(x1, 2)) ? size(x1, 2) : size(x1, 2) - 1))
    tr2 = moving_average_3d(x2, min(k2, isodd(size(x2, 2)) ? size(x2, 2) : size(x2, 2) - 1))
    tr3 = moving_average_3d(x3, min(k3, isodd(size(x3, 2)) ? size(x3, 2) : size(x3, 2) - 1))
    se1 = x1 .- tr1
    se2 = x2 .- tr2
    se3 = x3 .- tr3

    se1_2d = reshape(permutedims(se1, (2, 1, 3)), size(se1, 2), :)
    se2_2d = reshape(permutedims(se2, (2, 1, 3)), size(se2, 2), :)
    se3_2d = reshape(permutedims(se3, (2, 1, 3)), size(se3, 2), :)
    tr1_2d = reshape(permutedims(tr1, (2, 1, 3)), size(tr1, 2), :)
    tr2_2d = reshape(permutedims(tr2, (2, 1, 3)), size(tr2, 2), :)
    tr3_2d = reshape(permutedims(tr3, (2, 1, 3)), size(tr3, 2), :)

    se1_m_2d, st_se1 = m.seasonal_mixer_1(se1_2d, ps.seasonal_mixer_1, st.seasonal_mixer_1)
    se2_m_2d, st_se2 = m.seasonal_mixer_2(se2_2d, ps.seasonal_mixer_2, st.seasonal_mixer_2)
    se3_m_2d, st_se3 = m.seasonal_mixer_3(se3_2d, ps.seasonal_mixer_3, st.seasonal_mixer_3)
    tr1_m_2d, st_tr1 = m.trend_mixer_1(tr1_2d, ps.trend_mixer_1, st.trend_mixer_1)
    tr2_m_2d, st_tr2 = m.trend_mixer_2(tr2_2d, ps.trend_mixer_2, st.trend_mixer_2)
    tr3_m_2d, st_tr3 = m.trend_mixer_3(tr3_2d, ps.trend_mixer_3, st.trend_mixer_3)

    se1_m = permutedims(reshape(se1_m_2d, size(se1, 2), f, bsz), (2, 1, 3))
    se2_m = permutedims(reshape(se2_m_2d, size(se2, 2), f, bsz), (2, 1, 3))
    se3_m = permutedims(reshape(se3_m_2d, size(se3, 2), f, bsz), (2, 1, 3))
    tr1_m = permutedims(reshape(tr1_m_2d, size(tr1, 2), f, bsz), (2, 1, 3))
    tr2_m = permutedims(reshape(tr2_m_2d, size(tr2, 2), f, bsz), (2, 1, 3))
    tr3_m = permutedims(reshape(tr3_m_2d, size(tr3, 2), f, bsz), (2, 1, 3))

    se_fused =
        se1_m .+ upsample_repeat_to(se2_m, m.seq_len, 2) .+
        upsample_repeat_to(se3_m, m.seq_len, 4)
    tr_fused =
        tr1_m .+ upsample_repeat_to(tr2_m, m.seq_len, 2) .+
        upsample_repeat_to(tr3_m, m.seq_len, 4)
    x_fused = se_fused .+ tr_fused

    ch_in = reshape(x_fused, f, :)
    ch_out, st_ch = m.channel_mixer(ch_in, ps.channel_mixer, st.channel_mixer)
    x_mixed = reshape(ch_out, f, m.seq_len, bsz) .+ x_fused

    flat = reshape(x_mixed, f * m.seq_len, bsz)
    y, st_head = m.head(flat, ps.head, st.head)

    st = merge(
        st,
        (
            seasonal_mixer_1 = st_se1,
            seasonal_mixer_2 = st_se2,
            seasonal_mixer_3 = st_se3,
            trend_mixer_1 = st_tr1,
            trend_mixer_2 = st_tr2,
            trend_mixer_3 = st_tr3,
            channel_mixer = st_ch,
            head = st_head,
        ),
    )
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
    elseif model_type == :TimeSeriesPatchTST
        patch_len = get(config, :patch_len, 16)
        stride = get(config, :stride, 8)
        d_model = get(config, :d_model, 64)
        n_heads = get(config, :n_heads, 4)
        n_layers = get(config, :n_layers, 2)
        ff_dim = get(config, :ff_dim, 128)
        return TimeSeriesPatchTST(
            config.input_dim,
            config.seq_len,
            config.out_dim;
            patch_len = patch_len,
            stride = stride,
            d_model = d_model,
            n_heads = n_heads,
            n_layers = n_layers,
            ff_dim = ff_dim,
        )
    elseif model_type == :TimeSeriesTimeMixerPP
        hidden_dims = get(config, :hidden_dim, 128)
        kernel_sizes = Tuple(get(config, :kernel_sizes, (5, 13, 25)))
        length(kernel_sizes) == 3 || error(
            "TimeSeriesTimeMixerPP expects exactly 3 kernel_sizes, got $(kernel_sizes)",
        )
        return TimeSeriesTimeMixerPP(
            config.input_dim,
            config.seq_len,
            config.out_dim;
            hidden_dims = hidden_dims,
            kernel_sizes = kernel_sizes,
        )
    else
        error("Unknown model_type=$(model_type)")
    end
end
