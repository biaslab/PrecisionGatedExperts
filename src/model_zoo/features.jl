using FFTW

# feature_type: simple | fft
struct SimpleFeatures end

struct FFTFeatures
    n_harmonics::Int
end
FFTFeatures() = FFTFeatures(3)

function parse_feature_type(s::String)
    s == "simple" && return SimpleFeatures()
    startswith(s, "fft") || error("Unknown feature_type: $s")
    # "fft" or "fft:5" (number of harmonics)
    parts = split(s, ':')
    length(parts) == 1 && return FFTFeatures()
    return FFTFeatures(parse(Int, parts[2]))
end

function make_features(::SimpleFeatures, X_scaled)
    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j = 1:n
        x_last = Float64.(X_scaled[:, end, j])
        x_last_cos = map(cos, X_scaled[:, end, j])
        x_last_sin = map(sin, X_scaled[:, end, j])
        feats[j] = vcat(1.0, x_last, x_last_cos, x_last_sin)
    end
    return feats
end

function make_features(ft::FFTFeatures, X_scaled)
    F, W, n = size(X_scaled)
    K = ft.n_harmonics
    feats = Vector{Vector{Float64}}(undef, n)
    for j = 1:n
        x_last = Float64.(X_scaled[:, end, j])
        fft_part = Vector{Float64}()
        for i = 1:F
            signal = Float64.(X_scaled[i, :, j])
            c = rfft(signal)
            n_coeffs = min(K + 1, length(c))  # +1 to skip DC
            for k = 2:n_coeffs                 # skip DC (index 1)
                push!(fft_part, real(c[k]))
                push!(fft_part, imag(c[k]))
            end
        end
        feats[j] = vcat(1.0, x_last, fft_part)
    end
    return feats
end
