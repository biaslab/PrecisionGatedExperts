using FFTW
using Statistics

# feature_type: simple | fft
struct SimpleFeatures end

struct FFTFeatures
    n_harmonics::Int
end
FFTFeatures() = FFTFeatures(3)

function parse_feature_type(s::String)
    s == "simple" && return SimpleFeatures()
    if s == "window"
        retunr WindowFeatures()
    elseif s == "simple"
        return SimpleFeatures()
    else
        startswith(s, "fft") || error("Unknown feature_type: $s")
    # "fft" or "fft:5" (number of harmonics)
        parts = split(s, ':')
        length(parts) == 1 && return FFTFeatures()
        n_harmonics = parse(Int, parts[2])
        n_harmonics > 0 || error("fft harmonics must be > 0, got $n_harmonics")
        return FFTFeatures(n_harmonics)
    end
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

function make_features(::WindowFeatures, X_scaled)
    _, W, n = size(X_scaled)
    feats = Vector{Vector{Float64}}(undef, n)
    for j = 1:n
        x_last = Float64.(X_scaled[:, end, j])
        t_penultimate = max(1, W - 10)
        t_third_last = max(1, W - 20)
        t_fourth_last = max(1, W - 30)
        t_fifth_last = max(1, W - 40)
        t_six_last = max(1, W - 50)
        t_seventh_last = max(1, W - 60)
        x_penultimate = Float64.(X_scaled[:, t_penultimate, j])
        x_third_last = Float64.(X_scaled[:, t_third_last, j])
        x_fourth_last = Float64.(X_scaled[:, t_fourth_last, j])
        x_fifth_last = Float64.(X_scaled[:, t_fifth_last, j])
        t_six_last = Float64.(X_scaled[:, t_six_last, j])
        t_seventh_last = Float64.(X_scaled[:, t_seventh_last, j])
        feats[j] = vcat(
            1.0,
            x_last,
            x_penultimate,
            x_third_last,
            x_fourth_last,
            x_fifth_last,
            t_six_last,
            t_seventh_last
        )
    end
    return feats
end



function make_features(ft::FFTFeatures, X_scaled)
    F, W, n = size(X_scaled)
    K = ft.n_harmonics
    w_scale = sqrt(Float64(W))
    feats = Vector{Vector{Float64}}(undef, n)

    for j = 1:n
        x_last = Float64.(X_scaled[:, end, j])
        fft_part = Vector{Float64}()

        for i = 1:F
            signal = Float64.(X_scaled[i, :, j])
            c = rfft(signal) ./ w_scale

            n_coeffs = min(K + 1, length(c)) # +1 to skip DC
            for k = 2:n_coeffs                # skip DC (index 1)
                re = real(c[k])
                ang = angle(c[k])
                push!(fft_part, isfinite(re) ? re : 0.0)
                push!(fft_part, isfinite(ang) ? ang : 0.0)
            end
        end

        # Normalize only the FFT-derived block to stabilize training scale.
        if !isempty(fft_part)
            μ_fft = mean(fft_part)
            σ_fft = std(fft_part)
            if isfinite(σ_fft) && σ_fft > eps(Float64)
                fft_part .= (fft_part .- μ_fft) ./ σ_fft
            else
                fill!(fft_part, 0.0)
            end
        end

        feats[j] = vcat(1.0, x_last, fft_part)
    end

    return feats
end
