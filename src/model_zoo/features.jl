using FFTW
using Statistics
using Lux

# feature_type: simple | fft
struct SimpleFeatures end

struct WindowFeatures end

struct UniWindowFeatures end

struct AEFeatures end

struct VAEFeatures end

struct FFTFeatures
    n_harmonics::Int
end
FFTFeatures() = FFTFeatures(3)

const _LATENT_FEATURE_MODEL_CACHE = Dict{String,NamedTuple}()

_dataset_name(dataset) = dataset isa Val ? string(typeof(dataset).parameters[1]) : string(dataset)

function _find_latent_model_path(dataset, seq_len::Int, tag::String)
    models_dir = "models"
    isdir(models_dir) || error("Models directory not found: $(models_dir)")
    ds = _dataset_name(dataset)
    exact = joinpath(models_dir, "$(ds)_s$(seq_len)_$(tag)_enzyme.jld2")
    if isfile(exact)
        return exact
    end

    candidates = filter(
        f ->
            startswith(f, "$(ds)_s$(seq_len)_") &&
            occursin(tag, f) &&
            endswith(f, ".jld2"),
        readdir(models_dir),
    )
    isempty(candidates) && error(
        "No $(tag) model found for dataset=$(ds), seq_len=$(seq_len). " *
        "Expected $(exact) or similar name in $(models_dir).",
    )
    sort!(candidates)
    return joinpath(models_dir, last(candidates))
end

function _load_latent_model(kind::Symbol, dataset, seq_len::Int)
    key = string(kind, "|", _dataset_name(dataset), "|", seq_len)
    if haskey(_LATENT_FEATURE_MODEL_CACHE, key)
        return _LATENT_FEATURE_MODEL_CACHE[key]
    end

    tag = kind == :ae ? "AutoEncoder" : "VAE"
    path = _find_latent_model_path(dataset, seq_len, tag)
    saved = load_jld2_model(path)
    model = build_model(saved.model_type, saved.config)
    st = Lux.testmode(saved.states)
    loaded = (model = model, ps = saved.parameters, st = st, path = path)
    _LATENT_FEATURE_MODEL_CACHE[key] = loaded
    return loaded
end

function parse_feature_type(s::String)
    s == "simple" && return SimpleFeatures()
    if s == "window"
        return WindowFeatures()
    elseif s == "uniwindow"
        return UniWindowFeatures()
    elseif s == "simple"
        return SimpleFeatures()
    elseif s == "ae"
        return AEFeatures()
    elseif s == "vae"
        return VAEFeatures()
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

function make_features(::UniWindowFeatures, X_scaled, col_idx)
    _, W, n = size(X_scaled)
    feats = Vector{Vector{Float64}}(undef, n)
    for j = 1:n
        x_last = Float64.(X_scaled[col_idx, end, j])
        t_penultimate = max(1, W - 10)
        t_third_last = max(1, W - 20)
        t_fourth_last = max(1, W - 30)
        t_fifth_last = max(1, W - 40)
        t_six_last = max(1, W - 50)
        t_seventh_last = max(1, W - 60)
        t_eight_last = max(1, W - 70)
        t_nine_last = max(1, W - 80)
        x_penultimate = Float64.(X_scaled[col_idx, t_penultimate, j])
        x_third_last = Float64.(X_scaled[col_idx, t_third_last, j])
        x_fourth_last = Float64.(X_scaled[col_idx, t_fourth_last, j])
        x_fifth_last = Float64.(X_scaled[col_idx, t_fifth_last, j])
        t_six_last = Float64.(X_scaled[col_idx, t_six_last, j])
        t_seventh_last = Float64.(X_scaled[col_idx, t_seventh_last, j])
        t_eight_last = Float64.(X_scaled[col_idx, t_eight_last, j])
        t_nine_last = Float64.(X_scaled[col_idx, t_nine_last, j])
        feats[j] = vcat(
            1.0,
            x_last,
            x_penultimate,
            x_third_last,
            x_fourth_last,
            x_fifth_last,
            t_six_last,
            t_seventh_last,
            t_eight_last,
            t_nine_last
        )
    end
    return feats
end


function make_features(::VAEFeatures, X_scaled, dataset)
    seq_len = size(X_scaled, 2)
    loaded = _load_latent_model(:vae, dataset, seq_len)
    μ, _, _ = encode_vae(
        loaded.model,
        Float32.(X_scaled),
        loaded.ps,
        loaded.st,
    )

    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j = 1:n
        feats[j] = vcat(1.0, Float64.(Array(@view μ[:, j])))
    end
    return feats
end

function make_features(::AEFeatures, X_scaled, dataset)
    seq_len = size(X_scaled, 2)
    loaded = _load_latent_model(:ae, dataset, seq_len)
    z, _ = encode_latent(
        loaded.model,
        Float32.(X_scaled),
        loaded.ps,
        loaded.st,
    )

    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j = 1:n
        feats[j] = vcat(1.0, Float64.(Array(@view z[:, j])))
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
