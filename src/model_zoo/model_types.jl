# model_type: static | dynamic | hierarchical
struct Static end

function _break_symmetry_means(n_features::Int, n_forecasters::Int, strength::Float64)
    means = Matrix{Float64}(undef, n_features, n_forecasters)
    for i = 1:n_forecasters
        for k = 1:n_features
            means[k, i] = strength * sin(0.73 * (i - 1) + 1.17 * (k - 1))
        end
    end
    # Keep feature-wise means centered so we only break symmetry across experts.
    for k = 1:n_features
        means[k, :] .-= mean(means[k, :])
    end
    return means
end

function parse_mvn_mean_scale_precision_priors(
    cfg::Dict,
    n_forecasters::Int;
    prior_name::String,
)
    n_features = cfg["n_features"]
    scale = cfg["scale"]
    prior_type = cfg["type"]
    prior_type == "MvNormalMeanScalePrecision" || error(
        "Unknown $(prior_name) prior type: $(prior_type). Supported: MvNormalMeanScalePrecision",
    )

    break_symmetry = get(cfg, "break_symmetry_prior", false)
    if !break_symmetry
        return [
            MvNormalMeanScalePrecision(zeros(n_features), scale) for _ = 1:n_forecasters
        ]
    end

    strength = Float64(get(cfg, "break_symmetry_strength", 0.01))
    strength > 0.0 || error("$(prior_name).break_symmetry_strength must be > 0")
    @info "Using break-symmetry prior means" prior = prior_name n_forecasters n_features strength
    means = _break_symmetry_means(n_features, n_forecasters, strength)
    return [MvNormalMeanScalePrecision(vec(means[:, i]), scale) for i = 1:n_forecasters]
end

function parse_priors(::Static, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    γ_cfg = cfg["γ"]
    shape = γ_cfg["shape"]
    rate = γ_cfg["rate"]
    priors[:γ] = [GammaShapeRate(shape, rate) for _ = 1:n_forecasters]

    return priors
end


struct Dynamic end

function parse_priors(::Dynamic, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    shape = τ_cfg["shape"]
    rate = τ_cfg["rate"]
    priors[:τ] = [GammaShapeRate(shape, rate) for _ = 1:n_forecasters]

    β_cfg = cfg["β"]
    shape = β_cfg["shape"]
    rate = β_cfg["rate"]
    priors[:β] = [GammaShapeRate(shape, rate) for _ = 1:n_forecasters]

    w_cfg = cfg["w"]
    priors[:w] =
        parse_mvn_mean_scale_precision_priors(w_cfg, n_forecasters; prior_name = "w")

    return priors
end

struct Hierarchical end

function parse_priors(::Hierarchical, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    shape = τ_cfg["shape"]
    rate = τ_cfg["rate"]
    priors[:τ] = [GammaShapeRate(shape, rate) for _ = 1:n_forecasters]

    ρ_cfg = cfg["ρ"]
    shape = ρ_cfg["shape"]
    rate = ρ_cfg["rate"]
    priors[:ρ] = [GammaShapeRate(shape, rate) for _ = 1:n_forecasters]

    α_cfg = cfg["α"]
    value = α_cfg["value"]
    priors[:α] = value

    w_cfg = cfg["w"]
    priors[:w] =
        parse_mvn_mean_scale_precision_priors(w_cfg, n_forecasters; prior_name = "w")

    return priors
end

struct DynamicNoisyObservations end

function parse_priors(::DynamicNoisyObservations, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    priors[:τ] = [GammaShapeRate(τ_cfg["shape"], τ_cfg["rate"]) for _ = 1:n_forecasters]

    β_cfg = cfg["β"]
    priors[:β] = [GammaShapeRate(β_cfg["shape"], β_cfg["rate"]) for _ = 1:n_forecasters]

    κ_cfg = cfg["κ"]
    priors[:κ] = GammaShapeRate(κ_cfg["shape"], κ_cfg["rate"])

    w_cfg = cfg["w"]
    priors[:w] =
        parse_mvn_mean_scale_precision_priors(w_cfg, n_forecasters; prior_name = "w")

    return priors
end

struct Deep end

function parse_priors(::Deep, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    priors[:τ] = [GammaShapeRate(τ_cfg["shape"], τ_cfg["rate"]) for _ = 1:n_forecasters]

    ρ_cfg = cfg["ρ"]
    priors[:ρ] = [GammaShapeRate(ρ_cfg["shape"], ρ_cfg["rate"]) for _ = 1:n_forecasters]

    priors[:α] = cfg["α"]["value"]

    w_cfg = cfg["w"]
    priors[:w] =
        parse_mvn_mean_scale_precision_priors(w_cfg, n_forecasters; prior_name = "w")

    v_cfg = cfg["v"]
    priors[:v] =
        parse_mvn_mean_scale_precision_priors(v_cfg, n_forecasters; prior_name = "v")

    return priors
end
