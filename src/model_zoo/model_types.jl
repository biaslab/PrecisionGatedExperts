# model_type: static | dynamic | hierarchical
struct Static end

function parse_priors(::Static, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    γ_cfg = cfg["γ"]
    shape = γ_cfg["shape"]
    rate  = γ_cfg["rate"]
    priors[:γ] = [GammaShapeRate(shape, rate) for _ in 1:n_forecasters]

    return priors
end


struct Dynamic end

function parse_priors(::Dynamic, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    shape = τ_cfg["shape"]
    rate  = τ_cfg["rate"]
    priors[:τ] = [GammaShapeRate(shape, rate) for _ in 1:n_forecasters]

    β_cfg = cfg["β"]
    shape = β_cfg["shape"]
    rate  = β_cfg["rate"]
    priors[:β] = [GammaShapeRate(shape, rate) for _ in 1:n_forecasters]

    w_cfg      = cfg["w"]
    n_features = w_cfg["n_features"]
    scale      = w_cfg["scale"]
    w_type     = w_cfg["type"]
    if w_type == "MvNormalMeanScalePrecision"
        priors[:w] = [MvNormalMeanScalePrecision(zeros(n_features), scale) for _ in 1:n_forecasters]
    else
        error("Unknown w prior type: $w_type. Supported: MvNormalMeanScalePrecision")
    end

    return priors
end

struct Hierarchical end

function parse_priors(::Hierarchical, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()
    
    τ_cfg = cfg["τ"]
    shape = τ_cfg["shape"]
    rate  = τ_cfg["rate"]
    priors[:τ] = [GammaShapeRate(shape, rate) for _ in 1:n_forecasters]

    ρ_cfg = cfg["ρ"]
    shape = ρ_cfg["shape"]
    rate  = ρ_cfg["rate"]
    priors[:ρ] = [GammaShapeRate(shape, rate) for _ in 1:n_forecasters]

    α_cfg = cfg["α"]
    value = α_cfg["value"]
    priors[:α] = value

    w_cfg      = cfg["w"]
    n_features = w_cfg["n_features"]
    scale      = w_cfg["scale"]
    w_type     = w_cfg["type"]
    if w_type == "MvNormalMeanScalePrecision"
        priors[:w] = [MvNormalMeanScalePrecision(zeros(n_features), scale) for _ in 1:n_forecasters]
    else
        error("Unknown w prior type: $w_type. Supported: MvNormalMeanScalePrecision")
    end

    return priors
end