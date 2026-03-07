abstract type ModelType end

function _parse_model_type(s::String)
    model_symbol = Symbol(lowercase(s))
    create_based_on_symbol(Val(model_symbol))
end 

# model_type_name(::T) — defined per model type in its file

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
