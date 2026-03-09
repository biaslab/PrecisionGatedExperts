struct DynamicDiagonal <: ModelType end

create_based_on_symbol(::Val{:dynamic_diagonal}) = DynamicDiagonal()
model_type_name(::DynamicDiagonal) = "dynamic_diagonal"

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

function parse_priors(::DynamicDiagonal, cfg::Dict, n_forecasters::Int)
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
    priors[:w] = _parse_diagonal_mvn_priors(w_cfg, n_forecasters)

    return priors
end

"""
Parse w priors as `MvNormalWeightedMeanPrecision{..., Diagonal}` so that the
product rule dispatches to the diagonal ADF path (see `low_rank_diagonal.jl`).
"""
function _parse_diagonal_mvn_priors(cfg::Dict, n_forecasters::Int)
    n_features = cfg["n_features"]
    scale = Float64(cfg["scale"])

    break_symmetry = get(cfg, "break_symmetry_prior", false)
    if !break_symmetry
        return [
            MvNormalWeightedMeanPrecision(
                zeros(Float64, n_features),
                Diagonal(fill(scale, n_features)),
            ) for _ = 1:n_forecasters
        ]
    end

    strength = Float64(get(cfg, "break_symmetry_strength", 0.01))
    means = _break_symmetry_means(n_features, n_forecasters, strength)
    return [
        MvNormalWeightedMeanPrecision(
            scale .* vec(means[:, i]),
            Diagonal(fill(scale, n_features)),
        ) for i = 1:n_forecasters
    ]
end

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

function extract_prediction_priors(::DynamicDiagonal, saved)
    _to_gamma(d) = GammaShapeRate(d.a, d.b)

    # Reconstruct w posteriors — they may be MvNormalWeightedMeanPrecision with
    # Diagonal precision that JLD2 deserializes correctly, or we may need to
    # rebuild from the stored fields.
    w_raw = saved["w_posteriors"]
    w_priors = map(w_raw) do w
        xi = Vector{Float64}(weightedmean(w))
        d = Vector{Float64}(diag(invcov(w)))
        MvNormalWeightedMeanPrecision(xi, Diagonal(d))
    end

    return Dict{Symbol,Any}(
        :w => w_priors,
        :τ => map(_to_gamma, saved["τ_posteriors"]),
        :β => map(_to_gamma, saved["β_posteriors"]),
    )
end
