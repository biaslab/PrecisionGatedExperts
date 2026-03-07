struct Dynamic <: ModelType end

create_based_on_symbol(::Val{:dynamic}) = Dynamic()
model_type_name(::Dynamic) = "dynamic"

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

function extract_prediction_priors(::Dynamic, saved)
    # Reconstruct GammaShapeRate from JLD2 (may deserialize as ReconstructedStatic)
    _to_gamma(d) = GammaShapeRate(d.a, d.b)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :τ => map(_to_gamma, saved["τ_posteriors"]),
        :β => map(_to_gamma, saved["β_posteriors"]),
    )
end
