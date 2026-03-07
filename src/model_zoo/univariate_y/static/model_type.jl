struct Static <: ModelType end

create_based_on_symbol(::Val{:static}) = Static()
model_type_name(::Static) = "static"

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

function parse_priors(::Static, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    γ_cfg = cfg["γ"]
    shape = γ_cfg["shape"]
    rate = γ_cfg["rate"]
    priors[:γ] = [GammaShapeRate(shape, rate) for _ = 1:n_forecasters]

    return priors
end

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

function extract_prediction_priors(::Static, saved)
    # Reconstruct GammaShapeRate from JLD2 (may deserialize as ReconstructedStatic)
    γ_raw = saved["γ_posteriors"]
    γ = map(d -> GammaShapeRate(d.a, d.b), γ_raw)
    return Dict{Symbol,Any}(:γ => γ)
end

# Static models don't use features
function _before_rxinfer_features(
    ::ExperimentSpecifier{T,Static},
    feature_type,
    X,
    _,
) where {T}
    return nothing
end
