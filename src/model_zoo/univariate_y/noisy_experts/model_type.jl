struct NoisyExperts <: ModelType end

create_based_on_symbol(::Val{:noisy_experts}) = NoisyExperts()
model_type_name(::NoisyExperts) = "noisy_experts"

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

function parse_priors(::NoisyExperts, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    priors[:τ] = [GammaShapeRate(τ_cfg["shape"], τ_cfg["rate"]) for _ = 1:n_forecasters]

    β_cfg = cfg["β"]
    priors[:β] = [GammaShapeRate(β_cfg["shape"], β_cfg["rate"]) for _ = 1:n_forecasters]

    κ_cfg = cfg["κ"]
    priors[:κ] = [GammaShapeRate(κ_cfg["shape"], κ_cfg["rate"]) for _ = 1:n_forecasters]

    w_cfg = cfg["w"]
    priors[:w] =
        parse_mvn_mean_scale_precision_priors(w_cfg, n_forecasters; prior_name = "w")

    return priors
end

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

function extract_prediction_priors(::NoisyExperts, saved)
    _to_gamma(d) = GammaShapeRate(d.a, d.b)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :τ => map(_to_gamma, saved["τ_posteriors"]),
        :β => map(_to_gamma, saved["β_posteriors"]),
        :κ => map(_to_gamma, saved["κ_posteriors"]),
    )
end
