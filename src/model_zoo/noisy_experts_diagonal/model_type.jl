struct NoisyExpertsDiagonal <: ModelType end

create_based_on_symbol(::Val{:noisy_experts_diagonal}) = NoisyExpertsDiagonal()
model_type_name(::NoisyExpertsDiagonal) = "noisy_experts_diagonal"

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

function parse_priors(::NoisyExpertsDiagonal, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()

    τ_cfg = cfg["τ"]
    priors[:τ] = [GammaShapeRate(τ_cfg["shape"], τ_cfg["rate"]) for _ = 1:n_forecasters]

    β_cfg = cfg["β"]
    priors[:β] = [GammaShapeRate(β_cfg["shape"], β_cfg["rate"]) for _ = 1:n_forecasters]

    κ_cfg = cfg["κ"]
    priors[:κ] = [GammaShapeRate(κ_cfg["shape"], κ_cfg["rate"]) for _ = 1:n_forecasters]

    w_cfg = cfg["w"]
    priors[:w] = _parse_diagonal_mvn_priors(w_cfg, n_forecasters)

    return priors
end

# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

function extract_prediction_priors(::NoisyExpertsDiagonal, saved)
    _to_gamma(d) = GammaShapeRate(d.a, d.b)

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
        :κ => map(_to_gamma, saved["κ_posteriors"]),
    )
end
