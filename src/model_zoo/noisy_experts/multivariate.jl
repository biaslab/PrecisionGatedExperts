# ---------------------------------------------------------------------------
# Multivariate StrangeMissingMeta rules for MvNormalMeanScalePrecision
# ---------------------------------------------------------------------------

# 1. :μ when y message is Uninformative → propagate Uninformative upward
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
    m_out::Uninformative,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return Uninformative()
end

# 2. :μ when y message is a proper MvNormal → delegate to standard rule
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
    m_out::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
        m_out = m_out, q_γ = q_γ,
    )
end

# 3. :out_μ joint marginal when y is Uninformative → return missing
#    This prevents divergence in γ (same trick as univariate)
@marginalrule MvNormalMeanScalePrecision(:out_μ) (
    m_out::Uninformative,
    m_μ::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return missing
end

# 4. :γ (precision) from structured joint marginal → delegate
@rule MvNormalMeanScalePrecision(:γ, Marginalisation) (
    q_out_μ::MultivariateNormalDistributionsFamily,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule MvNormalMeanScalePrecision(:γ, Marginalisation) (
        q_out_μ = q_out_μ,
    )
end

# 5. :out (observation) message → delegate to standard rule
@rule MvNormalMeanScalePrecision(:out, Marginalisation) (
    m_μ::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (
        m_μ = m_μ, q_γ = q_γ,
    )
end

# ---------------------------------------------------------------------------
# Training model: y is a data variable (observed)
# ---------------------------------------------------------------------------

@model function multivariate_noisy_experts(
    n_forecasters,
    n_obs,
    features,
    predictions,
    y,
    priors,
)
    local w, z, γ, τ, β, κ, pred

    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        β[i] ~ priors[:β][i]
        κ[i] ~ priors[:κ][i]
    end

    for j = 1:n_obs
        for i = 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j])

            pred[i, j] ~ MvNormalMeanScalePrecision(predictions[i, j], κ[i])
            y[j] ~ MvNormalMeanScalePrecision(pred[i, j], γ[i, j])
        end
    end
end

# ---------------------------------------------------------------------------
# Prediction model: y is a latent variable with Uninformative()
# ---------------------------------------------------------------------------

@model function multivariate_noisy_experts_prediction(
    n_forecasters,
    n_obs,
    features,
    predictions,
    priors,
)
    local w, z, γ, τ, β, κ, pred, y

    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        β[i] ~ priors[:β][i]
        κ[i] ~ priors[:κ][i]
    end

    for j = 1:n_obs
        for i = 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j])

            pred[i, j] ~ MvNormalMeanScalePrecision(predictions[i, j], κ[i])
            y[j] ~ MvNormalMeanScalePrecision(pred[i, j], γ[i, j]) where {meta = StrangeMissingMeta()}
        end
        y[j] ~ Uninformative()
    end
end

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

@constraints function multivariate_noisy_experts_constraints(priors, prediction)
    if prediction
        q(w, z, γ, τ, β, κ, pred, y) = q(w)q(z, γ)q(τ)q(β)q(κ)q(y, pred)
    else
        q(w, z, γ, τ, β, κ, pred) = q(w)q(z, γ)q(τ)q(β)q(κ)q(pred)
    end
    q(
        z,
    )::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(
        γ,
    )::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    if prediction
        for (i, prior) in enumerate(deepcopy(priors[:w]))
            q(w[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:τ])
            q(τ[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:β])
            q(β[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:κ])
            q(κ[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
    end
end

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

@initialization function multivariate_noisy_experts_init(priors)
    q(w) = deepcopy(priors[:w])
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = priors[:τ]
    q(β) = priors[:β]
    q(κ) = priors[:κ]
    q(pred) = MvNormalMeanScalePrecision(zeros(priors[:output_dim]), 1.0)
end
