@model function univariate_dynamic_ensemble(
    n_forecasters,
    n_obs,
    features,
    predictions,
    y,
    priors,
)
    local w, z, γ, τ, β

    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        β[i] ~ priors[:β][i]
    end

    for j = 1:n_obs
        for i = 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j])
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function univariate_dynamic_ensemble_constraints(priors, prediction)
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
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
        for (i, prior) in enumerate(priors[:w])
            q(w[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:τ])
            q(τ[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:β])
            q(β[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
    end
end

@initialization function univariate_dynamic_ensemble_init(priors)
    q(w) = priors[:w]
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = priors[:τ]
    q(β) = priors[:β]
end
