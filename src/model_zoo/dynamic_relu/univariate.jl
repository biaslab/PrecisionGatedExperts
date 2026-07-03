using ClosedFormExpectations: EnzymeBackend

@model function univariate_dynamic_relu_ensemble(
    n_forecasters,
    n_obs,
    features,
    predictions,
    y,
    priors,
)
    local w, z, γ, τ

    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
    end

    for j = 1:n_obs
        for i = 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            γ[i, j] ~ ReLU(z[i, j])
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function univariate_dynamic_relu_ensemble_constraints(priors, prediction)
    q(w, z, γ, τ) = q(w)q(z, γ)q(τ)
    q(
        z,
    )::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())),
    )
    q(
        γ,
    )::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())),
    )
    if prediction
        for (i, prior) in enumerate(deepcopy(priors[:w]))
            q(w[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
        for (i, prior) in enumerate(priors[:τ])
            q(τ[i])::RxInfer.FixedMarginalFormConstraint(prior)
        end
    end
end

@initialization function univariate_dynamic_relu_ensemble_init(priors)
    q(w) = deepcopy(priors[:w])
    q(z) = GammaShapeScale(2.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]
end
