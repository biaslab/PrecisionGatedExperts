@model function multivariate_hierarchical_model(
    n_forecasters,
    n_obs,
    features,
    predictions,
    y,
    priors,
)
    local w, z, β, γ, τ, ρ

    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        τ[i] ~ priors[:τ][i]
        ρ[i] ~ priors[:ρ][i]
    end

    for j = 1:n_obs
        for i = 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i]) where {meta = LowRankMeta()}
            β[i, j] ~ GammaShapeRate(1.0, ρ[i])
            z[i, j] ~ Log(β[i, j])
            γ[i, j] ~ GammaShapeRate(priors[:α], β[i, j])
            y[j] ~ MvNormalMeanScalePrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function multivariate_hierarchical_constraints()
    q(w, z, β, γ, τ, ρ) = q(w)q(z, β)q(γ)q(τ)q(ρ)
    q(
        z,
    )::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(
        β,
    )::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    # q(w) :: SubsampleFormConstraint(100) # This will subsample messages inside the product
end

@initialization function multivariate_hierarchical_init(priors)
    q(w) = priors[:w]
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(β) = GammaShapeScale(1.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = priors[:τ]
    q(ρ) = priors[:ρ]
end
