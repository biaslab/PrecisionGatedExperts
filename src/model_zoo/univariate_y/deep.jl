@model function deep_model(n_forecasters, n_obs, features, predictions, y, priors)
    local w, v, z, h, β, γ, κ, τ, ρ

    for i = 1:n_forecasters
        w[i] ~ priors[:w][i]
        v[i] ~ priors[:v][i]
        τ[i] ~ priors[:τ][i]
        ρ[i] ~ priors[:ρ][i]
    end

    for j = 1:n_obs
        for i = 1:n_forecasters
            # Level 1: contextual precision κ = exp(v'features)
            h[i, j] ~ softdot(features[j], v[i], τ[i]) where {meta = LowRankMeta()}
            κ[i, j] ~ GammaShapeRate(1.0, ρ[i])
            h[i, j] ~ Log(κ[i, j])

            # Level 2: contextual expert weight β = exp(w'features) with precision κ
            z[i, j] ~ softdot(features[j], w[i], κ[i, j]) where {meta = LowRankMeta()}
            β[i, j] ~ GammaShapeRate(1.0, ρ[i])
            z[i, j] ~ Log(β[i, j])

            γ[i, j] ~ GammaShapeRate(priors[:α], β[i, j])
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function deep_constraints()
    q(w, v, z, h, β, κ, γ, τ, ρ) = q(w)q(v)q(h, κ)q(z, β)q(γ)q(τ)q(ρ)
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
    q(
        h,
    )::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(
        κ,
    )::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
end

@initialization function deep_init(priors)
    q(w) = priors[:w]
    q(v) = priors[:v]
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(h) = NormalMeanVariance(0.0, 1.0)
    q(β) = GammaShapeScale(1.0, 1.0)
    q(κ) = GammaShapeScale(1.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = priors[:τ]
    q(ρ) = priors[:ρ]
end
