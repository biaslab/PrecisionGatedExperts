# ---------------------------------------------------------------------------
# Free energy scoring for Uninformative node with observed (PointMass) marginal.
# E_q[-log Uninformative(x)] = 0 since log(const) = 0.
# ---------------------------------------------------------------------------

@average_energy Uninformative (q_out::PointMass,) = begin
    return 0.0
end

# BP message from Normal to Normal mean is uniformative when message is unformative
@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::Uninformative, q_τ::Any, ) = begin 
    return Uninformative()
end

# VMP rule
@marginalrule NormalMeanPrecision(:out_μ) (m_out::Uninformative, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    return missing
    # TODO: the rule below "is correct" however it gives the divergence in gamma because of the resulting message for \tau
    # If I put missing here then it will go a missing to tau and it will converge
    # xi_μ, W_μ = weightedmean_precision(m_μ)
    # W_bar = mean(q_τ)
    # W  = [W_bar -W_bar; -W_bar W_μ+W_bar]
    # xi = [zero(xi_μ); xi_μ]
    # return MvNormalWeightedMeanPrecision(xi, W)
end

# ---------------------------------------------------------------------------
# Training model: y is a data variable (observed)
# ---------------------------------------------------------------------------

@model function univariate_noisy_experts(
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

            # Two-level hierarchy (noisy experts):
            # pred[i,j] is the latent "true prediction" for expert i
            # predictions[i,j] (observed) is a noisy measurement with precision κ[i]
            # y[j] is the target with precision γ[i,j] (gating)
            pred[i, j] ~ NormalMeanPrecision(predictions[i, j], κ[i])
            y[j] ~ NormalMeanPrecision(pred[i, j], γ[i, j])
        end
    end
end

# ---------------------------------------------------------------------------
# Prediction model: y is a latent variable with Uninformative()
# NOT a data variable — this triggers BP via the custom @marginalrule,
# giving additive variance: Var(y) = 1/κ[i] + 1/γ[i,j]
# ---------------------------------------------------------------------------

@model function univariate_noisy_experts_prediction(
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

            pred[i, j] ~ NormalMeanPrecision(predictions[i, j], κ[i])
            y[j] ~ NormalMeanPrecision(pred[i, j], γ[i, j])
        end
        y[j] ~ Uninformative()
    end
end

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

@constraints function univariate_noisy_experts_constraints(priors, prediction)
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

@initialization function univariate_noisy_experts_init(priors)
    q(w) = deepcopy(priors[:w])
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = priors[:τ]
    q(β) = priors[:β]
    q(κ) = priors[:κ]
    q(pred) = NormalMeanVariance(0.0, 1.0)
end
