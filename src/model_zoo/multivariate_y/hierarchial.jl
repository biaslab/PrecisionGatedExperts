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

struct SubsampleFormConstraint <: AbstractFormConstraint 
    subsample_size::Int
end

ReactiveMP.default_form_check_strategy(::SubsampleFormConstraint) = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::SubsampleFormConstraint) = GenericProd()

function ReactiveMP.constrain_form(constraint::SubsampleFormConstraint, distribution::BayesBase.ProductOf)
    prior = distribution.left
    likelihood = distribution.right
    @assert prior isa ExponentialFamily.MvNormalMeanScalePrecision
    @assert likelihood isa BayesBase.LinearizedProductOf{<:LowRankNormalWeightedMeanPrecision}
    subsample_size = constraint.subsample_size
    random_subsample_from_messages = sample(likelihood.vector, subsample_size; replace = false)
    result = prod(GenericProd(), prior, random_subsample_from_messages[1])
    T = typeof(result)
    for i in 2:subsample_size
        result = prod(GenericProd(), result, random_subsample_from_messages[i])::T
    end
    return result
end

@meta function multivariate_hierarchical_meta()
    # (bvdmitri) This enables product from right to left
    # Though I didn't implement the proper rules
    # Instead it would result into a ProductOf structure 
    # Which then being processed by the `constrain_form` above 
    # I think it is better at the current stage since in this case we get access 
    # to all messages at once and it also allows us to parallelize the product easier
    # w -> (marginal_prod_strategy = ReactiveMP.FoldRightProdStrategy(), )
end
