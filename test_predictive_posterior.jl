using RxInfer

@marginalrule NormalMeanPrecision(:out_μ) (m_out::Uninformative, m_μ::UnivariateNormalDistributionsFamily, q_τ::Any) = begin
    return missing
    xi_μ, W_μ = weightedmean_precision(m_μ)
    W_bar = mean(q_τ)
    W  = [W_bar -W_bar; -W_bar W_μ+W_bar]
    xi = [zero(xi_μ); xi_μ]
    return MvNormalWeightedMeanPrecision(xi, W)
end

@rule NormalMeanPrecision(:μ, Marginalisation) (m_out::Uninformative, q_τ::GammaShapeRate, ) = begin 
    return Uninformative()
end

@model function check_pred(x)
    κ ~ GammaShapeRate(1, 10)
    prediction ~ NormalMeanPrecision(x, κ)
    γ ~ GammaShapeRate(1, 1)
    y ~ NormalMeanPrecision(prediction, γ)
    y ~ Uninformative()
end

@constraints function pred_constraints()
    q(y, prediction, γ, κ) = q(y, prediction)q(γ)q(κ)
end

result = infer(
    model = check_pred(),
    data = (x=10,),
    constraints = pred_constraints(),
    initialization = @initialization(begin
        q(γ) = GammaShapeRate(1, 1)
        q(κ) = GammaShapeRate(1, 1)
    end),
    iterations = 20,
);

# @show result.posteriors[:prediction]
@show result.posteriors[:γ]
@show result.posteriors[:y]
