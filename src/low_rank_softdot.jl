import BayesBase: PointMass
using ReactiveMP
using ReactiveMP: score, AverageEnergy, SoftDot
using LinearAlgebra: BLAS

using LowRankMatrices

export LowRankMeta

struct LowRankMeta end

# Variational MP: Mean-field with Low Rank Meta
@rule ReactiveMP.softdot(:y, Marginalisation) (
    q_θ::Any,
    q_x::Any,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    return @call_rule SoftDot(:y, Marginalisation) (q_θ = q_θ, q_x = q_x, q_γ = q_γ)
end

@rule ReactiveMP.softdot(:y, Marginalisation) (
    q_θ::Any,
    m_x::Any,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    return @call_rule SoftDot(:y, Marginalisation) (q_θ = q_θ, m_x = m_x, q_γ = q_γ)
end

@rule ReactiveMP.softdot(:θ, Marginalisation) (
    q_y::Any,
    q_x::PointMass,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    my = mean(q_y)
    mx = mean(q_x)
    mγ = mean(q_γ)
    return LowRankNormalWeightedMeanPrecision(mγ * mx * my, mx, mγ)
end

@rule ReactiveMP.softdot(:γ, Marginalisation) (
    q_y::Any,
    q_θ::Any,
    q_x::Any,
    meta::LowRankMeta,
) = begin
    return @call_rule SoftDot(:γ, Marginalisation) (q_y = q_y, q_θ = q_θ, q_x = q_x)
end

@rule ReactiveMP.softdot(:γ, Marginalisation) (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::Any,
    meta::LowRankMeta,
) = begin
    return @call_rule SoftDot(:γ, Marginalisation) (q_y_x = q_y_x, q_θ = q_θ)
end

@rule ReactiveMP.softdot(:x, Marginalisation) (
    q_y::Any,
    q_θ::PointMass,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    my = mean(q_y)
    mθ = mean(q_θ)
    mγ = mean(q_γ)
    return LowRankNormalWeightedMeanPrecision(mγ * mθ * my, mθ, mγ)
end

@rule ReactiveMP.softdot(:x, Marginalisation) (
    m_y::UnivariateNormalDistributionsFamily,
    q_θ::PointMass,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    my, Vy = mean_cov(m_y)
    mθ = mean(q_θ)
    mγ = mean(q_γ)
    scale = inv(ReactiveMP.add_transition(Vy, inv(mγ)))
    return LowRankNormalWeightedMeanPrecision(scale * mθ * my, mθ, scale)
end

# Average Energy: delegate to default SoftDot (no low-rank specialization)
@average_energy ReactiveMP.softdot (
    q_y::Any,
    q_θ::Any,
    q_x::Any,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    return score(AverageEnergy(), SoftDot, Val{(:y, :θ, :x, :γ)}(), marginals, nothing)
end

@average_energy ReactiveMP.softdot (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::Any,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    return score(AverageEnergy(), SoftDot, Val{(:y_x, :θ, :γ)}(), marginals, nothing)
end
