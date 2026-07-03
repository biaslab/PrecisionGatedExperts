# Variational MP: mean-field rules with LowRankMeta

using LinearAlgebra: dot

@rule ReactiveMP.softdot(:y, Marginalisation) (
    q_θ::Any,
    q_x::Any,
    q_γ::Any,
    meta::LowRankMeta,
) = begin
    return @call_rule SoftDot(:y, Marginalisation) (q_θ = q_θ, q_x = q_x, q_γ = q_γ)
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
    q_θ::PointMass,
    q_x::Any,
    meta::LowRankMeta,
) = begin
    my = mean(q_y)
    Vy = var(q_y)
    θ = mean(q_θ)
    mx = mean(q_x)
    Vx = cov(q_x)
    s = dot(θ, mx)
    β = (Vy + my^2) / 2 - my * s + (dot(θ, Vx, θ) + s^2) / 2
    return GammaShapeRate(3 // 2, β)
end

@rule ReactiveMP.softdot(:γ, Marginalisation) (
    q_y::Any,
    q_θ::Any,
    q_x::Any,
    meta::LowRankMeta,
) = begin
    return @call_rule SoftDot(:γ, Marginalisation) (q_y = q_y, q_θ = q_θ, q_x = q_x)
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
