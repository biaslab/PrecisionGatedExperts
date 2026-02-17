using LinearAlgebra: BLAS
using BayesBase
using ExponentialFamily
using LowRankMatrices

import Distributions: logdetcov, distrname, AbstractMvNormal
import LinearAlgebra: diag
import Base: ndims, precision, length, size, eltype, prod

export LowRankNormalWeightedMeanPrecision

# ─── Compact rank-1 message ──────────────────────────────────────────────────

"""
    LowRankNormalWeightedMeanPrecision(xi, u, scale)

Compact rank-1 Normal message in natural parametrization:

    ξ = xi               (weighted mean vector, O(n))
    Λ = scale * u * u'   (rank-1 precision, stored as O(n))

Produced by softdot rules. During message product with a prior or accumulator,
the rank-1 precision is applied via in-place BLAS.ger! rank-1 updates onto
a dense MvNormalWeightedMeanPrecision.
"""
struct LowRankNormalWeightedMeanPrecision{T <: Real, V <: AbstractVector{T}} <: AbstractMvNormal
    xi::V
    u::V
    scale::T
end

const LR = LowRankNormalWeightedMeanPrecision

Distributions.distrname(::LR) = "LowRankNormalWeightedMeanPrecision"
Base.eltype(::LR{T}) where {T} = T
Base.length(d::LR) = length(d.xi)
Base.ndims(d::LR) = length(d)
Base.size(d::LR) = (length(d),)

BayesBase.weightedmean(d::LR) = d.xi
BayesBase.invcov(d::LR) = d.scale * LowRankMatrix(d.u, d.u)
Base.precision(d::LR) = invcov(d)


# ─── Product rules ────────────────────────────────────────────────────────────
# Strategy: LR messages stay compact (O(n) each). When meeting a prior or
# an existing MvNormalWeightedMeanPrecision with dense Matrix precision,
# the rank-1 update is applied in-place via BLAS.ger!.
#
# No prod(LR, LR) rule is defined — this keeps the foldl type-stable.
# The prior comes first, creates a dense MvNormalWeightedMeanPrecision,
# and all subsequent LR messages update it in-place.

# --- MvNormalWeightedMeanPrecision × LR → in-place ger! ---
BayesBase.default_prod_rule(::Type{<:MvNormalWeightedMeanPrecision}, ::Type{<:LR}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalWeightedMeanPrecision{T, V, M}, right::LR) where {T, V, M <: Matrix}
    weightedmean(left) .+= right.xi
    BLAS.ger!(right.scale, right.u, right.u, invcov(left))
    return left
end

BayesBase.default_prod_rule(::Type{<:LR}, ::Type{<:MvNormalWeightedMeanPrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::LR, right::MvNormalWeightedMeanPrecision)
    return prod(PreserveTypeProd(Distribution), right, left)
end

# --- MvNormalMeanScalePrecision × LR → allocate dense MvNormalWeightedMeanPrecision ---
BayesBase.default_prod_rule(::Type{<:MvNormalMeanScalePrecision}, ::Type{<:LR}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::MvNormalMeanScalePrecision, right::LR)
    n = length(right.xi)
    T = promote_type(eltype(mean(left)), eltype(right))
    γ = T(BayesBase.scale(left))
    Λ = zeros(T, n, n)
    @inbounds for i in 1:n
        Λ[i, i] = γ
    end
    BLAS.ger!(right.scale, right.u, right.u, Λ)
    xi = Vector{T}(weightedmean(left))
    xi .+= right.xi
    return MvNormalWeightedMeanPrecision(xi, Λ)
end

BayesBase.default_prod_rule(::Type{<:LR}, ::Type{<:MvNormalMeanScalePrecision}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::LR, right::MvNormalMeanScalePrecision)
    return prod(PreserveTypeProd(Distribution), right, left)
end
