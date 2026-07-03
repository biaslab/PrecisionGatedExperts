using LinearAlgebra: BLAS, Diagonal, Symmetric, copytri!
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
the rank-1 precision is applied via in-place BLAS.syr! rank-1 updates onto
a dense MvNormalWeightedMeanPrecision.
"""
struct LowRankNormalWeightedMeanPrecision{T<:Real,V<:AbstractVector{T}}
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
# the rank-1 update is applied in-place via BLAS.syr!.
#
# The preferred path is prior first: it creates a dense
# MvNormalWeightedMeanPrecision, and subsequent LR messages update it in-place.
# RxInfer may still multiply likelihood messages before the prior, so LR × LR
# falls back to a dense MvNormalWeightedMeanPrecision accumulator.

# --- LR × LR → allocate dense MvNormalWeightedMeanPrecision ---
BayesBase.default_prod_rule(::Type{<:LR}, ::Type{<:LR}) = PreserveTypeProd(Distribution)

function BayesBase.prod(::PreserveTypeProd{Distribution}, left::LR, right::LR)
    n = length(left)
    length(right) == n ||
        throw(DimensionMismatch("Low-rank messages must have the same length"))

    T = promote_type(eltype(left), eltype(right))
    xi = Vector{T}(left.xi)
    xi .+= right.xi

    Λ = zeros(T, n, n)
    BLAS.syr!('U', T(left.scale), Vector{T}(left.u), Λ)
    BLAS.syr!('U', T(right.scale), Vector{T}(right.u), Λ)
    copytri!(Λ, 'U')

    return MvNormalWeightedMeanPrecision(xi, Λ)
end

# --- MvNormalWeightedMeanPrecision × LR → in-place syr! ---
BayesBase.default_prod_rule(::Type{<:MvNormalWeightedMeanPrecision}, ::Type{<:LR}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalWeightedMeanPrecision{T,V,M},
    right::LR,
) where {T,V,M<:Matrix}
    weightedmean(left) .+= right.xi
    Λ = invcov(left)
    BLAS.syr!('U', right.scale, right.u, Λ)
    copytri!(Λ, 'U')
    return left
end

function _dense_product(left::MvNormalWeightedMeanPrecision, right::LR)
    T = promote_type(eltype(weightedmean(left)), eltype(right))
    xi = Vector{T}(weightedmean(left))
    xi .+= right.xi

    Λ = Matrix{T}(Matrix(invcov(left)))
    BLAS.syr!('U', T(right.scale), Vector{T}(right.u), Λ)
    copytri!(Λ, 'U')

    return MvNormalWeightedMeanPrecision(xi, Λ)
end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalWeightedMeanPrecision{T,V,M},
    right::LR,
) where {T,V,M<:Diagonal}
    return _dense_product(left, right)
end

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalWeightedMeanPrecision{T,V,M},
    right::LR,
) where {T,V,M<:Symmetric}
    return _dense_product(left, right)
end

BayesBase.default_prod_rule(::Type{<:LR}, ::Type{<:MvNormalWeightedMeanPrecision}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::LR,
    right::MvNormalWeightedMeanPrecision,
)
    return prod(PreserveTypeProd(Distribution), right, left)
end

# --- MvNormalMeanScalePrecision × LR → allocate dense MvNormalWeightedMeanPrecision ---
BayesBase.default_prod_rule(::Type{<:MvNormalMeanScalePrecision}, ::Type{<:LR}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalMeanScalePrecision,
    right::LR,
)
    n = length(right.xi)
    T = promote_type(eltype(mean(left)), eltype(right))
    γ = T(BayesBase.scale(left))
    Λ = zeros(T, n, n)
    @inbounds for i = 1:n
        Λ[i, i] = γ
    end
    BLAS.syr!('U', right.scale, right.u, Λ)
    copytri!(Λ, 'U')
    xi = Vector{T}(weightedmean(left))
    xi .+= right.xi
    return MvNormalWeightedMeanPrecision(xi, Λ)
end

BayesBase.default_prod_rule(::Type{<:LR}, ::Type{<:MvNormalMeanScalePrecision}) =
    PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::LR,
    right::MvNormalMeanScalePrecision,
)
    return prod(PreserveTypeProd(Distribution), right, left)
end
