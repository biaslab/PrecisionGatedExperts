using LinearAlgebra: Diagonal, dot

# ─── LowRankDiagonalUpdate: rank-1 message for diagonal-precision accumulation ─
#
# Identical data to LowRankNormalWeightedMeanPrecision (xi, u, scale) but a
# distinct type so product rules dispatch to the diagonal ADF path instead of
# the dense BLAS.syr! path.

export LowRankUpdateDiagonal, LowRankDiagonalUpdate

struct LowRankUpdateDiagonal end

struct LowRankDiagonalUpdate{T<:Real,V<:AbstractVector{T}}
    xi::V
    u::V
    scale::T
end

const LRD = LowRankDiagonalUpdate

Distributions.distrname(::LRD) = "LowRankDiagonalUpdate"
Base.eltype(::LRD{T}) where {T} = T
Base.length(d::LRD) = length(d.xi)
Base.ndims(d::LRD) = length(d)
Base.size(d::LRD) = (length(d),)

BayesBase.weightedmean(d::LRD) = d.xi
BayesBase.invcov(d::LRD) = d.scale * LowRankMatrix(d.u, d.u)
Base.precision(d::LRD) = invcov(d)

# ─── Softdot VMP rules for LowRankUpdateDiagonal meta ─────────────────────────

@rule ReactiveMP.softdot(:y, Marginalisation) (
    q_θ::Any,
    q_x::Any,
    q_γ::Any,
    meta::LowRankUpdateDiagonal,
) = begin
    return @call_rule SoftDot(:y, Marginalisation) (q_θ = q_θ, q_x = q_x, q_γ = q_γ)
end

@rule ReactiveMP.softdot(:θ, Marginalisation) (
    q_y::Any,
    q_x::PointMass,
    q_γ::Any,
    meta::LowRankUpdateDiagonal,
) = begin
    my = mean(q_y)
    mx = mean(q_x)
    mγ = mean(q_γ)
    return LowRankDiagonalUpdate(mγ * mx * my, mx, mγ)
end

@rule ReactiveMP.softdot(:γ, Marginalisation) (
    q_y::Any,
    q_θ::Any,
    q_x::Any,
    meta::LowRankUpdateDiagonal,
) = begin
    return @call_rule SoftDot(:γ, Marginalisation) (q_y = q_y, q_θ = q_θ, q_x = q_x)
end

@rule ReactiveMP.softdot(:x, Marginalisation) (
    q_y::Any,
    q_θ::PointMass,
    q_γ::Any,
    meta::LowRankUpdateDiagonal,
) = begin
    my = mean(q_y)
    mθ = mean(q_θ)
    mγ = mean(q_γ)
    return LowRankDiagonalUpdate(mγ * mθ * my, mθ, mγ)
end

# Average Energy: delegate to default SoftDot
@average_energy ReactiveMP.softdot (
    q_y::Any,
    q_θ::Any,
    q_x::Any,
    q_γ::Any,
    meta::LowRankUpdateDiagonal,
) = begin
    return score(AverageEnergy(), SoftDot, Val{(:y, :θ, :x, :γ)}(), marginals, nothing)
end

# ─── Product rules: LRD into diagonal-precision MvNormalWeightedMeanPrecision ──
#
# Each LRD message is folded via Assumed Density Filtering (ADF):
#   1. Sherman–Morrison to recover the exact mean of (D + scale·uuᵀ)
#   2. Update diagonal precision:  d .+= scale * (u .⊙ u)
#   3. Recompute weighted mean:    xi .= d_new .* μ
#
# Total cost: O(n) per observation, no n×n matrix ever formed.

# --- MvNormalWeightedMeanPrecision{Diagonal} × LRD → in-place ADF step ---

BayesBase.default_prod_rule(
    ::Type{<:MvNormalWeightedMeanPrecision},
    ::Type{<:LRD},
) = PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::MvNormalWeightedMeanPrecision{T,V,M},
    right::LRD,
) where {T,V,M<:Diagonal}
    xi = weightedmean(left)
    d = invcov(left).diag       # mutable diagonal vector

    # Accumulate weighted mean contribution
    xi .+= right.xi

    # Sherman–Morrison: μ = (D + scale·uuᵀ)⁻¹ η  using OLD d
    #   v = D⁻¹u,  z = D⁻¹η,  s = 1/scale + uᵀv
    #   μ = z - v·(uᵀz / s)
    u = right.u
    scale = right.scale

    @inbounds begin
        s = inv(scale)
        utz = zero(T)
        for k in eachindex(d)
            vk = u[k] / d[k]
            zk = xi[k] / d[k]
            s += u[k] * vk
            utz += u[k] * zk
            # Stash z into xi temporarily (will overwrite below)
            xi[k] = zk
        end
        ratio = utz / s
        for k in eachindex(d)
            vk = u[k] / d[k]           # recompute v[k] (avoids O(n) temp alloc)
            μk = xi[k] - vk * ratio     # xi[k] currently holds z[k]
            d[k] += scale * u[k] * u[k] # update diagonal precision
            xi[k] = d[k] * μk           # recompute weighted mean
        end
    end

    return left
end

# --- Commutative: LRD × MvNormalWeightedMeanPrecision{Diagonal} ---

BayesBase.default_prod_rule(
    ::Type{<:LRD},
    ::Type{<:MvNormalWeightedMeanPrecision},
) = PreserveTypeProd(Distribution)

function BayesBase.prod(
    ::PreserveTypeProd{Distribution},
    left::LRD,
    right::MvNormalWeightedMeanPrecision{T,V,M},
) where {T,V,M<:Diagonal}
    return prod(PreserveTypeProd(Distribution), right, left)
end
