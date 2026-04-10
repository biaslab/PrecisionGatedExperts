using RxInfer
using ClosedFormExpectations
using ClosedFormExpectations: ReLUForwardMessage, ReLUBackwardMessage
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy

"""
    ReLU

A node representing the ReLU transformation of a variable.
ReLU(in) = out  =>  out = max(0, in)

Interface order: [out, in]
  - :out  →  γ  (the non-negative output, e.g. precision weight)
  - :in   →  z  (the pre-activation, any real value)

Uses exact message types from ClosedFormExpectations.jl v0.4:
  - ReLUForwardMessage:  exact forward message m_{f→y}(y)
  - ReLUBackwardMessage: exact backward message m_{f→x}(x)

Both q(z) and q(γ) should be projected onto Gamma (positive support),
since:
  - γ = max(0, z) is non-negative by construction
  - z's useful domain is (0, ∞); projecting z onto Normal would assign
    mass to z < 0 where Gamma's logpdf is -Inf
"""
struct ReLU end

export ReLU

@node ReLU Deterministic [out, in]

# ------------------------------------------------------------------
# Forward message rules: messages toward :out (γ)
# ------------------------------------------------------------------

# Given Normal belief on z → exact ReLU forward message to γ
@rule ReLU(:out, Marginalisation) (m_in::UnivariateGaussianDistributionsFamily,) = begin
    return ReLUForwardMessage(mean(m_in), var(m_in))
end

# Given Gamma belief on z → ReLU is identity on (0, ∞), forward message is just Gamma
@rule ReLU(:out, Marginalisation) (m_in::GammaDistributionsFamily,) = begin
    return convert(Gamma, m_in)
end

# ------------------------------------------------------------------
# Backward message rules: messages toward :in (z)
# ------------------------------------------------------------------

# Given Gamma belief on γ → ReLU is identity on (0, ∞), backward message is just Gamma
@rule ReLU(:in, Marginalisation) (m_out::GammaDistributionsFamily,) = begin
    return convert(Gamma, m_out)
end

# Given Normal belief on γ → exact ReLU backward message to z
@rule ReLU(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,) = begin
    return ReLUBackwardMessage(mean(m_out), var(m_out))
end

# ------------------------------------------------------------------
# Marginal rules (needed for Bethe Free Energy computation)
# ------------------------------------------------------------------
# These marginal rules are used by the BFE score computation, NOT by
# the VMP message passing (which uses the message rules + ProjectedTo
# constraints at the variable level).
#
# Strategy: use moment-matched Gamma approximation instead of full
# projection to avoid numerical issues (PosDefException/DomainError)
# when projecting Normal with extreme parameters onto Gamma manifold.

# Helper: moment-match a Gamma from mean and variance (both must be positive)
function _moment_matched_gamma(m::Real, v::Real)
    m = max(m, sqrt(eps(Float64)))
    v = max(v, eps(Float64))
    θ = v / m
    α = m / θ
    return Gamma(max(α, sqrt(eps(Float64))), max(θ, sqrt(eps(Float64))))
end

# Marginal on :in (z) given messages from both sides.
# m_out: Gamma (from downstream on γ), m_in: Normal (from softdot on z)
# Approximate q(z) ∝ Gamma(z) × Normal(z) by moment-matched Gamma.
@marginalrule ReLU(:in) (
    m_out::GammaDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
) = begin
    # Combine moments from both messages, weighting toward the Gamma support
    m_γ = mean(m_out)
    v_γ = var(m_out)
    m_n = max(mean(m_in), sqrt(eps(Float64)))  # clamp to positive
    v_n = var(m_in)
    # Precision-weighted combination (both treated as Gaussian-like)
    τ_γ = 1.0 / max(v_γ, eps(Float64))
    τ_n = 1.0 / max(v_n, eps(Float64))
    τ_total = τ_γ + τ_n
    m_combined = (τ_γ * m_γ + τ_n * m_n) / τ_total
    v_combined = 1.0 / τ_total
    return _moment_matched_gamma(m_combined, v_combined)
end

# Marginal on :in when both messages are Gamma (e.g. after first iteration)
@marginalrule ReLU(:in) (
    m_out::GammaDistributionsFamily,
    m_in::GammaDistributionsFamily,
) = begin
    return prod(ClosedProd(), convert(Gamma, m_out), convert(Gamma, m_in))
end

# Marginal on :out (γ) given messages from both sides.
# m_in: Normal (from softdot on z), m_out: Gamma (from NormalMeanPrecision on γ)
# Approximate q(γ) ∝ ReLUForward(γ) × Gamma(γ) by moment-matched Gamma.
@marginalrule ReLU(:out) (
    m_in::UnivariateGaussianDistributionsFamily,
    m_out::GammaDistributionsFamily,
) = begin
    # Forward ReLU message is like a truncated Normal on (0, ∞)
    # For positive mean, approximate moments: mean ≈ max(μ, 0), var ≈ σ²
    m_fwd = max(mean(m_in), sqrt(eps(Float64)))
    v_fwd = var(m_in)
    m_γ = mean(m_out)
    v_γ = var(m_out)
    τ_fwd = 1.0 / max(v_fwd, eps(Float64))
    τ_γ = 1.0 / max(v_γ, eps(Float64))
    τ_total = τ_fwd + τ_γ
    m_combined = (τ_fwd * m_fwd + τ_γ * m_γ) / τ_total
    v_combined = 1.0 / τ_total
    return _moment_matched_gamma(m_combined, v_combined)
end

# Marginal on :out when both messages are Gamma
@marginalrule ReLU(:out) (
    m_in::GammaDistributionsFamily,
    m_out::GammaDistributionsFamily,
) = begin
    return prod(ClosedProd(), convert(Gamma, m_in), convert(Gamma, m_out))
end
