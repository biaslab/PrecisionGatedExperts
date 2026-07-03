using RxInfer
using ClosedFormExpectations
using ClosedFormExpectations: LogGamma
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy

"""
    Exp

A node representing the exponential transformation of a real variable.
Exp(in) = out  =>  out = exp(in)

Interface order: [out, in]
  - :out  -> positive value, often a precision
  - :in   -> real-valued log-precision
"""
struct Exp end

export Exp

@node Exp Deterministic [out, in]

@rule Exp(:out, Marginalisation) (m_in::UnivariateGaussianDistributionsFamily,) = begin
    σ = max(std(m_in), sqrt(eps(Float64)))
    return LogNormal(mean(m_in), σ)
end

@rule Exp(:in, Marginalisation) (m_out::GammaDistributionsFamily,) = begin
    return LogGamma(scale(m_out), shape(m_out))
end

function _expnode_moment_matched_gamma(m::Real, v::Real)
    m = max(Float64(m), sqrt(eps(Float64)))
    v = max(Float64(v), eps(Float64))
    θ = v / m
    α = m / θ
    return Gamma(max(α, sqrt(eps(Float64))), max(θ, sqrt(eps(Float64))))
end

function _expnode_lognormal_moments(μ::Real, σ::Real)
    σ2 = min(max(Float64(σ)^2, eps(Float64)), 80.0)
    log_mean = clamp(Float64(μ) + 0.5 * σ2, -700.0, 700.0)
    log_second = clamp(2.0 * Float64(μ) + 2.0 * σ2, -700.0, 700.0)
    m = exp(log_mean)
    second = exp(log_second)
    v = max(second - m^2, eps(Float64))
    return m, v
end

function _expnode_precision_merge(m1::Real, v1::Real, m2::Real, v2::Real)
    τ1 = 1.0 / max(Float64(v1), eps(Float64))
    τ2 = 1.0 / max(Float64(v2), eps(Float64))
    τ = τ1 + τ2
    m = (τ1 * Float64(m1) + τ2 * Float64(m2)) / τ
    v = 1.0 / τ
    return m, v
end

function _expnode_digamma_approx(x::Real)
    y = max(Float64(x), sqrt(eps(Float64)))
    result = 0.0
    while y < 8.0
        result -= 1.0 / y
        y += 1.0
    end
    inv = 1.0 / y
    inv2 = inv^2
    return result + log(y) - 0.5 * inv -
           inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
end

function _expnode_trigamma_approx(x::Real)
    y = max(Float64(x), sqrt(eps(Float64)))
    result = 0.0
    while y < 8.0
        result += 1.0 / (y^2)
        y += 1.0
    end
    inv = 1.0 / y
    inv2 = inv^2
    return max(
        result + inv + 0.5 * inv2 + (1.0 / 6.0) * inv2 * inv - (1.0 / 30.0) * inv2^2 * inv +
        (1.0 / 42.0) * inv2^3 * inv,
        eps(Float64),
    )
end

function _expnode_loggamma_moments_from_gamma(gamma_message)
    α = max(shape(gamma_message), sqrt(eps(Float64)))
    θ = max(scale(gamma_message), sqrt(eps(Float64)))
    return log(θ) + _expnode_digamma_approx(α), _expnode_trigamma_approx(α)
end

# Marginal on :out, q(out) ∝ LogNormal(out) * Gamma(out), moment-matched to Gamma.
@marginalrule Exp(:out) (
    m_in::UnivariateGaussianDistributionsFamily,
    m_out::GammaDistributionsFamily,
) = begin
    m_lognormal, v_lognormal = _expnode_lognormal_moments(mean(m_in), std(m_in))
    m_gamma = mean(m_out)
    v_gamma = var(m_out)
    m, v = _expnode_precision_merge(m_lognormal, v_lognormal, m_gamma, v_gamma)
    return _expnode_moment_matched_gamma(m, v)
end

# Marginal on :in, q(in) ∝ LogGamma(in) * Normal(in), moment-matched to Normal.
@marginalrule Exp(:in) (
    m_out::GammaDistributionsFamily,
    m_in::UnivariateGaussianDistributionsFamily,
) = begin
    m_loggamma, v_loggamma = _expnode_loggamma_moments_from_gamma(m_out)
    m_normal = mean(m_in)
    v_normal = var(m_in)
    m, v = _expnode_precision_merge(m_loggamma, v_loggamma, m_normal, v_normal)
    return NormalMeanVariance(m, v)
end
