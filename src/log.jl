using RxInfer
using ClosedFormExpectations: LogGamma

"""
    Log

A node representing the logarithm transformation of a positive variable.
Log(in) = out  =>  in = exp(out)
"""
struct Log end

export Log

@node Log Deterministic [out, in]

# Define Custom Rules for the Log node to return distributions compatible with CFE
# We assume Log(in) = out  =>  in = exp(out)
# Message to `out` (x) given `in` (lambda, Gamma) is LogGamma.
# Message to `in` (lambda) given `out` (x, Normal) is LogNormal.
@rule Log(:out, Marginalisation) (m_in::GammaDistributionsFamily,) = begin
    # m_in is Gamma(α, θ) on λ.
    # Target on x: LogGamma(α, θ).
    # LogGamma(α, β) where α is scale, β is shape
    return LogGamma(scale(m_in), shape(m_in))
end

@rule Log(:in, Marginalisation) (m_out::UnivariateGaussianDistributionsFamily,) = begin
    # m_out is Normal(μ, σ) on x.
    # Target on λ: LogNormal(μ, σ).
    return LogNormal(mean(m_out), std(m_out))
end

# Marginal rule for free energy computation
# This is called when computing the marginal of `in` (γ) given messages from both sides
@marginalrule Log(:in) (m_out::UnivariateGaussianDistributionsFamily, m_in::GammaDistributionsFamily) = begin
    # Combine the incoming message from γ prior (Gamma) with the backward message from z (via Log)
    # The marginal is the product of the prior and likelihood
    # For now, return the prior as the marginal (will be updated by projection)
    return m_in
end
