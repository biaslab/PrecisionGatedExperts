using RxInfer
using ClosedFormExpectations
using ClosedFormExpectations: LogGamma
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy

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
# q(γ) ∝ m_in(γ) × m_out(log(γ))  =>  log q(γ) = log m_in(γ) + log m_out(log(γ))
# We want to project this onto a Gamma distribution for `in` (γ).
@marginalrule Log(:in) (m_out::UnivariateGaussianDistributionsFamily, m_in::GammaDistributionsFamily) = begin
    σ = max(std(m_out), sqrt(eps(Float64)))
    log_normal = LogNormal(mean(m_out), σ)
    prj = ProjectedTo(Gamma; parameters = ProjectionParameters(
        strategy = ClosedFormStrategy(),
        niterations = 50,
        tolerance = 1e-6,
        stepsize = ExponentialFamilyProjection.Manopt.ArmijoLinesearch(
            initial_stepsize = 1e-2,
            stop_increasing_at_step = 0
        ),
        direction = ExponentialFamilyProjection.BoundedNormUpdateRule(0.5)
    ))

    supplementary = convert(Gamma, m_in)
    α = max(shape(supplementary), sqrt(eps(Float64)))
    θ = max(scale(supplementary), sqrt(eps(Float64)))
    # With supplementary projection, natural parameters are shifted by subtraction.
    # Choosing θ_init < θ keeps the effective η₂ strictly negative.
    initial = Gamma(α, max(0.1 * θ, sqrt(eps(Float64))))

    # Project q(γ) ∝ LogNormal(γ) * m_in(γ)
    return project_to(prj, log_normal, supplementary; initialpoint = initial)
end
