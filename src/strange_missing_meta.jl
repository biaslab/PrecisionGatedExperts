# ---------------------------------------------------------------------------
# Free energy scoring for Uninformative node with observed (PointMass) marginal.
# E_q[-log Uninformative(x)] = 0 since log(const) = 0.
# ---------------------------------------------------------------------------

@average_energy Uninformative (q_out::PointMass,) = begin
    return 0.0
end

struct StrangeMissingMeta end

# ---------------------------------------------------------------------------
# Univariate StrangeMissingMeta rules for NormalMeanPrecision
# ---------------------------------------------------------------------------

# BP message from Normal to Normal mean is uninformative when message is uninformative
@rule NormalMeanPrecision(:μ, Marginalisation) (
    m_out::Uninformative,
    q_τ::Any,
    meta::StrangeMissingMeta,
) = begin
    return Uninformative()
end

@rule NormalMeanPrecision(:μ, Marginalisation) (
    m_out::NormalWeightedMeanPrecision,
    q_τ::Gamma,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule NormalMeanPrecision(:μ, Marginalisation) (m_out = m_out, q_τ = q_τ)
end

# VMP rule
@marginalrule NormalMeanPrecision(:out_μ) (
    m_out::Uninformative,
    m_μ::UnivariateNormalDistributionsFamily,
    q_τ::Any,
    meta::StrangeMissingMeta,
) = begin
    return missing
    # TODO: the rule below "is correct" however it gives the divergence in gamma because of the resulting message for \tau
    # If I put missing here then it will go a missing to tau and it will converge, because \tau will not depend on the joint
    # xi_μ, W_μ = weightedmean_precision(m_μ)
    # W_bar = mean(q_τ)
    # W  = [W_bar -W_bar; -W_bar W_μ+W_bar]
    # xi = [zero(xi_μ); xi_μ]
    # return MvNormalWeightedMeanPrecision(xi, W)
end

@rule NormalMeanPrecision(:τ, Marginalisation) (
    q_out_μ::MvNormalWeightedMeanPrecision,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule NormalMeanPrecision(:τ, Marginalisation) (q_out_μ = q_out_μ,)
end

@rule NormalMeanPrecision(:out, Marginalisation) (
    m_μ::NormalMeanPrecision,
    q_τ::Gamma,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule NormalMeanPrecision(:out, Marginalisation) (m_μ = m_μ, q_τ = q_τ)
end

# ---------------------------------------------------------------------------
# Multivariate StrangeMissingMeta rules for MvNormalMeanScalePrecision
# ---------------------------------------------------------------------------

# 1. :μ when y message is Uninformative → propagate Uninformative upward
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
    m_out::Uninformative,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return Uninformative()
end

# 2. :μ when y message is a proper MvNormal → delegate to standard rule
@rule MvNormalMeanScalePrecision(:μ, Marginalisation) (
    m_out::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule MvNormalMeanScalePrecision(:μ, Marginalisation) (m_out = m_out, q_γ = q_γ)
end

# 3. :out_μ joint marginal when y is Uninformative → return missing
#    This prevents divergence in γ (same trick as univariate)
@marginalrule MvNormalMeanScalePrecision(:out_μ) (
    m_out::Uninformative,
    m_μ::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return missing
end

# 4. :γ (precision) from structured joint marginal → delegate
@rule MvNormalMeanScalePrecision(:γ, Marginalisation) (
    q_out_μ::MultivariateNormalDistributionsFamily,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule MvNormalMeanScalePrecision(:γ, Marginalisation) (q_out_μ = q_out_μ,)
end

# 5. :out (observation) message → delegate to standard rule
@rule MvNormalMeanScalePrecision(:out, Marginalisation) (
    m_μ::MultivariateNormalDistributionsFamily,
    q_γ::Any,
    meta::StrangeMissingMeta,
) = begin
    return @call_rule MvNormalMeanScalePrecision(:out, Marginalisation) (m_μ = m_μ, q_γ = q_γ)
end
