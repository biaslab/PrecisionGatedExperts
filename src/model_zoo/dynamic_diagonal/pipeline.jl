# ---------------------------------------------------------------------------
# Generic pipeline hooks for DynamicDiagonal
# ---------------------------------------------------------------------------

build_rxinfer_model(::Univariate, ::DynamicDiagonal, nf, no, p) =
    univariate_dynamic_diagonal_ensemble(n_forecasters = nf, n_obs = no, priors = p)
build_rxinfer_model(::Multivariate, ::DynamicDiagonal, nf, no, p) =
    multivariate_dynamic_diagonal_ensemble(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Univariate, ::DynamicDiagonal, p, pred) =
    univariate_dynamic_diagonal_ensemble_constraints(p, pred)
build_rxinfer_constraints(::Multivariate, ::DynamicDiagonal, p, pred) =
    multivariate_dynamic_diagonal_ensemble_constraints(p, pred)

build_rxinfer_init(::Univariate, ::DynamicDiagonal, p) =
    univariate_dynamic_diagonal_ensemble_init(p)
build_rxinfer_init(::Multivariate, ::DynamicDiagonal, p) =
    multivariate_dynamic_diagonal_ensemble_init(p)

training_posterior_keys(::DynamicDiagonal) = (:w, :τ, :β, :γ)

function model_results(::Any, ::DynamicDiagonal, posteriors, test_posteriors)
    return (
        w_posteriors = posteriors[:w],
        τ_posteriors = posteriors[:τ],
        β_posteriors = posteriors[:β],
        γ_test = mean.(test_posteriors[:γ]),
    )
end
