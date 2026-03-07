# ---------------------------------------------------------------------------
# Generic pipeline hooks for Dynamic
# ---------------------------------------------------------------------------

build_rxinfer_model(::Univariate, ::Dynamic, nf, no, p) =
    univariate_dynamic_ensemble(n_forecasters = nf, n_obs = no, priors = p)
build_rxinfer_model(::Multivariate, ::Dynamic, nf, no, p) =
    multivariate_dynamic_ensemble(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Univariate, ::Dynamic, p, pred) =
    univariate_dynamic_ensemble_constraints(p, pred)
build_rxinfer_constraints(::Multivariate, ::Dynamic, p, pred) =
    multivariate_dynamic_ensemble_constraints(p, pred)

build_rxinfer_init(::Univariate, ::Dynamic, p) = univariate_dynamic_ensemble_init(p)
build_rxinfer_init(::Multivariate, ::Dynamic, p) = multivariate_dynamic_ensemble_init(p)

training_posterior_keys(::Dynamic) = (:w, :τ, :β, :γ)

function model_results(::Any, ::Dynamic, posteriors, test_posteriors)
    return (
        w_posteriors = posteriors[:w],
        τ_posteriors = posteriors[:τ],
        β_posteriors = posteriors[:β],
        γ_test = mean.(test_posteriors[:γ]),
    )
end
