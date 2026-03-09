# ---------------------------------------------------------------------------
# Generic pipeline hooks for Static
# ---------------------------------------------------------------------------

build_rxinfer_model(::Univariate, ::Static, nf, _, p) =
    univariate_ensemble_precision_model(n_forecasters = nf, priors = p)
build_rxinfer_model(::Multivariate, ::Static, nf, _, p) =
    multivariate_ensemble_precision_model(n_forecasters = nf, priors = p)

build_training_data(::Any, ::Static, y, _, predictions) = (y = y, X = predictions)

training_posterior_keys(::Static) = (:γ,)
prediction_prior_keys(::Static) = (:γ,)

function model_results(::Any, ::Static, posteriors, _test_posteriors)
    γ_posteriors = posteriors[:γ]
    γ_means = map(mean, γ_posteriors)
    weights = γ_means ./ sum(γ_means)
    return (γ_posteriors = γ_posteriors, weights = weights)
end
