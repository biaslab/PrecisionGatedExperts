# ---------------------------------------------------------------------------
# Generic pipeline hooks for DynamicExp
# ---------------------------------------------------------------------------

build_rxinfer_model(::Univariate, ::DynamicExp, nf, no, p) =
    univariate_dynamic_exp_ensemble(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Univariate, ::DynamicExp, p, pred) =
    univariate_dynamic_exp_ensemble_constraints(p, pred)

build_rxinfer_init(::Univariate, ::DynamicExp, p) = univariate_dynamic_exp_ensemble_init(p)

training_posterior_keys(::DynamicExp) = (:w, :τ, :γ)

function model_results(::Any, ::DynamicExp, posteriors, test_posteriors)
    return (
        w_posteriors = posteriors[:w],
        τ_posteriors = posteriors[:τ],
        γ_test = mean.(test_posteriors[:γ]),
    )
end
