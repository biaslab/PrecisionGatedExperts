# ---------------------------------------------------------------------------
# Generic pipeline hooks for DynamicReLU
# ---------------------------------------------------------------------------

build_rxinfer_model(::Univariate, ::DynamicReLU, nf, no, p) =
    univariate_dynamic_relu_ensemble(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Univariate, ::DynamicReLU, p, pred) =
    univariate_dynamic_relu_ensemble_constraints(p, pred)

build_rxinfer_init(::Univariate, ::DynamicReLU, p) =
    univariate_dynamic_relu_ensemble_init(p)

training_posterior_keys(::DynamicReLU) = (:w, :τ, :γ)

function model_results(::Any, ::DynamicReLU, posteriors, test_posteriors)
    return (
        w_posteriors = posteriors[:w],
        τ_posteriors = posteriors[:τ],
        γ_test = mean.(test_posteriors[:γ]),
    )
end
