# ---------------------------------------------------------------------------
# Generic pipeline hooks for NoisyExpertsDiagonal
# ---------------------------------------------------------------------------

# Univariate
build_rxinfer_model(::Univariate, ::NoisyExpertsDiagonal, nf, no, p) =
    univariate_noisy_experts_diagonal(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Univariate, ::NoisyExpertsDiagonal, p, pred) =
    univariate_noisy_experts_diagonal_constraints(p, pred)

build_rxinfer_init(::Univariate, ::NoisyExpertsDiagonal, p) =
    univariate_noisy_experts_diagonal_init(p)
build_returnvars(::NoisyExpertsDiagonal) = (
    pred = KeepLast(),
    w = KeepEach(),
    τ = KeepEach(),
    β = KeepEach(),
    κ = KeepEach(),
    γ = KeepEach(),
)

training_posterior_keys(::NoisyExpertsDiagonal) = (:w, :τ, :β, :κ, :γ)

function model_results(::Any, ::NoisyExpertsDiagonal, posteriors, test_posteriors)
    return (
        w_posteriors = posteriors[:w],
        τ_posteriors = posteriors[:τ],
        β_posteriors = posteriors[:β],
        κ_posteriors = posteriors[:κ],
        γ_test = mean.(test_posteriors[:γ]),
    )
end

# ---------------------------------------------------------------------------
# Univariate prediction override
# ---------------------------------------------------------------------------

function predict_with_model(
    ::Univariate,
    ::NoisyExpertsDiagonal,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    model = univariate_noisy_experts_diagonal_prediction(
        n_forecasters = n_forecasters,
        n_obs = n_steps,
        priors = priors,
    )
    data = (features = features_test, predictions = predictions_test)

    constraints = univariate_noisy_experts_diagonal_constraints(priors, true)
    init = univariate_noisy_experts_diagonal_init(priors)

    result = infer(;
        model = model,
        data = data,
        constraints = constraints,
        initialization = init,
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    return (predictions = Dict(:y => result.posteriors[:y]), posteriors = result.posteriors)
end

# ---------------------------------------------------------------------------
# Multivariate pipeline hooks
# ---------------------------------------------------------------------------

function prepare_priors!(::Multivariate, ::NoisyExpertsDiagonal, priors, predictions)
    priors[:output_dim] = length(predictions[1, 1])
    return nothing
end

build_rxinfer_model(::Multivariate, ::NoisyExpertsDiagonal, nf, no, p) =
    multivariate_noisy_experts_diagonal(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Multivariate, ::NoisyExpertsDiagonal, p, pred) =
    multivariate_noisy_experts_diagonal_constraints(p, pred)

build_rxinfer_init(::Multivariate, ::NoisyExpertsDiagonal, p) =
    multivariate_noisy_experts_diagonal_init(p)

function predict_with_model(
    ::Multivariate,
    ::NoisyExpertsDiagonal,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    if !haskey(priors, :output_dim)
        priors[:output_dim] = length(predictions_test[1, 1])
    end

    model = multivariate_noisy_experts_diagonal_prediction(
        n_forecasters = n_forecasters,
        n_obs = n_steps,
        priors = priors,
    )
    data = (features = features_test, predictions = predictions_test)

    constraints = multivariate_noisy_experts_diagonal_constraints(priors, true)
    init = multivariate_noisy_experts_diagonal_init(priors)

    result = infer(;
        model = model,
        data = data,
        constraints = constraints,
        initialization = init,
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    return (predictions = Dict(:y => result.posteriors[:y]), posteriors = result.posteriors)
end
