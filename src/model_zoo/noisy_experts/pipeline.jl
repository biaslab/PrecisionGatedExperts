# ---------------------------------------------------------------------------
# Generic pipeline hooks for NoisyExperts
# ---------------------------------------------------------------------------

# Training model (y is data)
build_rxinfer_model(::Univariate, ::NoisyExperts, nf, no, p) =
    univariate_noisy_experts(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Univariate, ::NoisyExperts, p, pred) =
    univariate_noisy_experts_constraints(p, pred)

build_rxinfer_init(::Univariate, ::NoisyExperts, p) = univariate_noisy_experts_init(p)

training_posterior_keys(::NoisyExperts) = (:w, :τ, :β, :κ, :γ)

function model_results(::Any, ::NoisyExperts, posteriors, test_posteriors)
    return (
        w_posteriors = posteriors[:w],
        τ_posteriors = posteriors[:τ],
        β_posteriors = posteriors[:β],
        κ_posteriors = posteriors[:κ],
        γ_test = mean.(test_posteriors[:γ]),
    )
end

# ---------------------------------------------------------------------------
# Prediction override: use the prediction model where y is a latent variable
# (not data), so the custom BP @marginalrule fires via Uninformative().
# ---------------------------------------------------------------------------

function predict_with_model(
    ::Univariate,
    ::NoisyExperts,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,       # ignored — y is latent in prediction model
    predictions_test,
    features_test,
    prediction_iterations,
)
    # Prediction model: y is NOT in the signature → true latent with Uninformative()
    model = univariate_noisy_experts_prediction(
        n_forecasters = n_forecasters,
        n_obs = n_steps,
        priors = priors,
    )
    # Data without y — only features and expert predictions
    data = (features = features_test, predictions = predictions_test)

    constraints = univariate_noisy_experts_constraints(priors, true)
    init = univariate_noisy_experts_init(priors)

    result = infer(;
        model = model,
        data = data,
        constraints = constraints,
        initialization = init,
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    # The generic pipeline expects result.predictions[:y].
    # In the prediction model, y is a posterior (latent), not a prediction (data).
    # Wrap the result so the pipeline can find y in .predictions[:y].
    return (
        predictions = Dict(:y => result.posteriors[:y]),
        posteriors = result.posteriors,
    )
end

# ---------------------------------------------------------------------------
# Multivariate pipeline hooks
# ---------------------------------------------------------------------------

function prepare_priors!(::Multivariate, ::NoisyExperts, priors, predictions)
    priors[:output_dim] = length(predictions[1, 1])
    return nothing
end

build_rxinfer_model(::Multivariate, ::NoisyExperts, nf, no, p) =
    multivariate_noisy_experts(n_forecasters = nf, n_obs = no, priors = p)

build_rxinfer_constraints(::Multivariate, ::NoisyExperts, p, pred) =
    multivariate_noisy_experts_constraints(p, pred)

build_rxinfer_init(::Multivariate, ::NoisyExperts, p) =
    multivariate_noisy_experts_init(p)

function predict_with_model(
    ::Multivariate,
    ::NoisyExperts,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    # Ensure output_dim is available for initialization
    if !haskey(priors, :output_dim)
        priors[:output_dim] = length(predictions_test[1, 1])
    end

    model = multivariate_noisy_experts_prediction(
        n_forecasters = n_forecasters,
        n_obs = n_steps,
        priors = priors,
    )
    data = (features = features_test, predictions = predictions_test)

    constraints = multivariate_noisy_experts_constraints(priors, true)
    init = multivariate_noisy_experts_init(priors)

    result = infer(;
        model = model,
        data = data,
        constraints = constraints,
        initialization = init,
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    return (
        predictions = Dict(:y => result.posteriors[:y]),
        posteriors = result.posteriors,
    )
end
