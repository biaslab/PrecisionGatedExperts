"""
Dynamic Probabilistic Ensemble Forecasting with RxInfer

This script demonstrates how to create a dynamic probabilistic ensemble model using RxInfer
where the weight of each forecaster depends on the input features. This allows the ensemble
to automatically trust different forecasters in different regions of input space.

## Mathematical Model

We make precision depend on input features through a linear-exponential link:

    z_i(x) = w_i^T x           (linear projection)
    γ_i(x) = exp(z_i(x))       (ensures positivity)

The full generative model:

    w_i ~ N(0, Σ_w)                           for i = 1, ..., n
    z_{i,j} = w_i^T x_j                       (deterministic)
    γ_{i,j} = exp(z_{i,j})
    y_j ~ N(f_i(x_j), γ_{i,j}^{-1})           for all i, j

## Why This Works

- If w_i^T x is large positive → γ_i(x) is large → forecaster i gets high weight
- If w_i^T x is large negative → γ_i(x) ≈ 0 → forecaster i is effectively ignored
- The gating is learned from data: forecasters that perform well in certain regions
  will have w_i that activates them there

## Comparison with Static Ensemble

| Aspect              | Static Ensemble      | Dynamic Ensemble                    |
|---------------------|----------------------|-------------------------------------|
| Precision           | γ_i (global)         | γ_i(x) = exp(w_i^T x)               |
| Weights             | Constant everywhere  | Input-dependent                     |
| Parameters          | n precisions         | n × d weight coefficients           |
| Non-conjugacy       | None                 | Log link requires projection        |
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Random
using Statistics
using LinearAlgebra
using Plots

# Include utilities from the main package
using ProbabilisticEnsembling

# =============================================================================
# Dynamic Ensemble Model Definition
# =============================================================================

@model function dynamic_ensemble_model(n_forecasters, n_obs, features, predictions, y, w_priors)
    # features: Vector of vectors, each of length n_features
    # predictions: [n_forecasters × n_obs] - forecaster outputs
    # w_priors: vector of MvNormal priors for gating weights

    local w, z, γ, τ

    # Gating weights for each forecaster
    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
    end

    # For each observation
    for j in 1:n_obs
        # For each forecaster
        for i in 1:n_forecasters
            # τ is precision for softdot (z = w^T x with noise)
            τ[i, j] ~ GammaShapeScale(1.0, 1.0)
            z[i, j] ~ softdot(features[j], w[i], τ[i, j])

            # γ = exp(z) via Log node: z = Log(γ)
            # Use GammaShapeScale for CFE compatibility
            γ[i, j] ~ GammaShapeScale(1.0, 1.0)
            z[i, j] ~ Log(γ[i, j])

            # Likelihood: observation given forecaster i's prediction
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function dynamic_ensemble_constraints()
    # Mean-field factorization: all variables factorize
    q(w, z, γ, τ) = q(w)q(z)q(γ)q(τ)

    # Projection constraints with ClosedFormStrategy for faster inference
    # Use Gamma (GammaShapeScale) instead of GammaShapeRate for projection - it has manifold support
    q(z) :: ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ) :: ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

@initialization function dynamic_ensemble_init()
    q(w) = MvNormalMeanPrecision(zeros(2), diagm(ones(2)))
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
end

# =============================================================================
# Static Ensemble Model (for comparison)
# =============================================================================

@model function static_ensemble_model(n_forecasters, predictions, y, priors)
    local γ

    for i in 1:n_forecasters
        γ[i] ~ priors[i]
    end

    for i in 1:n_forecasters
        for j in eachindex(y)
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i])
        end
    end
end

# =============================================================================
# Feature Design Functions
# =============================================================================

"""
    create_features(x, feature_type::Symbol)

Create feature vectors from raw input x.

Available feature types:
- :raw       - Just x (linear dependence)
- :bias      - [1, x] (baseline + linear)
- :quadratic - [1, x, x²] (quadratic dependence)
- :trig      - [1, sin(x), cos(x)] (periodic features)
"""
function create_features(x, feature_type::Symbol)
    n = length(x)

    if feature_type == :raw
        return [Float64[xi] for xi in x]
    elseif feature_type == :bias
        return [Float64[1.0, xi] for xi in x]
    elseif feature_type == :quadratic
        return [Float64[1.0, xi, xi^2] for xi in x]
    elseif feature_type == :trig
        return [Float64[1.0, sin(xi), cos(xi)] for xi in x]
    else
        error("Unknown feature type: $feature_type")
    end
end

# =============================================================================
# Demo Function
# =============================================================================

"""
    demo_dynamic_ensemble()

Demonstrate the dynamic ensemble inference with synthetic data.
"""
function demo_dynamic_ensemble()
    Random.seed!(42)

    @info "Dynamic Probabilistic Ensemble Forecasting Demo"
    @info "================================================"

    # =========================================================================
    # Step 1: Generate synthetic data with region-dependent truth
    # =========================================================================
    @info "Step 1: Generating Synthetic Data with Region-Dependent Structure"

    n_train = 30
    noise_level = 0.15

    x_train = collect(range(0, 4π, length=n_train))

    # True function: sin dominates in first half, cos in second half
    function true_function(x)
        transition_point = 2π
        transition_width = π
        weight_cos = 0.5 * (1 + tanh((x - transition_point) / transition_width))
        weight_sin = 1 - weight_cos
        return weight_sin * sin(x) + weight_cos * cos(x)
    end

    y_train = true_function.(x_train) .+ noise_level .* randn(n_train)

    # Define forecasters
    forecasters = [
        ("sin(x)", x -> sin(x)),
        ("cos(x)", x -> cos(x)),
        ("sin+cos", x -> 0.5 * (sin(x) + cos(x))),
    ]

    n_forecasters = length(forecasters)

    # Generate predictions
    predictions_train = zeros(n_forecasters, n_train)
    for (i, (_, f)) in enumerate(forecasters)
        predictions_train[i, :] = f.(x_train)
    end

    @info "Data Summary" n_train n_forecasters true_function="transition from sin to cos" noise_level

    # Create feature vectors for gating
    feature_type = :bias  # [1, x] - allows baseline + linear gating
    features_train = create_features(x_train, feature_type)
    n_features = length(features_train[1])

    @info "Feature Design" feature_type n_features

    # =========================================================================
    # Step 2: Train Dynamic Ensemble using RxInfer
    # =========================================================================
    @info "Step 2: Training Dynamic Ensemble using RxInfer"

    # Initial priors on gating weights
    w_priors_init = [MvNormalMeanPrecision(zeros(n_features), 0.1 * diagm(ones(n_features))) for _ in 1:n_forecasters]

    @info "Running variational inference with projection..."

    dynamic_result = infer(
        model = dynamic_ensemble_model(
            n_forecasters = n_forecasters,
            n_obs = n_train,
            w_priors = w_priors_init
        ),
        data = (y = y_train, features = features_train, predictions = predictions_train),
        constraints = dynamic_ensemble_constraints(),
        initialization = dynamic_ensemble_init(),
        iterations = 30,
        free_energy = true
    )

    # Extract learned weight posteriors
    w_posteriors = dynamic_result.posteriors[:w][end]

    @info "Learned Gating Weight Posteriors"
    for (i, (name, _)) in enumerate(forecasters)
        w_mean = mean(w_posteriors[i])
        @info "  Forecaster $i" name w_bias=round(w_mean[1], digits=4) w_slope=round(w_mean[2], digits=4)
    end

    return (
        w_posteriors = w_posteriors,
        forecasters = forecasters
    )
end

function main()
    demo_dynamic_ensemble()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
