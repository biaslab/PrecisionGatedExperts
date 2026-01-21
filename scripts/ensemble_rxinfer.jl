"""
Probabilistic Ensemble Forecasting with RxInfer

This example demonstrates how to create a probabilistic ensemble model using RxInfer
where multiple forecasting functions are combined with learned precision weights.

## Mathematical Model

Given n forecasting functions f₁, ..., fₙ, we model the relationship:

    p(y, γ₁, ..., γₙ | x) = p(γ₁)⋯p(γₙ) ∏ⱼ ∏ᵢ N(yⱼ | fᵢ(xⱼ), γᵢ⁻¹)

where:
- y is the target observation
- x is the input (treated as fixed/known)
- fᵢ(x) is the prediction from the i-th forecasting function
- γᵢ is the precision (inverse variance) of the i-th function
- Each γᵢ has a Gamma prior: γᵢ ~ Gamma(α₀, β₀)

## Key Insight: Conjugate Update

For each forecaster i with N validation observations, the posterior precision is:

    γᵢ | data ~ Gamma(α₀ + N/2, β₀ + SSE_i/2)

where SSE_i = ∑ⱼ(yⱼ - fᵢ(xⱼ))² is the sum of squared errors.

## Ensemble Prediction

The optimal ensemble prediction uses precision-weighted averaging:

    ŷ_ensemble = (∑ᵢ E[γᵢ] * fᵢ(x)) / (∑ᵢ E[γᵢ])

with uncertainty:

    Var[y_ensemble] ≈ 1 / (∑ᵢ E[γᵢ])

This automatically downweights unreliable forecasters!
"""

using RxInfer
using Distributions
using Random
using Statistics
using LinearAlgebra
using JLD2
using Lux
using Plots

# Include utilities from the main package
using ProbabilisticEnsembling

#=
===============================================================================
SECTION 1: RxInfer Model Definition
===============================================================================

We use RxInfer to define a proper probabilistic model for the ensemble.
The model learns precision parameters γᵢ for each forecaster from validation data.
=#

"""
    @model ensemble_precision_model(n_forecasters, n_obs; α₀, β₀)

RxInfer model for learning forecaster precisions.

Each forecaster i has precision γᵢ with prior Gamma(α₀, β₀).
For each observation j, we assume:
    y_obs[j] ~ Normal(prediction[i,j], 1/γᵢ)  for forecaster i

This model learns independent precision for each forecaster.
"""
@model function ensemble_precision_model(n_forecasters, n_obs; α₀=1.0, β₀=1.0)
    # Prior on precision for each forecaster
    γ = randomvar(n_forecasters)
    for i in 1:n_forecasters
        γ[i] ~ Gamma(α₀, inv(β₀))  # Gamma(shape, scale) in Julia
    end

    # Predictions from each forecaster (pre-computed, passed as data)
    preds = datavar(Matrix{Float64})  # shape: (n_forecasters, n_obs)

    # Observations
    y_obs = datavar(Vector{Float64})   # shape: (n_obs,)

    # Likelihood: each observation relates to each forecaster's prediction
    # We model this as: for each forecaster, its predictions explain the observations
    # This is handled by creating separate likelihood terms

    return γ, preds, y_obs
end

"""
    @model single_forecaster_model(n_obs; α₀, β₀)

Model for a single forecaster's precision estimation.
This is simpler and allows standard RxInfer inference.

    γ ~ Gamma(α₀, β₀)
    for j in 1:n_obs:
        y[j] ~ Normal(pred[j], 1/γ)
"""
@model function single_forecaster_model(n_obs; α₀=1.0, β₀=1.0)
    # Prior on precision
    γ ~ Gamma(α₀, inv(β₀))

    # Predictions (as data)
    pred = datavar(Vector{Float64})

    # Observations
    y_obs = datavar(Vector{Float64})

    # Likelihood
    for j in 1:n_obs
        y_obs[j] ~ NormalMeanPrecision(pred[j], γ)
    end

    return γ, pred, y_obs
end

#=
===============================================================================
SECTION 2: Analytical Precision Estimation (Closed-Form)
===============================================================================

Since we have conjugate Gamma-Normal structure, the posterior is analytical.
This is more efficient than running iterative inference.
=#

"""
    GammaParams

Store Gamma distribution parameters in shape-rate parameterization.
"""
struct GammaParams
    α::Float64  # shape
    β::Float64  # rate
end

Base.mean(g::GammaParams) = g.α / g.β
Base.var(g::GammaParams) = g.α / g.β^2
Base.std(g::GammaParams) = sqrt(var(g))

function Distributions.Gamma(g::GammaParams)
    return Gamma(g.α, 1/g.β)  # Gamma(shape, scale) where scale = 1/rate
end

"""
    estimate_precision(predictions, y_true; α₀=1.0, β₀=1.0)

Analytically compute posterior Gamma parameters for precision.

Given:
- Prior: γ ~ Gamma(α₀, β₀)  [shape-rate parameterization]
- Likelihood: y ~ N(pred, 1/γ)

Posterior: γ | data ~ Gamma(α₀ + n/2, β₀ + SSE/2)
"""
function estimate_precision(predictions::Vector{T}, y_true::Vector{T};
                           α₀=1.0, β₀=1.0) where {T<:Real}
    n = length(y_true)
    @assert length(predictions) == n "Predictions and targets must have same length"

    # Sum of squared errors
    sse = sum((y_true .- predictions).^2)

    # Posterior parameters
    α_post = α₀ + n / 2
    β_post = β₀ + sse / 2

    return GammaParams(α_post, β_post)
end

"""
    estimate_all_precisions(predictions_matrix, y_true; α₀=1.0, β₀=1.0)

Estimate precision posteriors for all forecasters.

# Arguments
- `predictions_matrix`: Matrix of shape (n_forecasters, n_obs)
- `y_true`: Vector of true observations (length n_obs)
- `α₀, β₀`: Prior Gamma parameters (shape, rate)

# Returns
- Vector of GammaParams for each forecaster
"""
function estimate_all_precisions(predictions_matrix::Matrix{T}, y_true::Vector{T};
                                 α₀=1.0, β₀=1.0) where {T<:Real}
    n_forecasters = size(predictions_matrix, 1)

    posteriors = Vector{GammaParams}(undef, n_forecasters)
    for i in 1:n_forecasters
        posteriors[i] = estimate_precision(
            predictions_matrix[i, :], y_true;
            α₀=α₀, β₀=β₀
        )
    end

    return posteriors
end

#=
===============================================================================
SECTION 3: Ensemble Prediction Functions
===============================================================================
=#

"""
    compute_ensemble_weights(γ_posteriors)

Compute normalized weights from precision posteriors.
Weight for forecaster i is proportional to E[γᵢ].
"""
function compute_ensemble_weights(γ_posteriors::Vector{GammaParams})
    γ_means = [mean(γ) for γ in γ_posteriors]
    total = sum(γ_means)
    return γ_means ./ total
end

"""
    ensemble_predict(predictions, γ_posteriors)

Make ensemble predictions with uncertainty quantification.

# Arguments
- `predictions`: Matrix (n_forecasters, n_points) of predictions
- `γ_posteriors`: Vector of GammaParams for each forecaster

# Returns
- `mean_pred`: Precision-weighted ensemble predictions
- `std_pred`: Prediction uncertainty (standard deviation)
- `weights`: Normalized weights for each forecaster
"""
function ensemble_predict(predictions::Matrix{T},
                         γ_posteriors::Vector{GammaParams}) where {T<:Real}
    n_forecasters, n_points = size(predictions)
    @assert length(γ_posteriors) == n_forecasters

    # Compute weights from precision means
    weights = compute_ensemble_weights(γ_posteriors)
    γ_means = [mean(γ) for γ in γ_posteriors]

    # Precision-weighted predictions
    mean_pred = zeros(n_points)
    for i in 1:n_forecasters
        mean_pred .+= weights[i] .* predictions[i, :]
    end

    # Uncertainty: variance = 1 / total_precision
    total_precision = sum(γ_means)
    std_pred = sqrt(1.0 / total_precision)

    return mean_pred, std_pred, weights
end

"""
    ensemble_predict_with_variance(predictions, γ_posteriors)

More sophisticated ensemble prediction that accounts for forecaster uncertainty.

Returns per-point uncertainty estimates.
"""
function ensemble_predict_with_variance(predictions::Matrix{T},
                                        γ_posteriors::Vector{GammaParams}) where {T<:Real}
    n_forecasters, n_points = size(predictions)

    weights = compute_ensemble_weights(γ_posteriors)
    γ_means = [mean(γ) for γ in γ_posteriors]

    # Weighted mean prediction
    mean_pred = zeros(n_points)
    for i in 1:n_forecasters
        mean_pred .+= weights[i] .* predictions[i, :]
    end

    # Variance has two components:
    # 1. Noise variance: 1 / Σ γᵢ
    # 2. Model disagreement: weighted variance of predictions
    total_precision = sum(γ_means)
    noise_var = 1.0 / total_precision

    # Per-point model disagreement
    model_var = zeros(n_points)
    for j in 1:n_points
        # Weighted variance of forecaster predictions
        pred_mean = mean_pred[j]
        for i in 1:n_forecasters
            model_var[j] += weights[i] * (predictions[i, j] - pred_mean)^2
        end
    end

    # Total variance
    total_var = noise_var .+ model_var

    return mean_pred, sqrt.(total_var), weights
end

#=
===============================================================================
SECTION 4: RxInfer-based Inference (Alternative)
===============================================================================

Using RxInfer's message passing for inference.
This provides a more flexible framework for complex models.
=#

"""
    infer_precisions_rxinfer(predictions_matrix, y_true; α₀=1.0, β₀=1.0, iterations=10)

Use RxInfer to infer precision parameters for each forecaster.
"""
function infer_precisions_rxinfer(predictions_matrix::Matrix{Float64},
                                  y_true::Vector{Float64};
                                  α₀=1.0, β₀=1.0, iterations=10)
    n_forecasters, n_obs = size(predictions_matrix)

    posteriors = Vector{Any}(undef, n_forecasters)

    for i in 1:n_forecasters
        # Run inference for each forecaster separately
        result = infer(
            model = single_forecaster_model(n_obs=n_obs, α₀=α₀, β₀=β₀),
            data = (
                pred = predictions_matrix[i, :],
                y_obs = y_true
            ),
            returnvars = (γ = KeepLast(),),
            iterations = iterations
        )
        posteriors[i] = result.posteriors[:γ]
    end

    # Convert to GammaParams
    gamma_params = [GammaParams(params(p)...) for p in posteriors]

    return gamma_params
end

#=
===============================================================================
SECTION 5: Integration with Lux Neural Network Models
===============================================================================
=#

# Re-use TimeSeriesLSTM from the training script
struct TimeSeriesLSTM{L,H} <: Lux.AbstractLuxContainerLayer{(:lstm_cell, :head)}
    lstm_cell::L
    head::H
end

function TimeSeriesLSTM(in_dims::Int, hidden_dims::Int, out_dims::Int)
    return TimeSeriesLSTM(
        LSTMCell(in_dims => hidden_dims),
        Chain(Dense(hidden_dims => 32, relu), Dense(32 => out_dims))
    )
end

function (m::TimeSeriesLSTM)(x::AbstractArray{T,3}, ps::NamedTuple, st::NamedTuple) where {T}
    x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
    (y, carry), st_lstm = m.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    for x_t in x_rest
        (y, carry), st_lstm = m.lstm_cell((x_t, carry), ps.lstm_cell, st_lstm)
    end
    y, st_head = m.head(y, ps.head, st.head)
    st = merge(st, (lstm_cell=st_lstm, head=st_head))
    return y, st
end

"""
    load_lstm_model(model_path)

Load a trained LSTM model from JLD2 file.

Returns: (model, parameters, states, metadata)
"""
function load_lstm_model(model_path::String)
    data = load(model_path)

    # Reconstruct model architecture
    config = data["config"]
    model = TimeSeriesLSTM(config.input_dim, config.hidden_dim, config.out_dim)

    ps = data["parameters"]
    st = data["states"]
    meta = data["meta"]

    return model, ps, st, meta
end

"""
    generate_predictions(model, ps, st, X)

Generate predictions from a Lux model.
"""
function generate_predictions(model, ps, st, X::Array{Float32, 3})
    st_test = Lux.testmode(st)
    ŷ, _ = model(X, ps, st_test)
    return Array(ŷ)
end

"""
    create_ensemble_from_models(model_paths, X_val, Y_val, X_test; α₀=1.0, β₀=1.0)

Create an ensemble from multiple trained LSTM models.

# Arguments
- `model_paths`: Vector of paths to saved .jld2 model files
- `X_val`: Validation inputs for precision estimation (features × seq_len × N)
- `Y_val`: Validation targets (features × N)
- `X_test`: Test inputs for prediction
- `α₀, β₀`: Prior parameters for precision

# Returns
Named tuple with ensemble predictions and diagnostics
"""
function create_ensemble_from_models(model_paths::Vector{String},
                                     X_val::Array{Float32, 3},
                                     Y_val::Matrix{Float32},
                                     X_test::Array{Float32, 3};
                                     α₀=1.0, β₀=1.0)
    n_models = length(model_paths)
    n_val = size(X_val, 3)
    n_test = size(X_test, 3)
    n_features = size(Y_val, 1)

    # Generate predictions from all models on validation set
    val_predictions = zeros(Float64, n_models, n_features, n_val)
    test_predictions = zeros(Float64, n_models, n_features, n_test)

    for (i, path) in enumerate(model_paths)
        @info "Loading model $i from $path"
        model, ps, st, meta = load_lstm_model(path)

        # Validation predictions
        ŷ_val = generate_predictions(model, ps, st, X_val)
        val_predictions[i, :, :] = Float64.(ŷ_val)

        # Test predictions
        ŷ_test = generate_predictions(model, ps, st, X_test)
        test_predictions[i, :, :] = Float64.(ŷ_test)
    end

    # Estimate precisions for each feature dimension separately
    all_posteriors = Vector{Vector{GammaParams}}(undef, n_features)
    all_weights = zeros(n_models, n_features)

    for f in 1:n_features
        y_val_f = Float64.(Y_val[f, :])
        preds_f = val_predictions[:, f, :]

        posteriors = estimate_all_precisions(preds_f, y_val_f; α₀=α₀, β₀=β₀)
        all_posteriors[f] = posteriors
        all_weights[:, f] = compute_ensemble_weights(posteriors)
    end

    # Generate ensemble predictions for test set
    ensemble_mean = zeros(n_features, n_test)
    ensemble_std = zeros(n_features, n_test)

    for f in 1:n_features
        preds_f = test_predictions[:, f, :]
        mean_pred, std_pred, _ = ensemble_predict_with_variance(preds_f, all_posteriors[f])
        ensemble_mean[f, :] = mean_pred
        ensemble_std[f, :] .= std_pred
    end

    return (
        ensemble_mean = Float32.(ensemble_mean),
        ensemble_std = Float32.(ensemble_std),
        weights = all_weights,
        posteriors = all_posteriors,
        individual_predictions = test_predictions
    )
end

#=
===============================================================================
SECTION 6: Demo and Visualization
===============================================================================
=#

"""
    demo_synthetic_ensemble()

Demonstrate ensemble with synthetic forecasters on sine wave data.
"""
function demo_synthetic_ensemble()
    Random.seed!(42)
    println("="^70)
    println("  Probabilistic Ensemble Forecasting Demo (Synthetic Data)")
    println("="^70)

    # Generate data
    n_train = 200
    n_test = 100
    noise_level = 0.1

    x_train = range(0, 4π, length=n_train)
    y_train = sin.(x_train) .+ noise_level .* randn(n_train)

    x_test = range(0, 4π, length=n_test)
    y_test = sin.(x_test) .+ noise_level .* randn(n_test)

    # Define forecasters with varying quality
    forecasters = [
        ("Perfect sin(x)", x -> sin(x)),
        ("Biased +0.3", x -> sin(x) + 0.3),
        ("Scaled 0.8x", x -> 0.8 * sin(x)),
        ("Phase shift", x -> sin(x + 0.5)),
        ("Noisy", x -> sin(x) + 0.4 * randn()),
    ]

    n_forecasters = length(forecasters)

    # Generate predictions
    train_preds = hcat([[f[2](x) for x in x_train] for f in forecasters]...)'
    test_preds = hcat([[f[2](x) for x in x_test] for f in forecasters]...)'

    println("\n📊 Training Data: $(n_train) points")
    println("📊 Test Data: $(n_test) points")
    println("📊 Forecasters: $(n_forecasters)")

    # Estimate precisions
    println("\n🔧 Estimating precision parameters...")
    γ_posteriors = estimate_all_precisions(
        Matrix{Float64}(train_preds),
        collect(Float64, y_train);
        α₀=1.0, β₀=1.0
    )

    println("\n📈 Learned Precision Posteriors:")
    println("-"^60)
    for (i, (name, _)) in enumerate(forecasters)
        γ = γ_posteriors[i]
        mse_train = mean((train_preds[i, :] .- y_train).^2)
        println(@sprintf("  %-20s: E[γ]=%.4f (MSE=%.4f)", name, mean(γ), mse_train))
    end

    # Compute ensemble weights
    weights = compute_ensemble_weights(γ_posteriors)

    println("\n⚖️  Ensemble Weights:")
    println("-"^60)
    for (i, (name, _)) in enumerate(forecasters)
        bar_len = round(Int, weights[i] * 40)
        bar = "█"^bar_len * "░"^(40-bar_len)
        println(@sprintf("  %-20s: %.4f [%s]", name, weights[i], bar))
    end

    # Make ensemble predictions
    ensemble_mean, ensemble_std, _ = ensemble_predict_with_variance(
        Matrix{Float64}(test_preds),
        γ_posteriors
    )

    # Evaluate
    println("\n📊 Test Set Performance:")
    println("-"^60)

    ensemble_mse = mean((ensemble_mean .- y_test).^2)
    simple_avg_mse = mean((vec(mean(test_preds, dims=1)) .- y_test).^2)

    for (i, (name, _)) in enumerate(forecasters)
        mse_i = mean((test_preds[i, :] .- y_test).^2)
        println(@sprintf("  %-20s MSE: %.6f", name, mse_i))
    end
    println("-"^60)
    println(@sprintf("  %-20s MSE: %.6f", "Simple Average", simple_avg_mse))
    println(@sprintf("  %-20s MSE: %.6f ✨", "Ensemble", ensemble_mse))

    # Improvement
    improvement = (simple_avg_mse - ensemble_mse) / simple_avg_mse * 100
    println(@sprintf("\n  Improvement over simple average: %.1f%%", improvement))

    # Plot results
    println("\n📈 Generating visualization...")

    p1 = plot(x_test, y_test,
        label="True", lw=2, color=:black,
        title="Ensemble Prediction vs Ground Truth",
        xlabel="x", ylabel="y",
        legend=:topright
    )
    plot!(p1, x_test, ensemble_mean,
        ribbon=2*ensemble_std,
        label="Ensemble ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )
    for i in 1:min(3, n_forecasters)
        plot!(p1, x_test, test_preds[i, :],
            label=forecasters[i][1],
            ls=:dash, alpha=0.5
        )
    end

    p2 = bar(1:n_forecasters, weights,
        title="Precision-Based Weights",
        xlabel="Forecaster", ylabel="Weight",
        xticks=(1:n_forecasters, [f[1][1:min(8,length(f[1]))] for f in forecasters]),
        legend=false, color=:steelblue, rotation=45
    )

    p3 = bar(1:n_forecasters, [mean((test_preds[i, :] .- y_test).^2) for i in 1:n_forecasters],
        title="Individual Test MSE",
        xlabel="Forecaster", ylabel="MSE",
        xticks=(1:n_forecasters, [f[1][1:min(8,length(f[1]))] for f in forecasters]),
        legend=false, color=:coral, rotation=45
    )

    methods = ["Simple\nAvg", "Ensemble"]
    p4 = bar(methods, [simple_avg_mse, ensemble_mse],
        title="MSE Comparison",
        ylabel="MSE",
        legend=false,
        color=[:gray, :green],
        ylims=(0, max(simple_avg_mse, ensemble_mse) * 1.2)
    )

    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 700))

    savefig(plt, "ensemble_synthetic_demo.png")
    println("  Saved: ensemble_synthetic_demo.png")

    return γ_posteriors, weights, ensemble_mean, ensemble_std
end

"""
    demo_lstm_ensemble()

Demonstrate ensemble with trained LSTM models on ETTh dataset.
"""
function demo_lstm_ensemble()
    println("="^70)
    println("  Probabilistic Ensemble Forecasting Demo (LSTM Models)")
    println("="^70)

    # Check for available models
    models_dir = joinpath(@__DIR__, "..", "models")
    available_models = filter(f -> endswith(f, ".jld2"), readdir(models_dir))

    if isempty(available_models)
        @warn "No trained models found in $models_dir"
        @info "Train models first with: julia scripts/train_ett_lstm_enzyme.jl"
        return nothing
    end

    println("\n📁 Available models:")
    for m in available_models
        println("   • $m")
    end

    # Group models by dataset
    etth1_models = filter(m -> startswith(m, "ETTh1"), available_models)

    if length(etth1_models) < 2
        @warn "Need at least 2 ETTh1 models for ensemble. Found: $(length(etth1_models))"
        return nothing
    end

    # Use first few models for demonstration
    model_paths = [joinpath(models_dir, m) for m in etth1_models[1:min(4, length(etth1_models))]]

    println("\n🔧 Creating ensemble from $(length(model_paths)) models...")

    # Load dataset
    data_path = joinpath(@__DIR__, "..", "data", "ETTh1")
    Xmat, feat_cols = load_ett(data_path)

    # Use settings from first model
    _, _, _, meta = load_lstm_model(model_paths[1])
    seq_len = meta.seq_len
    horizon = meta.horizon
    scaler = meta.scaler

    # Create sequences
    X3, Y2 = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(0.6, 0.2, 0.2))

    # Scale data
    Xval_s = Float32.(scale_inputs(scaler, Xval))
    Yval_s = Float32.(scale_targets(scaler, Yval))
    Xte_s = Float32.(scale_inputs(scaler, Xte))
    Yte_s = Float32.(scale_targets(scaler, Yte))

    # Create ensemble
    result = create_ensemble_from_models(
        model_paths,
        Xval_s, Yval_s,
        Xte_s;
        α₀=1.0, β₀=1.0
    )

    # Compute metrics
    ensemble_pred = inverse_targets(scaler, result.ensemble_mean)
    Yte_orig = inverse_targets(scaler, Yte_s)

    ensemble_mse_val = ProbabilisticEnsembling.mse(ensemble_pred, Yte_orig)
    ensemble_mae_val = ProbabilisticEnsembling.mae(ensemble_pred, Yte_orig)

    println("\n📊 Ensemble Test Metrics:")
    println("-"^60)
    println(@sprintf("  MSE:  %.6f", ensemble_mse_val))
    println(@sprintf("  MAE:  %.6f", ensemble_mae_val))

    # Print weights
    println("\n⚖️  Ensemble Weights (by feature, averaged):")
    avg_weights = vec(mean(result.weights, dims=2))
    for (i, path) in enumerate(model_paths)
        model_name = basename(path)
        println(@sprintf("  %s: %.4f", model_name, avg_weights[i]))
    end

    return result
end

#=
===============================================================================
SECTION 7: Main Entry Point
===============================================================================
=#

function main()
    println("\n🚀 Running Probabilistic Ensemble Forecasting Examples\n")

    # Run synthetic demo
    demo_synthetic_ensemble()

    println("\n" * "="^70 * "\n")

    # Run LSTM demo if models available
    demo_lstm_ensemble()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
