#!/usr/bin/env julia

"""
Hierarchical Probabilistic Ensemble (RxInfer) for multivariate outputs.

Uses the dynamic neural ensemble pipeline with a hierarchical latent
precision model (beta-gamma chain with learnable rho priors) and evaluates
on all target columns.

Usage:
    julia scripts/hierarchical_neural_ensemble_mv.jl <model1.jld2> <model2.jld2> [more...]
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Statistics
using LinearAlgebra
using Lux
using Reactant
using ProbabilisticEnsembling
using Plots

# -----------------------------------------------------------------------------
# RxInfer hierarchical model (multivariate outputs)
# -----------------------------------------------------------------------------

@model function hierarchical_mv_ensemble_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors, ρ_priors, α_fixed)
    local w, z, β, γ, τ, ρ

    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
        ρ[i] ~ ρ_priors[i]
    end

    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])
            β[i, j] ~ GammaShapeRate(1.0, ρ[i])
            z[i, j] ~ Log(β[i, j])
            γ[i, j] ~ GammaShapeRate(α_fixed, β[i, j])
            y[j] ~ MvNormalMeanScalePrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function hierarchical_mv_constraints()
    q(w, z, β, γ, τ, ρ) = q(w)q(z, β)q(γ)q(τ)q(ρ)
    q(z)::ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(β)::ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ)::ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

@initialization function hierarchical_mv_init(w_init, τ_init, ρ_init)
    q(w) = w_init
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(β) = GammaShapeScale(1.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = τ_init
    q(ρ) = ρ_init
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

reactant_device() = (
    try
        Reactant.default_device()
    catch
        Lux.cpu_device()
    end
)
cpu_device() = Lux.cpu_device()

function same_scaler(s1, s2; atol=1.0f-3)
    length(s1.μ) == length(s2.μ) || return false
    length(s1.σ) == length(s2.σ) || return false
    return maximum(abs.(s1.μ .- s2.μ)) ≤ atol && maximum(abs.(s1.σ .- s2.σ)) ≤ atol
end

function predict_unscaled(model, ps, st, X_scaled; dev=reactant_device())
    Xd = dev(Float32.(X_scaled))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)
    model_compiled = if dev isa ReactantDevice
        @compile model(Xd, ps_d, st_test)
    else
        model
    end
    y_sc, _ = model_compiled(Xd, ps_d, st_test)
    y_sc = cpu_device()(y_sc)
    return Array(y_sc)
end

function to_vecs(Y::AbstractMatrix)
    return [Vector{Float64}(Y[:, j]) for j in 1:size(Y, 2)]
end

function make_features(X_scaled)
    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        x_last_cos = map(cos, X_scaled[:, end, j])
        x_last_sin = map(sin, X_scaled[:, end, j])
        x_last = Float64.(X_scaled[:, end, j])
        feats[j] = vcat(1.0, x_last, x_last_cos, x_last_sin)
    end
    return feats
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    if length(ARGS) < 2
        println("Usage: julia scripts/hierarchical_neural_ensemble_mv.jl <model1.jld2> <model2.jld2> [more...]")
        return
    end

    model_paths = ARGS
    models = map(load_jld2_model, model_paths)

    base_meta = models[1].meta
    for m in models[2:end]
        if m.meta.dataset != base_meta.dataset || m.meta.seq_len != base_meta.seq_len || m.meta.horizon != base_meta.horizon
            error("All models must share dataset, seq_len, and horizon. Got $(m.meta.dataset), seq_len=$(m.meta.seq_len), horizon=$(m.meta.horizon)")
        end
        if !same_scaler(m.meta.scaler, base_meta.scaler)
            error("All models must share the same scaler (train split).")
        end
    end

    @info "Loading dataset" dataset = base_meta.dataset seq_len = base_meta.seq_len horizon = base_meta.horizon

    data_dir = joinpath(@__DIR__, "..", "data")
    ds_path = joinpath(data_dir, String(base_meta.dataset))
    Xmat, _ = load_ett(ds_path)

    X3, Y2 = make_sequences(Xmat; seq_len=Int(base_meta.seq_len), horizon=Int(base_meta.horizon))
    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xens_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Yens_sc = scale_targets(scaler, Yval)
    Yte_sc = scale_targets(scaler, Yte)

    n_forecasters = length(models)
    n_train = size(Xval, 3)
    n_test = size(Xte, 3)
    d = size(Yval, 1)

    # Add two constant baselines: per-dimension q10 and q90 from ensemble-train targets
    n_total = n_forecasters + 2
    predictions_train = Array{Float64}(undef, n_total, d, n_train)
    predictions_test = Array{Float64}(undef, n_total, d, n_test)

    @info "Running forecasters" n_forecasters n_train n_test output_dim = d

    for (i, m) in enumerate(models)
        model = build_model(m.model_type, m.config)
        yhat_tr_sc = predict_unscaled(model, m.parameters, m.states, Xens_s)
        yhat_te_sc = predict_unscaled(model, m.parameters, m.states, Xte_s)

        predictions_train[i, :, :] = Float64.(yhat_tr_sc)
        predictions_test[i, :, :] = Float64.(yhat_te_sc)
        @info "Forecaster ready" index = i model_type = m.model_type path = model_paths[i]
    end

    y_train = to_vecs(Float64.(Yens_sc))

    y_q10 = [quantile(Float64.(view(Yens_sc, i, :)), 0.1) for i in 1:d]
    y_q90 = [quantile(Float64.(view(Yens_sc, i, :)), 0.9) for i in 1:d]
    idx_min = n_forecasters + 1
    idx_max = n_forecasters + 2
    for j in 1:n_train
        predictions_train[idx_min, :, j] = y_q10
        predictions_train[idx_max, :, j] = y_q90
    end
    for j in 1:n_test
        predictions_test[idx_min, :, j] = y_q10
        predictions_test[idx_max, :, j] = y_q90
    end
    @info "Added constant baselines" q10_idx = idx_min q90_idx = idx_max

    predictions_train_vec = Array{Vector{Float64}}(undef, n_total, n_train)
    predictions_test_vec = Array{Vector{Float64}}(undef, n_total, n_test)
    for i in 1:n_total
        for j in 1:n_train
            predictions_train_vec[i, j] = Vector{Float64}(predictions_train[i, :, j])
        end
        for j in 1:n_test
            predictions_test_vec[i, j] = Vector{Float64}(predictions_test[i, :, j])
        end
    end

    features_train = make_features(Xens_s)
    features_test = make_features(Xte_s)
    n_features = length(features_train[1])

    @info "Step 1: Training hierarchical ensemble using RxInfer" n_features = n_features
    α = 1.0
    w_priors_init = [MvNormalMeanScalePrecision(zeros(n_features), 0.1) for _ in 1:n_total]
    τ_priors_init = [GammaShapeScale(1.0, 1e12) for _ in 1:n_total]
    ρ_priors_init = [GammaShapeRate(1.0, 1.0) for _ in 1:n_total]

    train_result = infer(
        model=hierarchical_mv_ensemble_model(
            n_forecasters=n_total,
            n_obs=n_train,
            w_priors=w_priors_init,
            τ_priors=τ_priors_init,
            ρ_priors=ρ_priors_init,
            α_fixed=α,
        ),
        data=(y=y_train, features=features_train, predictions=predictions_train_vec),
        constraints=hierarchical_mv_constraints(),
        initialization=hierarchical_mv_init(w_priors_init, τ_priors_init, ρ_priors_init),
        iterations=30,
        free_energy=false,
        showprogress=true,
    )

    w_posteriors = train_result.posteriors[:w][end]
    τ_posteriors = train_result.posteriors[:τ][end]
    ρ_posteriors = train_result.posteriors[:ρ][end]

    @info "Step 2: Hierarchical ensemble prediction on test"
    y_missing = [missing for _ in 1:n_test]

    pred_result = infer(
        model=hierarchical_mv_ensemble_model(
            n_forecasters=n_total,
            n_obs=n_test,
            w_priors=w_posteriors,
            τ_priors=τ_posteriors,
            ρ_priors=ρ_posteriors,
            α_fixed=α,
        ),
        data=(y=y_missing, features=features_test, predictions=predictions_test_vec),
        constraints=hierarchical_mv_constraints(),
        initialization=hierarchical_mv_init(w_posteriors, τ_posteriors, ρ_posteriors),
        iterations=1,
        showprogress=true,
        free_energy=false,
    )

    y_predictions = pred_result.predictions[:y][end]
    ensemble_mean = hcat(map(mean, y_predictions)...)
    ensemble_std = map((a) -> a[1, 1], map(std, y_predictions))

    # γ posteriors for dynamic weights
    γ_posteriors = pred_result.posteriors[:γ][end]
    γ_values_all = mean.(γ_posteriors)

    y_test_mat = Float64.(Yte_sc)

    ensemble_metrics = (
        mse=mse_mv(ensemble_mean, y_test_mat),
        mae=mae_mv(ensemble_mean, y_test_mat),
        mean_std=mean(ensemble_std),
    )

    @info "Step 3: Performance comparison on test"
    individual = []
    for i in 1:n_total
        yhat = predictions_test[i, :, :]
        push!(individual, (
            path=i <= n_forecasters ? model_paths[i] : (i == idx_min ? "const_q10_train" : "const_q90_train"),
            mse=mse_mv(yhat, y_test_mat),
            mae=mae_mv(yhat, y_test_mat),
        ))
    end

    simple_avg = vec(mean(predictions_test; dims=1)) |> x -> reshape(x, d, n_test)
    simple_metrics = (
        mse=mse_mv(simple_avg, y_test_mat),
        mae=mae_mv(simple_avg, y_test_mat),
    )

    @info "Hierarchical ensemble metrics" ensemble_metrics...
    @info "Simple average metrics" simple_metrics...
    println("METRIC|kind=dynamic|mse=$(ensemble_metrics.mse)|mae=$(ensemble_metrics.mae)")
    println("METRIC|kind=average|mse=$(simple_metrics.mse)|mae=$(simple_metrics.mae)")
    for (i, m) in enumerate(individual)
        @info "Forecaster metrics" index = i path = m.path mse = m.mse mae = m.mae
        println("METRIC|kind=forecaster|index=$(i)|path=$(m.path)|mse=$(m.mse)|mae=$(m.mae)")
    end

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    x_test = 1:n_test
    y_test_mat_T = permutedims(y_test_mat, (2, 1))
    ensemble_mean_T = permutedims(ensemble_mean, (2, 1))
    simple_avg_T = permutedims(simple_avg, (2, 1))

    p1 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Hierarchical Ensemble vs Individual (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright
    )
    plot!(p1, x_test, ensemble_mean_T[:, 1],
        ribbon=2 .* ensemble_std,
        label="Hierarchical ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )

    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan, :gray, :black]
    for i in 1:n_total
        plot!(p1, x_test, predictions_test[i, 1, :],
            label="F$(i)", ls=:dash, alpha=0.6, color=colors[mod1(i, length(colors))]
        )
    end

    p2 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Simple Avg vs Hierarchical (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright
    )
    plot!(p2, x_test, simple_avg_T[:, 1],
        label="Simple Avg", lw=2, color=:orange, ls=:dash
    )
    plot!(p2, x_test, ensemble_mean_T[:, 1],
        label="Hierarchical", lw=2, color=:blue
    )

    p3 = plot(title="Hierarchical Precision Weights",
        xlabel="t", ylabel="γ(x)",
        legend=:topright
    )
    for i in 1:n_total
        plot!(p3, x_test, γ_values_all[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    p4 = plot(title="Uncertainty (Hierarchical)",
        xlabel="t", ylabel="Standard Deviation (σ)",
        legend=:topright
    )
    plot!(p4, x_test, ensemble_std, label="Hierarchical σ", lw=2, color=:blue)

    all_mses = vcat([m.mse for m in individual], [simple_metrics.mse, ensemble_metrics.mse])
    all_labels = vcat(["F$(i)" for i in 1:n_total], ["Simple Avg", "Hierarchical"])
    bar_colors = vcat(fill(:gray, n_total), [:orange, :blue])
    p5 = bar(1:length(all_mses), all_mses,
        title="MSE Comparison (Test Set, all columns)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45
    )

    normalized_weights = γ_values_all ./ sum(γ_values_all; dims=1)
    p6 = plot(title="Normalized Hierarchical Weights",
        xlabel="t", ylabel="Weight (normalized)",
        legend=:outerright
    )
    for i in 1:n_total
        plot!(p6, x_test, normalized_weights[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    plt = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1200, 1200))
    plot_file = "viz/hierarchical_neural_ensemble_mv_$(base_meta.dataset).png"
    savefig(plt, plot_file)
    @info "Saved visualization" file = plot_file

    @info "Done"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
