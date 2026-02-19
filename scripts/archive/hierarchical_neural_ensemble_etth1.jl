#!/usr/bin/env julia

"""
Hierarchical Probabilistic Ensemble (α=1) on ETTh1 — OT column only.

Runs for all four horizons (96, 192, 336, 720), loading CNN/LSTM/MLP
forecasters for each. Uses the hierarchical model with scalar
NormalMeanPrecision (since y = OT is univariate) and RxInfer-native
predictions.  Compares with simple average baseline.

Usage:
    julia scripts/hierarchical_neural_ensemble_etth1.jl
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

# =============================================================================
# Model (hierarchical, α = 1, scalar y)
# =============================================================================

@model function hierarchical_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors, ρ_priors, α_fixed)
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
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function hierarchical_constraints()
    q(w, z, β, γ, τ, ρ) = q(w)q(z, β)q(γ)q(τ)q(ρ)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(β) :: ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
end

@initialization function hierarchical_init(w_init, τ_init, ρ_init)
    q(w) = w_init
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(β) = GammaShapeScale(1.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = τ_init
    q(ρ) = ρ_init
end

# =============================================================================
# Helpers
# =============================================================================

reactant_device() = (
    try
        Reactant.default_device()
    catch
        Lux.cpu_device()
    end
)
cpu_device() = Lux.cpu_device()

function same_scaler(s1, s2; atol = 1.0f-3)
    length(s1.μ) == length(s2.μ) || return false
    length(s1.σ) == length(s2.σ) || return false
    return maximum(abs.(s1.μ .- s2.μ)) ≤ atol && maximum(abs.(s1.σ .- s2.σ)) ≤ atol
end

function predict_unscaled(model, ps, st, X_scaled; dev = reactant_device())
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

# =============================================================================
# Run one horizon
# =============================================================================

function run_horizon(horizon::Int, dataset::AbstractString = "ETTh1"; models_dir = joinpath(@__DIR__, "..", "models"),
                     data_dir = joinpath(@__DIR__, "..", "data"),
                     n_iterations = 30, α = 1.0)

    # --- Discover and load models ---
    pattern = "$(dataset)_h$(horizon)_"
    paths = filter(f -> startswith(basename(f), pattern) && contains(basename(f), "_s"),
                   readdir(models_dir; join = true))
    paths = filter(f -> endswith(f, ".jld2"), paths)

    if isempty(paths)
        @warn "No models found for $(dataset) h=$horizon, skipping"
        return nothing
    end

    sort!(paths)
    models = map(load_jld2_model, paths)
    model_names = [splitext(basename(p))[1] for p in paths]

    base_meta = models[1].meta
    for m in models[2:end]
        @assert m.meta.dataset == base_meta.dataset
        @assert m.meta.seq_len == base_meta.seq_len
        @assert m.meta.horizon == base_meta.horizon
        @assert same_scaler(m.meta.scaler, base_meta.scaler)
    end

    @info "=" ^ 70
    @info "$(dataset)  horizon=$horizon  models=$(length(models)): $(model_names)"
    @info "=" ^ 70

    # --- Load & split data ---
    ds_path = joinpath(data_dir, String(base_meta.dataset))
    Xmat, feat_cols = load_ett(ds_path)
    d = size(Xmat, 1)          # total features (7 for ETTh1)
    ot_idx = d                  # OT is the last column

    X3, Y2 = make_sequences(Xmat; seq_len = Int(base_meta.seq_len), horizon = Int(base_meta.horizon))
    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios = (split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)
    Yval_sc = scale_targets(scaler, Yval)
    Yte_sc  = scale_targets(scaler, Yte)

    n_forecasters = length(models)
    n_train = size(Xval, 3)
    n_test  = size(Xte, 3)

    # --- Add constant baselines (OT q10 / q90 from val) ---
    n_total = n_forecasters + 2
    idx_min = n_forecasters + 1
    idx_max = n_forecasters + 2

    ot_val = Float64.(Yval_sc[ot_idx, :])
    ot_q10 = quantile(ot_val, 0.1)
    ot_q90 = quantile(ot_val, 0.9)

    # OT-only predictions: (n_total, n_samples)
    pred_train_ot = zeros(n_total, n_train)
    pred_test_ot  = zeros(n_total, n_test)

    # Full predictions for individual metrics
    pred_train_full = Array{Float64}(undef, n_total, d, n_train)
    pred_test_full  = Array{Float64}(undef, n_total, d, n_test)

    @info "Running forecasters" n_forecasters n_train n_test

    for (i, m) in enumerate(models)
        model = build_model(m.model_type, m.config)
        yhat_tr = predict_unscaled(model, m.parameters, m.states, Xval_s)
        yhat_te = predict_unscaled(model, m.parameters, m.states, Xte_s)

        pred_train_ot[i, :] = Float64.(yhat_tr[ot_idx, :])
        pred_test_ot[i, :]  = Float64.(yhat_te[ot_idx, :])

        pred_train_full[i, :, :] = Float64.(yhat_tr)
        pred_test_full[i, :, :]  = Float64.(yhat_te)

        @info "  Forecaster ready" index = i name = model_names[i]
    end

    # Constant baselines (OT only)
    pred_train_ot[idx_min, :] .= ot_q10
    pred_train_ot[idx_max, :] .= ot_q90
    pred_test_ot[idx_min, :]  .= ot_q10
    pred_test_ot[idx_max, :]  .= ot_q90

    # Full-dim baselines for simple-average comparison
    y_q10_full = [quantile(Float64.(Yval_sc[i, :]), 0.1) for i in 1:d]
    y_q90_full = [quantile(Float64.(Yval_sc[i, :]), 0.9) for i in 1:d]
    for j in 1:n_train
        pred_train_full[idx_min, :, j] = y_q10_full
        pred_train_full[idx_max, :, j] = y_q90_full
    end
    for j in 1:n_test
        pred_test_full[idx_min, :, j] = y_q10_full
        pred_test_full[idx_max, :, j] = y_q90_full
    end

    all_names = vcat(model_names, ["const_q10", "const_q90"])

    # --- Features ---
    features_train = make_features(Xval_s)
    features_test  = make_features(Xte_s)
    n_features = length(features_train[1])

    # --- y target: OT column only (scalar) ---
    y_train_ot = Float64.(Yval_sc[ot_idx, :])
    y_test_ot  = Float64.(Yte_sc[ot_idx, :])

    # =====================================================================
    # Training
    # =====================================================================
    @info "Training hierarchical ensemble (α=$α)" n_features n_total

    w_priors_init = [MvNormalMeanScalePrecision(zeros(n_features), 0.1) for _ in 1:n_total]
    τ_priors_init = [GammaShapeScale(1.0, 1e12) for _ in 1:n_total]
    ρ_priors_init = [GammaShapeRate(1.0, 1.0) for _ in 1:n_total]

    train_result = infer(
        model = hierarchical_model(
            n_forecasters = n_total,
            n_obs = n_train,
            w_priors = w_priors_init,
            τ_priors = τ_priors_init,
            ρ_priors = ρ_priors_init,
            α_fixed = α),
        data = (y = y_train_ot, features = features_train, predictions = pred_train_ot),
        constraints = hierarchical_constraints(),
        initialization = hierarchical_init(w_priors_init, τ_priors_init, ρ_priors_init),
        iterations = n_iterations, free_energy = false, showprogress = true
    )

    w_posteriors = train_result.posteriors[:w][end]
    τ_posteriors = train_result.posteriors[:τ][end]
    ρ_posteriors = train_result.posteriors[:ρ][end]

    # =====================================================================
    # Prediction via RxInfer
    # =====================================================================
    @info "Predicting on test set via RxInfer"

    y_missing = [missing for _ in 1:n_test]

    pred_result = infer(
        model = hierarchical_model(
            n_forecasters = n_total,
            n_obs = n_test,
            w_priors = w_posteriors,
            τ_priors = τ_posteriors,
            ρ_priors = ρ_posteriors,
            α_fixed = α),
        data = (y = y_missing, features = features_test, predictions = pred_test_ot),
        constraints = hierarchical_constraints(),
        initialization = hierarchical_init(w_posteriors, τ_posteriors, ρ_posteriors),
        iterations = 1, free_energy = false, showprogress = false
    )

    y_predictions = pred_result.predictions[:y][end]
    ensemble_pred = map(mean, y_predictions)
    ensemble_std  = map(std, y_predictions)

    γ_posteriors = pred_result.posteriors[:γ][end]
    γ_values = mean.(γ_posteriors)   # (n_total, n_test)

    # =====================================================================
    # Simple average baseline (OT only)
    # =====================================================================
    simple_avg_ot = vec(mean(pred_test_ot[1:n_forecasters, :]; dims = 1))

    # =====================================================================
    # Metrics (OT, scaled space)
    # =====================================================================
    @info "RESULTS  —  $(dataset) h=$horizon (OT column, scaled)"

    ens_mse   = mse(ensemble_pred, y_test_ot)
    ens_mae   = mae(ensemble_pred, y_test_ot)
    ens_rmse  = rmse(ensemble_pred, y_test_ot)
    ens_r2    = r2(ensemble_pred, y_test_ot)
    ens_smape = smape(ensemble_pred, y_test_ot)
    @info "  Hierarchical α=$α" MSE = ens_mse MAE = ens_mae RMSE = ens_rmse R2 = ens_r2 SMAPE = ens_smape mean_std = mean(ensemble_std)

    sa_mse   = mse(simple_avg_ot, y_test_ot)
    sa_mae   = mae(simple_avg_ot, y_test_ot)
    sa_rmse  = rmse(simple_avg_ot, y_test_ot)
    sa_r2    = r2(simple_avg_ot, y_test_ot)
    sa_smape = smape(simple_avg_ot, y_test_ot)
    @info "  Simple average" MSE = sa_mse MAE = sa_mae RMSE = sa_rmse R2 = sa_r2 SMAPE = sa_smape

    for i in 1:n_total
        fi_pred = pred_test_ot[i, :]
        @info "  $(all_names[i])" MSE = mse(fi_pred, y_test_ot) MAE = mae(fi_pred, y_test_ot) RMSE = rmse(fi_pred, y_test_ot) R2 = r2(fi_pred, y_test_ot) SMAPE = smape(fi_pred, y_test_ot)
    end

    # =====================================================================
    # Plots
    # =====================================================================
    mkpath("viz")
    prefix = "viz/hier_$(dataset)_h$(horizon)"
    x_ax = 1:n_test

    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan, :gray]

    # --- Predictions ---
    p1 = plot(x_ax, y_test_ot, label = "True OT", lw = 2, color = :black, ls = :dot,
              title = "$(dataset) h=$horizon — Hierarchical α=$α vs Forecasters (OT)",
              xlabel = "t", ylabel = "OT (scaled)", legend = :topright)
    plot!(p1, x_ax, ensemble_pred, ribbon = 2 .* ensemble_std,
          label = "Hierarchical ±2σ", lw = 2, color = :blue, fillalpha = 0.3)
    for i in 1:n_total
        plot!(p1, x_ax, pred_test_ot[i, :], label = all_names[i],
              ls = :dash, alpha = 0.5, color = colors[mod1(i, length(colors))])
    end

    # --- Simple avg vs hierarchical ---
    p2 = plot(x_ax, y_test_ot, label = "True OT", lw = 2, color = :black, ls = :dot,
              title = "Simple Avg vs Hierarchical (OT)",
              xlabel = "t", ylabel = "OT (scaled)", legend = :topright)
    plot!(p2, x_ax, simple_avg_ot, label = "Simple Avg", lw = 2, color = :orange, ls = :dash)
    plot!(p2, x_ax, ensemble_pred, label = "Hierarchical", lw = 2, color = :blue)

    # --- Precision weights ---
    p3 = plot(title = "Precision Weights γ(x)", xlabel = "t", ylabel = "γ", legend = :topright)
    for i in 1:n_total
        plot!(p3, x_ax, γ_values[i, :], label = all_names[i], lw = 2,
              color = colors[mod1(i, length(colors))])
    end

    # --- Uncertainty ---
    p4 = plot(x_ax, ensemble_std, label = "σ", lw = 2, color = :blue,
              title = "Uncertainty", xlabel = "t", ylabel = "σ", legend = :topright)

    # --- MSE bar chart ---
    all_mses = vcat([mse(pred_test_ot[i, :], y_test_ot) for i in 1:n_total],
                    [sa_mse, ens_mse])
    all_labels = vcat(all_names, ["Simple Avg", "Hierarchical"])
    bar_colors = vcat(fill(:gray, n_total), [:orange, :blue])
    p5 = bar(1:length(all_mses), all_mses,
             title = "MSE Comparison (OT)", xlabel = "Method", ylabel = "MSE",
             xticks = (1:length(all_mses), all_labels),
             legend = false, color = bar_colors, xrotation = 45)

    # --- Normalized weights ---
    norm_w = γ_values ./ sum(γ_values; dims = 1)
    p6 = plot(title = "Normalized Weights", xlabel = "t", ylabel = "weight", legend = :outerright)
    for i in 1:n_total
        plot!(p6, x_ax, norm_w[i, :], label = all_names[i], lw = 2,
              color = colors[mod1(i, length(colors))])
    end

    plt = plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), size = (1200, 1200))
    savefig(plt, "$(prefix).png")
    @info "Saved" file = "$(prefix).png"

    return (; horizon, ens_mse, ens_mae, ens_rmse, ens_r2, ens_smape,
              sa_mse, sa_mae, sa_rmse, sa_r2, sa_smape)
end

# =============================================================================
# Main — all horizons
# =============================================================================

function main()
    datasets = ["ETTh1", "ETTh2"]
    horizons = [96, 192, 336, 720]

    results_by_dataset = Dict{String, Vector{Any}}()

    # Run all experiments first.
    for dataset in datasets
        results = []

        for h in horizons
            r = run_horizon(h, dataset)
            r !== nothing && push!(results, r)
        end
        results_by_dataset[dataset] = results
    end

    # Print reports after all runs are done, so run logs don't appear between reports.
    for dataset in datasets
        results = get(results_by_dataset, dataset, Any[])
        # --- Summary table ---
        println("\n" * "=" ^ 90)
        println("SUMMARY  —  $(dataset) OT column, Hierarchical α=1 vs Simple Average")
        println("=" ^ 90)
        println(rpad("Horizon", 10),
                rpad("Method", 15),
                rpad("MSE", 12), rpad("MAE", 12), rpad("RMSE", 12),
                rpad("R²", 12), rpad("SMAPE", 12))
        println("-" ^ 90)
        for r in results
            println(rpad(r.horizon, 10),
                    rpad("Hierarchical", 15),
                    rpad(round(r.ens_mse,  digits = 6), 12),
                    rpad(round(r.ens_mae,  digits = 6), 12),
                    rpad(round(r.ens_rmse, digits = 6), 12),
                    rpad(round(r.ens_r2,   digits = 6), 12),
                    rpad(round(r.ens_smape, digits = 4), 12))
            println(rpad("", 10),
                    rpad("Simple Avg", 15),
                    rpad(round(r.sa_mse,  digits = 6), 12),
                    rpad(round(r.sa_mae,  digits = 6), 12),
                    rpad(round(r.sa_rmse, digits = 6), 12),
                    rpad(round(r.sa_r2,   digits = 6), 12),
                    rpad(round(r.sa_smape, digits = 4), 12))
            println("-" ^ 90)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
