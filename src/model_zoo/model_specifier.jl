using YAML

using Distributions
using Statistics
using JLD2
using Lux
using Reactant

export run_experiment

# prediction_type: univariate | multivariate
struct Univariate end
struct Multivariate end

struct ExperimentSpecifier{P,M,D}
    prediction_type::P
    model_type::M
    column::Union{String,Nothing}
    horizon::Int
    dataset::D            # Val{:ETTh1}, Val{:electricity}, etc. — dispatches load_dataset
    dataset_path::String  # absolute path to the CSV
    experts::Vector{String}
    priors::Dict{Symbol,Any}
    inference_iterations::Int
    prediction_iterations::Int
end

# ---------------------------------------------------------------------------
# YAML → ExperimentSpecifier
# ---------------------------------------------------------------------------

function parse_prediction_type(s::String)
    s == "univariate"  && return Univariate()
    s == "multivariate" && return Multivariate()
    error("Unknown prediction_type: $s")
end

function parse_model_type(s::String)
    s == "static"       && return Static()
    s == "dynamic"      && return Dynamic()
    s == "hierarchical" && return Hierarchical()
    error("Unknown model_type: $s")
end

function run_experiment(path_to_yaml::String)
    config = YAML.load_file(path_to_yaml)
    p      = config["params"]

    prediction_type       = parse_prediction_type(p["prediction_type"])
    model_type            = parse_model_type(p["model_type"])
    column                = get(p, "column", nothing)
    horizon               = p["horizon"]
    dataset               = Val(Symbol(p["dataset"]))
    dataset_path          = p["dataset_path"]
    experts               = String.(p["experts"])
    priors                = parse_priors(model_type, p["priors"], length(experts))
    inference_iterations  = p["inference_iterations"]
    prediction_iterations = p["prediction_iterations"]

    spec = ExperimentSpecifier(prediction_type, model_type, column, horizon, dataset, dataset_path, experts, priors, inference_iterations, prediction_iterations)
    return run_experiment(spec)
end

# ---------------------------------------------------------------------------
# Dispatch: ExperimentSpecifier → concrete pipeline
# ---------------------------------------------------------------------------

function run_experiment(spec::ExperimentSpecifier{Univariate,Static})
    return run_static_univariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Multivariate,Static})
    return run_static_multivariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Univariate,Dynamic})
    return run_dynamic_univariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Multivariate,Dynamic})
    return run_dynamic_multivariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Univariate,Hierarchical})
    return run_hierarchical_univariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Multivariate,Hierarchical})
    return run_hierarchical_multivariate(spec)
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

reactant_device() = try Reactant.default_device() catch; Lux.cpu_device() end
cpu_dev() = Lux.cpu_device()

function predict_unscaled(model, ps, st, X_scaled; dev=reactant_device())
    Xd      = dev(Float32.(X_scaled))
    st_test = Lux.testmode(st) |> dev
    ps_d    = dev(ps)
    model_c = dev isa ReactantDevice ? (@compile model(Xd, ps_d, st_test)) : model
    y_sc, _ = model_c(Xd, ps_d, st_test)
    return Array(cpu_dev()(y_sc))
end

function find_column_index(feat_cols::Vector{String}, column::String)
    idx = findfirst(==(column), feat_cols)
    isnothing(idx) && error("Column \"$column\" not found. Available: $feat_cols")
    return idx
end

function make_features(X_scaled)
    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        x_last     = Float64.(X_scaled[:, end, j])
        x_last_cos = map(cos, X_scaled[:, end, j])
        x_last_sin = map(sin, X_scaled[:, end, j])
        feats[j] = vcat(1.0, x_last, x_last_cos, x_last_sin)
    end
    return feats
end

# ---------------------------------------------------------------------------
# Dataset loading — dispatch on Val{:dataset_name}
# Add new methods for other dataset formats, e.g.:
#   load_dataset(::Val{:electricity}, path::String) = ...
# ---------------------------------------------------------------------------

load_dataset(::Val{:ETTh1}, path::String) = load_ett(path)
load_dataset(::Val{:ETTh2}, path::String) = load_ett(path)
load_dataset(::Val{:electricity}, path::String) = load_ett(path)
load_dataset(::Val{:exchange_rate}, path::String) = load_ett(path)

# ---------------------------------------------------------------------------
# Expert prediction generation
# ---------------------------------------------------------------------------

function generate_expert_predictions(::Univariate, experts, scaler, Xval_s, Xte_s, col_idx)
    n_forecasters = length(experts)
    n_val  = size(Xval_s, 3)
    n_test = size(Xte_s, 3)

    predictions_val  = Matrix{Float64}(undef, n_forecasters, n_val)
    predictions_test = Matrix{Float64}(undef, n_forecasters, n_test)

    @info "Generating expert predictions" n_forecasters n_val n_test
    for (i, m) in enumerate(experts)
        model    = build_model(m.model_type, m.config)
        yhat_val = inverse_targets(scaler, predict_unscaled(model, m.parameters, m.states, Xval_s))
        yhat_te  = inverse_targets(scaler, predict_unscaled(model, m.parameters, m.states, Xte_s))
        predictions_val[i, :]  = Float64.(yhat_val[col_idx, :])
        predictions_test[i, :] = Float64.(yhat_te[col_idx, :])
        @info "Expert ready" index=i model_type=m.model_type
    end

    return predictions_val, predictions_test
end

function generate_expert_predictions(::Multivariate, experts, scaler, Xval_s, Xte_s)
    n_forecasters = length(experts)
    n_val  = size(Xval_s, 3)
    n_test = size(Xte_s, 3)

    predictions_val  = Matrix{Vector{Float64}}(undef, n_forecasters, n_val)
    predictions_test = Matrix{Vector{Float64}}(undef, n_forecasters, n_test)

    @info "Generating expert predictions (multivariate)" n_forecasters n_val n_test
    for (i, m) in enumerate(experts)
        model    = build_model(m.model_type, m.config)
        yhat_val = inverse_targets(scaler, predict_unscaled(model, m.parameters, m.states, Xval_s))
        yhat_te  = inverse_targets(scaler, predict_unscaled(model, m.parameters, m.states, Xte_s))
        for j in 1:n_val
            predictions_val[i, j] = Float64.(yhat_val[:, j])
        end
        for j in 1:n_test
            predictions_test[i, j] = Float64.(yhat_te[:, j])
        end
        @info "Expert ready" index=i model_type=m.model_type
    end

    return predictions_val, predictions_test
end

# ---------------------------------------------------------------------------
# Static Univariate pipeline
# ---------------------------------------------------------------------------

function run_static_univariate(spec::ExperimentSpecifier{Univariate,Static})
    # 1. Load expert models
    @info "Loading expert models" n=length(spec.experts)
    experts = map(load_jld2_model, spec.experts)
    base_meta = experts[1].meta

    # 2. Load raw data & split
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    col_idx = find_column_index(feat_cols, spec.column)

    seq_len = Int(base_meta.seq_len)
    horizon = Int(base_meta.horizon)
    X3, Y2  = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)

    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)

    y_val  = Float64.(Yval[col_idx, :])
    y_test = Float64.(Yte[col_idx, :])

    # 3. Generate expert predictions
    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type, experts, scaler, Xval_s, Xte_s, col_idx
    )

    n_forecasters = length(experts)
    n_test = length(y_test)

    # 4. Fit static ensemble on validation data
    @info "Fitting static ensemble on validation data"
    result = infer(
        model       = univariate_ensemble_precision_model(n_forecasters=n_forecasters, priors=spec.priors),
        data        = (y=y_val, X=predictions_val),
        iterations  = spec.inference_iterations,
        free_energy = true
    )

    free_energy  = result.free_energy
    γ_posteriors = result.posteriors[:γ][end]
    γ_means      = map(mean, γ_posteriors)
    weights      = γ_means ./ sum(γ_means)

    @info "Learned precision weights"
    for i in 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        @info "Expert $i" E_γ=round(γ_means[i]; digits=4) val_MSE=round(mse_i; digits=6) weight=round(weights[i]; digits=4)
    end

    # 5. Ensemble predictions on test
    @info "Generating ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(:γ => γ_posteriors)
    prediction_array = [missing for _ in 1:n_test]
    infer_test = infer(
        model      = univariate_ensemble_precision_model(n_forecasters=n_forecasters, priors=posterior_priors),
        data       = (y=prediction_array, X=predictions_test),
        iterations = spec.prediction_iterations
    )

    ensemble_preds = infer_test.predictions[:y][end]
    ensemble_mean  = map(mean, ensemble_preds)
    ensemble_std   = map(std, ensemble_preds)

    # 6. Metrics
    ensemble_metrics = (
        mse   = mse(ensemble_mean, y_test),
        mae   = mae(ensemble_mean, y_test),
        rmse  = rmse(ensemble_mean, y_test),
        r2    = r2(ensemble_mean, y_test),
        mape  = mape(ensemble_mean, y_test),
        smape = smape(ensemble_mean, y_test),
    )

    @info "Ensemble test metrics" ensemble_metrics...

    # 7. Save results
    results = (
        γ_posteriors     = γ_posteriors,
        weights          = weights,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = y_test,
        spec             = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type      = string(typeof(spec.model_type)),
            column          = spec.column,
            horizon         = spec.horizon,
            dataset         = typeof(spec.dataset).parameters[1],
            dataset_path    = spec.dataset_path,
            experts         = spec.experts,
        ),
    )

    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    fname = "$(ds_name)_h$(spec.horizon)_$(spec.column)_static.jld2"
    results_path = joinpath(results_dir, fname)
    JLD2.jldsave(results_path;
        γ_posteriors     = γ_posteriors,
        weights          = weights,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = y_test,
        spec             = results.spec,
    )
    @info "Results saved" path=results_path

    return results
end

# ---------------------------------------------------------------------------
# Static Multivariate pipeline
# ---------------------------------------------------------------------------

function run_static_multivariate(spec::ExperimentSpecifier{Multivariate,Static})
    # 1. Load expert models
    @info "Loading expert models" n=length(spec.experts)
    experts = map(load_jld2_model, spec.experts)
    base_meta = experts[1].meta

    # 2. Load raw data & split
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    d       = length(feat_cols)
    seq_len = Int(base_meta.seq_len)
    horizon = Int(base_meta.horizon)
    X3, Y2  = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)

    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    n_scaler = length(scaler.μ)
    n_feat   = size(Xval, 1)
    n_scaler == n_feat || error(
        "Scaler dimension ($n_scaler) from expert model does not match " *
        "dataset features ($n_feat). The expert was likely trained on a " *
        "different dataset. Check that `dataset_path` in the YAML matches " *
        "the experts' training data."
    )
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)

    n_val  = size(Xval, 3)
    n_test = size(Xte, 3)

    # Ground truth as vectors of vectors (for RxInfer)
    y_val  = [Float64.(Yval[:, j]) for j in 1:n_val]
    y_test = [Float64.(Yte[:, j])  for j in 1:n_test]

    # 3. Generate expert predictions
    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type, experts, scaler, Xval_s, Xte_s
    )

    n_forecasters = length(experts)

    # 4. Fit static ensemble on validation data
    @info "Fitting static multivariate ensemble on validation data" d n_forecasters
    result = infer(
        model       = multivariate_ensemble_precision_model(n_forecasters=n_forecasters, priors=spec.priors),
        data        = (y=y_val, X=predictions_val),
        iterations  = spec.inference_iterations,
        free_energy = true,
        showprogress = true
    )

    free_energy  = result.free_energy
    γ_posteriors = result.posteriors[:γ][end]
    γ_means      = map(mean, γ_posteriors)
    weights      = γ_means ./ sum(γ_means)

    # Per-expert validation MSE (multivariate)
    Yval_mat = Float64.(Yval)
    @info "Learned precision weights"
    for i in 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])  # (d, n_val)
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        @info "Expert $i" E_γ=round(γ_means[i]; digits=4) val_MSE=round(mse_i; digits=6) weight=round(weights[i]; digits=4)
    end

    # 5. Ensemble predictions on test
    @info "Generating ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(:γ => γ_posteriors)
    prediction_array = [missing for _ in 1:n_test]
    infer_test = infer(
        model      = multivariate_ensemble_precision_model(n_forecasters=n_forecasters, priors=posterior_priors),
        data       = (y=prediction_array, X=predictions_test),
        iterations = spec.prediction_iterations
    )

    ensemble_preds = infer_test.predictions[:y][end]
    ensemble_mean  = reduce(hcat, map(mean, ensemble_preds))  # (d, n_test)
    ensemble_std   = reduce(hcat, map(std, ensemble_preds))   # (d, n_test)

    # 6. Metrics (multivariate)
    Yte_mat = Float64.(Yte)
    ensemble_metrics = (
        mse   = mse_mv(ensemble_mean, Yte_mat),
        mae   = mae_mv(ensemble_mean, Yte_mat),
        rmse  = rmse_mv(ensemble_mean, Yte_mat),
        r2    = r2_mv(ensemble_mean, Yte_mat),
        mape  = mape_mv(ensemble_mean, Yte_mat),
        smape = smape_mv(ensemble_mean, Yte_mat),
    )

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # 7. Save results
    results = (
        γ_posteriors     = γ_posteriors,
        weights          = weights,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = Yte_mat,
        spec             = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type      = string(typeof(spec.model_type)),
            column          = nothing,
            horizon         = spec.horizon,
            dataset         = typeof(spec.dataset).parameters[1],
            dataset_path    = spec.dataset_path,
            experts         = spec.experts,
        ),
    )

    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    fname = "$(ds_name)_h$(spec.horizon)_multivariate_static.jld2"
    results_path = joinpath(results_dir, fname)
    JLD2.jldsave(results_path;
        γ_posteriors     = γ_posteriors,
        weights          = weights,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = Yte_mat,
        spec             = results.spec,
    )
    @info "Results saved" path=results_path

    return results
end

function run_dynamic_univariate(spec::ExperimentSpecifier{Univariate,Dynamic})
    # 1. Load expert models
    @info "Loading expert models" n=length(spec.experts)
    experts = map(load_jld2_model, spec.experts)
    base_meta = experts[1].meta

    # 2. Load raw data & split
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    col_idx = find_column_index(feat_cols, spec.column)

    seq_len = Int(base_meta.seq_len)
    horizon = Int(base_meta.horizon)
    X3, Y2  = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)

    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)

    y_val  = Float64.(Yval[col_idx, :])
    y_test = Float64.(Yte[col_idx, :])

    # 3. Generate expert predictions
    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type, experts, scaler, Xval_s, Xte_s, col_idx
    )

    n_forecasters = length(experts)
    n_val  = length(y_val)
    n_test = length(y_test)

    # 4. Construct features from scaled inputs
    features_val  = make_features(Xval_s)
    features_test = make_features(Xte_s)

    # 5. Fit dynamic ensemble on validation data
    @info "Fitting dynamic ensemble on validation data" n_forecasters n_val
    result = infer(
        model       = univariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs         = n_val,
            priors        = spec.priors,
        ),
        data          = (y = y_val, features = features_val, predictions = predictions_val),
        constraints   = univariate_dynamic_ensemble_constraints(),
        initialization = univariate_dynamic_ensemble_init(spec.priors),
        iterations    = spec.inference_iterations,
        free_energy   = true,
        showprogress  = true,
    )

    free_energy    = result.free_energy
    w_posteriors   = result.posteriors[:w][end]
    τ_posteriors   = result.posteriors[:τ][end]
    β_posteriors   = result.posteriors[:β][end]
    γ_posteriors   = result.posteriors[:γ][end]  # (n_forecasters, n_val)

    γ_means_val = mean.(γ_posteriors)

    @info "Learned dynamic weights on validation"
    for i in 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ=round(avg_γ; digits=4) val_MSE=round(mse_i; digits=6)
    end

    # 6. Ensemble predictions on test
    @info "Generating dynamic ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(:w => w_posteriors, :τ => τ_posteriors, :β => β_posteriors)
    prediction_array = [missing for _ in 1:n_test]

    infer_test = infer(
        model       = univariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs         = n_test,
            priors        = posterior_priors,
        ),
        data          = (y = prediction_array, features = features_test, predictions = predictions_test),
        constraints   = univariate_dynamic_ensemble_constraints(),
        initialization = univariate_dynamic_ensemble_init(posterior_priors),
        iterations    = spec.prediction_iterations,
        free_energy   = false,
        showprogress  = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    ensemble_mean  = map(mean, ensemble_preds)
    ensemble_std   = map(std, ensemble_preds)

    # Extract dynamic precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test      = mean.(γ_test_posteriors)

    # 7. Metrics
    ensemble_metrics = (
        mse   = mse(ensemble_mean, y_test),
        mae   = mae(ensemble_mean, y_test),
        rmse  = rmse(ensemble_mean, y_test),
        r2    = r2(ensemble_mean, y_test),
        mape  = mape(ensemble_mean, y_test),
        smape = smape(ensemble_mean, y_test),
    )

    @info "Ensemble test metrics" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        β_posteriors     = β_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = y_test,
        spec             = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type      = string(typeof(spec.model_type)),
            column          = spec.column,
            horizon         = spec.horizon,
            dataset         = typeof(spec.dataset).parameters[1],
            dataset_path    = spec.dataset_path,
            experts         = spec.experts,
        ),
    )

    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    fname = "$(ds_name)_h$(spec.horizon)_$(spec.column)_dynamic.jld2"
    results_path = joinpath(results_dir, fname)
    JLD2.jldsave(results_path;
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        β_posteriors     = β_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = y_test,
        spec             = results.spec,
    )
    @info "Results saved" path=results_path

    return results
end

# ---------------------------------------------------------------------------
# Dynamic Multivariate pipeline
# ---------------------------------------------------------------------------

function run_dynamic_multivariate(spec::ExperimentSpecifier{Multivariate,Dynamic})
    # 1. Load expert models
    @info "Loading expert models" n=length(spec.experts)
    experts = map(load_jld2_model, spec.experts)
    base_meta = experts[1].meta

    # 2. Load raw data & split
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    d       = length(feat_cols)
    seq_len = Int(base_meta.seq_len)
    horizon = Int(base_meta.horizon)
    X3, Y2  = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)

    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    n_scaler = length(scaler.μ)
    n_feat   = size(Xval, 1)
    n_scaler == n_feat || error(
        "Scaler dimension ($n_scaler) from expert model does not match " *
        "dataset features ($n_feat). The expert was likely trained on a " *
        "different dataset. Check that `dataset_path` in the YAML matches " *
        "the experts' training data."
    )
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)

    n_val  = size(Xval, 3)
    n_test = size(Xte, 3)

    # Ground truth as vectors of vectors (for RxInfer)
    y_val  = [Float64.(Yval[:, j]) for j in 1:n_val]
    y_test = [Float64.(Yte[:, j])  for j in 1:n_test]

    # 3. Generate expert predictions
    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type, experts, scaler, Xval_s, Xte_s
    )

    n_forecasters = length(experts)

    # 4. Construct features from scaled inputs
    features_val  = make_features(Xval_s)
    features_test = make_features(Xte_s)

    # 5. Fit dynamic ensemble on validation data
    @info "Fitting dynamic multivariate ensemble on validation data" d n_forecasters n_val
    result = infer(
        model       = multivariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs         = n_val,
            priors        = spec.priors,
        ),
        data          = (y = y_val, features = features_val, predictions = predictions_val),
        constraints   = multivariate_dynamic_ensemble_constraints(),
        initialization = multivariate_dynamic_ensemble_init(spec.priors),
        iterations    = spec.inference_iterations,
        free_energy   = true,
        showprogress  = true,
    )

    free_energy    = result.free_energy
    w_posteriors   = result.posteriors[:w][end]
    τ_posteriors   = result.posteriors[:τ][end]
    β_posteriors   = result.posteriors[:β][end]
    γ_posteriors   = result.posteriors[:γ][end]  # (n_forecasters, n_val)

    γ_means_val = mean.(γ_posteriors)

    # Per-expert validation MSE (multivariate)
    Yval_mat = Float64.(Yval)
    @info "Learned dynamic weights on validation"
    for i in 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])  # (d, n_val)
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ=round(avg_γ; digits=4) val_MSE=round(mse_i; digits=6)
    end

    # 6. Ensemble predictions on test
    @info "Generating dynamic ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(:w => w_posteriors, :τ => τ_posteriors, :β => β_posteriors)
    prediction_array = [missing for _ in 1:n_test]

    infer_test = infer(
        model = multivariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs         = n_test,
            priors        = posterior_priors,
        ),
        data          = (y = prediction_array, features = features_test, predictions = predictions_test),
        constraints   = multivariate_dynamic_ensemble_constraints(),
        initialization = multivariate_dynamic_ensemble_init(posterior_priors),
        iterations    = spec.prediction_iterations,
        free_energy   = false,
        showprogress  = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    ensemble_mean  = reduce(hcat, map(mean, ensemble_preds))  # (d, n_test)
    ensemble_std   = reduce(hcat, map(std, ensemble_preds))   # (d, n_test)

    # Extract dynamic precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test      = mean.(γ_test_posteriors)

    # 7. Metrics (multivariate)
    Yte_mat = Float64.(Yte)
    ensemble_metrics = (
        mse   = mse_mv(ensemble_mean, Yte_mat),
        mae   = mae_mv(ensemble_mean, Yte_mat),
        rmse  = rmse_mv(ensemble_mean, Yte_mat),
        r2    = r2_mv(ensemble_mean, Yte_mat),
        mape  = mape_mv(ensemble_mean, Yte_mat),
        smape = smape_mv(ensemble_mean, Yte_mat),
    )

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        β_posteriors     = β_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = Yte_mat,
        spec             = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type      = string(typeof(spec.model_type)),
            column          = nothing,
            horizon         = spec.horizon,
            dataset         = typeof(spec.dataset).parameters[1],
            dataset_path    = spec.dataset_path,
            experts         = spec.experts,
        ),
    )

    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    fname = "$(ds_name)_h$(spec.horizon)_multivariate_dynamic.jld2"
    results_path = joinpath(results_dir, fname)
    JLD2.jldsave(results_path;
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        β_posteriors     = β_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = Yte_mat,
        spec             = results.spec,
    )
    @info "Results saved" path=results_path

    return results
end

# ---------------------------------------------------------------------------
# Hierarchical Univariate pipeline
# ---------------------------------------------------------------------------

function run_hierarchical_univariate(spec::ExperimentSpecifier{Univariate,Hierarchical})
    # 1. Load expert models
    @info "Loading expert models" n=length(spec.experts)
    experts = map(load_jld2_model, spec.experts)
    base_meta = experts[1].meta

    # 2. Load raw data & split
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    col_idx = find_column_index(feat_cols, spec.column)

    seq_len = Int(base_meta.seq_len)
    horizon = Int(base_meta.horizon)
    X3, Y2  = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)

    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)

    y_val  = Float64.(Yval[col_idx, :])
    y_test = Float64.(Yte[col_idx, :])

    # 3. Generate expert predictions
    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type, experts, scaler, Xval_s, Xte_s, col_idx
    )

    n_forecasters = length(experts)
    n_val  = length(y_val)
    n_test = length(y_test)

    # 4. Construct features from scaled inputs
    features_val  = make_features(Xval_s)
    features_test = make_features(Xte_s)

    # 5. Fit hierarchical ensemble on validation data
    @info "Fitting hierarchical ensemble on validation data" n_forecasters n_val
    result = infer(
        model = hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs         = n_val,
            priors        = spec.priors,
        ),
        data          = (y = y_val, features = features_val, predictions = predictions_val),
        constraints   = hierarchical_constraints(),
        initialization = hierarchical_init(spec.priors),
        iterations    = spec.inference_iterations,
        free_energy   = true,
        showprogress  = true,
    )

    free_energy    = result.free_energy
    w_posteriors   = result.posteriors[:w][end]
    τ_posteriors   = result.posteriors[:τ][end]
    ρ_posteriors   = result.posteriors[:ρ][end]
    γ_posteriors   = result.posteriors[:γ][end]  # (n_forecasters, n_val)

    γ_means_val = mean.(γ_posteriors)

    @info "Learned hierarchical weights on validation"
    for i in 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ=round(avg_γ; digits=4) val_MSE=round(mse_i; digits=6)
    end

    # 6. Ensemble predictions on test
    @info "Generating hierarchical ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(
        :w => w_posteriors,
        :τ => τ_posteriors,
        :ρ => ρ_posteriors,
        :α => spec.priors[:α],
    )
    prediction_array = [missing for _ in 1:n_test]

    infer_test = infer(
        model       = hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs         = n_test,
            priors        = posterior_priors,
        ),
        data          = (y = prediction_array, features = features_test, predictions = predictions_test),
        constraints   = hierarchical_constraints(),
        initialization = hierarchical_init(posterior_priors),
        iterations    = spec.prediction_iterations,
        free_energy   = false,
        showprogress  = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    ensemble_mean  = map(mean, ensemble_preds)
    ensemble_std   = map(std, ensemble_preds)

    # Extract precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test      = mean.(γ_test_posteriors)

    # 7. Metrics
    ensemble_metrics = (
        mse   = mse(ensemble_mean, y_test),
        mae   = mae(ensemble_mean, y_test),
        rmse  = rmse(ensemble_mean, y_test),
        r2    = r2(ensemble_mean, y_test),
        mape  = mape(ensemble_mean, y_test),
        smape = smape(ensemble_mean, y_test),
    )

    @info "Ensemble test metrics" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        ρ_posteriors     = ρ_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = y_test,
        spec             = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type      = string(typeof(spec.model_type)),
            column          = spec.column,
            horizon         = spec.horizon,
            dataset         = typeof(spec.dataset).parameters[1],
            dataset_path    = spec.dataset_path,
            experts         = spec.experts,
        ),
    )

    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    fname = "$(ds_name)_h$(spec.horizon)_$(spec.column)_hierarchical.jld2"
    results_path = joinpath(results_dir, fname)
    JLD2.jldsave(results_path;
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        ρ_posteriors     = ρ_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = y_test,
        spec             = results.spec,
    )
    @info "Results saved" path=results_path

    return results
end

# ---------------------------------------------------------------------------
# Hierarchical Multivariate pipeline
# ---------------------------------------------------------------------------

function run_hierarchical_multivariate(spec::ExperimentSpecifier{Multivariate,Hierarchical})
    # 1. Load expert models
    @info "Loading expert models" n=length(spec.experts)
    experts = map(load_jld2_model, spec.experts)
    base_meta = experts[1].meta

    # 2. Load raw data & split
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    d       = length(feat_cols)
    seq_len = Int(base_meta.seq_len)
    horizon = Int(base_meta.horizon)
    X3, Y2  = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)

    split = base_meta.split
    _, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    n_scaler = length(scaler.μ)
    n_feat   = size(Xval, 1)
    n_scaler == n_feat || error(
        "Scaler dimension ($n_scaler) from expert model does not match " *
        "dataset features ($n_feat). The expert was likely trained on a " *
        "different dataset. Check that `dataset_path` in the YAML matches " *
        "the experts' training data."
    )
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s  = scale_inputs(scaler, Xte)

    n_val  = size(Xval, 3)
    n_test = size(Xte, 3)

    # Ground truth as vectors of vectors (for RxInfer)
    y_val  = [Float64.(Yval[:, j]) for j in 1:n_val]
    y_test = [Float64.(Yte[:, j])  for j in 1:n_test]

    # 3. Generate expert predictions
    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type, experts, scaler, Xval_s, Xte_s
    )

    n_forecasters = length(experts)

    # 4. Construct features from scaled inputs
    features_val  = make_features(Xval_s)
    features_test = make_features(Xte_s)

    # 5. Fit hierarchical ensemble on validation data
    @info "Fitting hierarchical multivariate ensemble on validation data" d n_forecasters n_val
    result = infer(
        model       = multivariate_hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs         = n_val,
            priors        = spec.priors,
        ),
        data          = (y = y_val, features = features_val, predictions = predictions_val),
        constraints   = multivariate_hierarchical_constraints(),
        initialization = multivariate_hierarchical_init(spec.priors),
        iterations    = spec.inference_iterations,
        free_energy   = true,
        showprogress  = true,
    )

    free_energy    = result.free_energy
    w_posteriors   = result.posteriors[:w][end]
    τ_posteriors   = result.posteriors[:τ][end]
    ρ_posteriors   = result.posteriors[:ρ][end]
    γ_posteriors   = result.posteriors[:γ][end]

    γ_means_val = mean.(γ_posteriors)

    # Per-expert validation MSE (multivariate)
    Yval_mat = Float64.(Yval)
    @info "Learned hierarchical weights on validation"
    for i in 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ=round(avg_γ; digits=4) val_MSE=round(mse_i; digits=6)
    end

    # 6. Ensemble predictions on test
    @info "Generating hierarchical ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(
        :w => w_posteriors,
        :τ => τ_posteriors,
        :ρ => ρ_posteriors,
        :α => spec.priors[:α],
    )
    prediction_array = [missing for _ in 1:n_test]

    infer_test = infer(
        model       = multivariate_hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs         = n_test,
            priors        = posterior_priors,
        ),
        data          = (y = prediction_array, features = features_test, predictions = predictions_test),
        constraints   = multivariate_hierarchical_constraints(),
        initialization = multivariate_hierarchical_init(posterior_priors),
        iterations    = spec.prediction_iterations,
        free_energy   = false,
        showprogress  = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    ensemble_mean  = reduce(hcat, map(mean, ensemble_preds))  # (d, n_test)
    ensemble_std   = reduce(hcat, map(std, ensemble_preds))   # (d, n_test)

    # Extract precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test      = mean.(γ_test_posteriors)

    # 7. Metrics (multivariate)
    Yte_mat = Float64.(Yte)
    ensemble_metrics = (
        mse   = mse_mv(ensemble_mean, Yte_mat),
        mae   = mae_mv(ensemble_mean, Yte_mat),
        rmse  = rmse_mv(ensemble_mean, Yte_mat),
        r2    = r2_mv(ensemble_mean, Yte_mat),
        mape  = mape_mv(ensemble_mean, Yte_mat),
        smape = smape_mv(ensemble_mean, Yte_mat),
    )

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        ρ_posteriors     = ρ_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = Yte_mat,
        spec             = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type      = string(typeof(spec.model_type)),
            column          = nothing,
            horizon         = spec.horizon,
            dataset         = typeof(spec.dataset).parameters[1],
            dataset_path    = spec.dataset_path,
            experts         = spec.experts,
        ),
    )

    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    fname = "$(ds_name)_h$(spec.horizon)_multivariate_hierarchical.jld2"
    results_path = joinpath(results_dir, fname)
    JLD2.jldsave(results_path;
        w_posteriors     = w_posteriors,
        τ_posteriors     = τ_posteriors,
        ρ_posteriors     = ρ_posteriors,
        γ_test           = γ_means_test,
        free_energy      = free_energy,
        ensemble_mean    = ensemble_mean,
        ensemble_std     = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test           = Yte_mat,
        spec             = results.spec,
    )
    @info "Results saved" path=results_path

    return results
end