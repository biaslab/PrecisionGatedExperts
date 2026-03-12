# ---------------------------------------------------------------------------
# Neural Ensemble Pipeline — Adaptive Mixture of Local Experts
# ---------------------------------------------------------------------------

using LinearAlgebra

# ---------------------------------------------------------------------------
# Feature dispatch helper (similar to _before_rxinfer_features)
# ---------------------------------------------------------------------------

function _neural_ensemble_features(
    feature_type::SimpleFeatures,
    X_scaled,
    _dataset,
    _col_idx,
)
    return make_features(feature_type, X_scaled)
end

function _neural_ensemble_features(
    feature_type::WindowFeatures,
    X_scaled,
    _dataset,
    _col_idx,
)
    return make_features(feature_type, X_scaled)
end

function _neural_ensemble_features(feature_type::FFTFeatures, X_scaled, _dataset, _col_idx)
    return make_features(feature_type, X_scaled)
end

function _neural_ensemble_features(
    feature_type::UniWindowFeatures,
    X_scaled,
    _dataset,
    col_idx,
)
    col_idx === nothing && error(
        "UniWindowFeatures requires a column index (use prediction_type: univariate with a column)",
    )
    return make_features(feature_type, X_scaled, col_idx)
end

function _neural_ensemble_features(feature_type::VAEFeatures, X_scaled, dataset, _col_idx)
    return make_features(feature_type, X_scaled, dataset)
end

function _neural_ensemble_features(feature_type::AEFeatures, X_scaled, dataset, _col_idx)
    return make_features(feature_type, X_scaled, dataset)
end

# ---------------------------------------------------------------------------
# Quantile baselines — added from train target distribution
# ---------------------------------------------------------------------------

function add_quantile_baselines!(
    predictions_train,
    predictions_val,
    predictions_test,
    Ytr_sc,
    selected_quantiles,
)
    n_model_forecasters = size(predictions_train, 1)
    d = size(predictions_train, 2)
    n_train = size(predictions_train, 3)
    n_val = size(predictions_val, 3)
    n_test = size(predictions_test, 3)
    n_quantile = length(selected_quantiles)

    n_total = n_model_forecasters + n_quantile
    out_train = Array{Float64}(undef, n_total, d, n_train)
    out_val = Array{Float64}(undef, n_total, d, n_val)
    out_test = Array{Float64}(undef, n_total, d, n_test)

    out_train[1:n_model_forecasters, :, :] = predictions_train
    out_val[1:n_model_forecasters, :, :] = predictions_val
    out_test[1:n_model_forecasters, :, :] = predictions_test

    for (offset, q_pct) in enumerate(selected_quantiles)
        idx = n_model_forecasters + offset
        q_vec = [quantile(Float64.(view(Ytr_sc, i, :)), q_pct / 100.0) for i = 1:d]
        for j = 1:n_train
            out_train[idx, :, j] = q_vec
        end
        for j = 1:n_val
            out_val[idx, :, j] = q_vec
        end
        for j = 1:n_test
            out_test[idx, :, j] = q_vec
        end
    end

    @info "Added quantile baselines" quantiles = selected_quantiles n_total
    return out_train, out_val, out_test
end

# ---------------------------------------------------------------------------
# Convert 3D predictions array to vec format for gating
# ---------------------------------------------------------------------------

function to_predictions_vec(predictions::Array{Float64,3})
    n_experts, d, n_samples = size(predictions)
    out = Array{Vector{Float64}}(undef, n_experts, n_samples)
    for i = 1:n_experts
        for j = 1:n_samples
            out[i, j] = Vector{Float64}(predictions[i, :, j])
        end
    end
    return out
end

function to_target_vecs(Y::AbstractMatrix)
    return [Vector{Float64}(Y[:, j]) for j = 1:size(Y, 2)]
end

# ---------------------------------------------------------------------------
# Restrict predictions and targets to a single column (Univariate)
# ---------------------------------------------------------------------------

function restrict_to_column(
    predictions_vec::Array{Vector{Float64},2},
    y_vecs::Vector{Vector{Float64}},
    col_idx::Int,
)
    n_experts, n_samples = size(predictions_vec)
    restricted_preds = Array{Vector{Float64}}(undef, n_experts, n_samples)
    for i = 1:n_experts
        for j = 1:n_samples
            restricted_preds[i, j] = [predictions_vec[i, j][col_idx]]
        end
    end
    restricted_y = [Vector{Float64}([y[col_idx]]) for y in y_vecs]
    return restricted_preds, restricted_y
end

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

function before_neural_ensemble(spec::NeuralEnsembleSpecifier{Univariate})
    @info "Loading expert models" n = length(spec.experts)
    experts = map(load_jld2_model, spec.experts)

    length(experts) < 2 && error("Need at least 2 expert models, got $(length(experts))")

    base_meta = experts[1].meta
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)
    col_idx = find_column_index(feat_cols, spec.column)

    seq_len = Int(base_meta.seq_len)
    horizon = spec.horizon
    X3, Y2 = make_sequences(Xmat; seq_len = seq_len, horizon = horizon)

    split = base_meta.split
    Xtr, Ytr, Xval, Yval, Xte, Yte =
        train_val_test_split(X3, Y2; ratios = (split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xtr_s = scale_inputs(scaler, Xtr)
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Ytr_sc = scale_targets(scaler, Ytr)
    Yval_sc = scale_targets(scaler, Yval)
    Yte_sc = scale_targets(scaler, Yte)

    pred_train, pred_val, pred_test =
        generate_expert_predictions_three_splits(experts, Xtr_s, Xval_s, Xte_s)
    pred_train, pred_val, pred_test = add_quantile_baselines!(
        pred_train,
        pred_val,
        pred_test,
        Ytr_sc,
        spec.selected_quantiles,
    )

    n_total = size(pred_train, 1)
    d = size(pred_train, 2)

    # Full multivariate vec format (for test evaluation)
    predictions_train_vec = to_predictions_vec(pred_train)
    predictions_val_vec = to_predictions_vec(pred_val)
    predictions_test_vec = to_predictions_vec(pred_test)

    y_train = to_target_vecs(Float64.(Ytr_sc))
    y_val = to_target_vecs(Float64.(Yval_sc))
    y_test = to_target_vecs(Float64.(Yte_sc))

    # Restrict to target column for gating training
    predictions_train_vec_moe, y_train_moe =
        restrict_to_column(predictions_train_vec, y_train, col_idx)
    predictions_val_vec_moe, y_val_moe =
        restrict_to_column(predictions_val_vec, y_val, col_idx)

    features_train =
        _neural_ensemble_features(spec.feature_type, Xtr_s, spec.dataset, col_idx)
    features_val =
        _neural_ensemble_features(spec.feature_type, Xval_s, spec.dataset, col_idx)
    features_test =
        _neural_ensemble_features(spec.feature_type, Xte_s, spec.dataset, col_idx)
    n_features = length(features_train[1])

    return (
        n_total = n_total,
        d = d,
        col_idx = col_idx,
        n_features = n_features,
        predictions_train_vec_moe = predictions_train_vec_moe,
        predictions_val_vec_moe = predictions_val_vec_moe,
        predictions_test_vec = predictions_test_vec,
        y_train_moe = y_train_moe,
        y_val_moe = y_val_moe,
        y_test_mat = Float64.(Yte_sc),
        features_train = features_train,
        features_val = features_val,
        features_test = features_test,
        scaler = scaler,
    )
end

function before_neural_ensemble(spec::NeuralEnsembleSpecifier{Multivariate})
    @info "Loading expert models" n = length(spec.experts)
    experts = map(load_jld2_model, spec.experts)

    length(experts) < 2 && error("Need at least 2 expert models, got $(length(experts))")

    base_meta = experts[1].meta
    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    seq_len = Int(base_meta.seq_len)
    horizon = spec.horizon
    X3, Y2 = make_sequences(Xmat; seq_len = seq_len, horizon = horizon)

    split = base_meta.split
    Xtr, Ytr, Xval, Yval, Xte, Yte =
        train_val_test_split(X3, Y2; ratios = (split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xtr_s = scale_inputs(scaler, Xtr)
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Ytr_sc = scale_targets(scaler, Ytr)
    Yval_sc = scale_targets(scaler, Yval)
    Yte_sc = scale_targets(scaler, Yte)

    pred_train, pred_val, pred_test =
        generate_expert_predictions_three_splits(experts, Xtr_s, Xval_s, Xte_s)
    pred_train, pred_val, pred_test = add_quantile_baselines!(
        pred_train,
        pred_val,
        pred_test,
        Ytr_sc,
        spec.selected_quantiles,
    )

    n_total = size(pred_train, 1)
    d = size(pred_train, 2)

    predictions_train_vec = to_predictions_vec(pred_train)
    predictions_val_vec = to_predictions_vec(pred_val)
    predictions_test_vec = to_predictions_vec(pred_test)

    y_train = to_target_vecs(Float64.(Ytr_sc))
    y_val = to_target_vecs(Float64.(Yval_sc))

    features_train =
        _neural_ensemble_features(spec.feature_type, Xtr_s, spec.dataset, nothing)
    features_val =
        _neural_ensemble_features(spec.feature_type, Xval_s, spec.dataset, nothing)
    features_test =
        _neural_ensemble_features(spec.feature_type, Xte_s, spec.dataset, nothing)
    n_features = length(features_train[1])

    return (
        n_total = n_total,
        d = d,
        col_idx = nothing,
        n_features = n_features,
        predictions_train_vec_moe = predictions_train_vec,
        predictions_val_vec_moe = predictions_val_vec,
        predictions_test_vec = predictions_test_vec,
        y_train_moe = y_train,
        y_val_moe = y_val,
        y_test_mat = Float64.(Yte_sc),
        features_train = features_train,
        features_val = features_val,
        features_test = features_test,
        scaler = scaler,
    )
end

# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------

function create_normal_prediction(::Univariate, gating, ps, st, predictions, features)
    logits, _ = gating(Float32.(features), ps, st)
    precisions = map(exp, logits)
    normals = [
        NormalMeanPrecision(pred, precision) for
        (pred, precision) in zip(predictions, precisions)
    ]
    normal_prediction = reduce((x, y) -> prod(BayesBase.GenericProd(), x, y), normals)
    return normal_prediction
end

function create_normal_prediction(::Multivariate, gating, ps, st, predictions, features)
    logits, _ = gating(Float32.(features), ps, st)
    precisions = map(exp, logits)
    normals = [
        MvNormalMeanScalePrecision(pred, prec) for
        (pred, prec) in zip(predictions, precisions)
    ]
    normal_prediction = reduce((x, y) -> prod(BayesBase.GenericProd(), x, y), normals)
    return normal_prediction
end

function evaluate_neural_ensemble(
    ::Univariate,
    gating,
    ps,
    st,
    predictions_test_vec,
    features_test,
    y_test_mat,
    col_idx,
)
    n_test = size(predictions_test_vec, 2)

    ensemble_mean = hcat(
        [
            moe_predict(predictions_test_vec[:, j], gating, ps, st, features_test[j])
            for j = 1:n_test
        ]...,
    )
    ensemble_std = sqrt.(
        hcat(
            [
                moe_var(predictions_test_vec[:, j], gating, ps, st, features_test[j])
                for j = 1:n_test
            ]...,
        ),
    )
    normal_predictions = [
        create_normal_prediction(
            Univariate(),
            gating,
            ps,
            st,
            [predictions_test_vec[i, j][col_idx] for i = 1:size(predictions_test_vec, 1)],
            features_test[j],
        ) for j = 1:n_test
    ]
    gating_weights =
        hcat([gating_probs(gating, ps, st, features_test[j]) for j = 1:n_test]...)

    # Evaluate on target column only
    y_eval = y_test_mat[col_idx:col_idx, :]
    ensemble_eval = ensemble_mean[col_idx:col_idx, :]
    y_test = [y_test_mat[col_idx, j] for j = 1:n_test]

    # CI95 from Bayesian product-of-experts posterior
    ci95_lower = [
        mean(normal_predictions[j]) - ZSCORE_95 * std(normal_predictions[j]) for
        j = 1:n_test
    ]
    ci95_upper = [
        mean(normal_predictions[j]) + ZSCORE_95 * std(normal_predictions[j]) for
        j = 1:n_test
    ]
    ci95_target_overlap = mean((y_test .>= ci95_lower) .& (y_test .<= ci95_upper))
    ci95_avg_width = mean(ci95_upper .- ci95_lower)
    ci95_interval_score = mean(
        (ci95_upper .- ci95_lower) .+
        (2 / ALPHA_95) .* ((ci95_lower .- y_test) .* (y_test .< ci95_lower)) .+
        (2 / ALPHA_95) .* ((y_test .- ci95_upper) .* (y_test .> ci95_upper)),
    )

    ensemble_metrics = (
        mse = mse_mv(ensemble_eval, y_eval),
        mae = mae_mv(ensemble_eval, y_eval),
        rmse = rmse_mv(ensemble_eval, y_eval),
        r2 = r2_mv(ensemble_eval, y_eval),
        mape = mape_mv(ensemble_eval, y_eval),
        smape = smape_mv(ensemble_eval, y_eval),
        nll = mean(
            map((y_dist) -> logpdf(y_dist[2], y_dist[1]), zip(y_test, normal_predictions)),
        ),
        ci95_target_overlap = ci95_target_overlap,
        ci95_avg_width = ci95_avg_width,
        ci95_interval_score = ci95_interval_score,
    )

    return (
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        gating_weights = gating_weights,
        normal_predictions = normal_predictions,
    )
end

function evaluate_neural_ensemble(
    ::Multivariate,
    gating,
    ps,
    st,
    predictions_test_vec,
    features_test,
    y_test_mat,
    _col_idx,
)
    n_test = size(predictions_test_vec, 2)

    ensemble_mean = hcat(
        [
            moe_predict(predictions_test_vec[:, j], gating, ps, st, features_test[j])
            for j = 1:n_test
        ]...,
    )
    ensemble_std = sqrt.(
        hcat(
            [
                moe_var(predictions_test_vec[:, j], gating, ps, st, features_test[j])
                for j = 1:n_test
            ]...,
        ),
    )
    normal_predictions = [
        create_normal_prediction(
            Multivariate(),
            gating,
            ps,
            st,
            predictions_test_vec[:, j],
            features_test[j],
        ) for j = 1:n_test
    ]
    gating_weights =
        hcat([gating_probs(gating, ps, st, features_test[j]) for j = 1:n_test]...)

    # NLL from MvNormal product-of-experts posterior
    nll = mean([logpdf(normal_predictions[j], y_test_mat[:, j]) for j = 1:n_test])

    # CI95 from Bayesian product-of-experts posterior
    posterior_mean = reduce(hcat, map(mean, normal_predictions))
    posterior_std = reduce(hcat, map(_marginal_std, normal_predictions))
    ci95_lower = posterior_mean .- ZSCORE_95 .* posterior_std
    ci95_upper = posterior_mean .+ ZSCORE_95 .* posterior_std
    ci95_target_overlap = mean((y_test_mat .>= ci95_lower) .& (y_test_mat .<= ci95_upper))
    ci95_avg_width = mean(ci95_upper .- ci95_lower)
    ci95_interval_score = mean(
        (ci95_upper .- ci95_lower) .+
        (2 / ALPHA_95) .* ((ci95_lower .- y_test_mat) .* (y_test_mat .< ci95_lower)) .+
        (2 / ALPHA_95) .* ((y_test_mat .- ci95_upper) .* (y_test_mat .> ci95_upper)),
    )

    ensemble_metrics = (
        mse = mse_mv(ensemble_mean, y_test_mat),
        mae = mae_mv(ensemble_mean, y_test_mat),
        rmse = rmse_mv(ensemble_mean, y_test_mat),
        r2 = r2_mv(ensemble_mean, y_test_mat),
        mape = mape_mv(ensemble_mean, y_test_mat),
        smape = smape_mv(ensemble_mean, y_test_mat),
        nll = nll,
        ci95_target_overlap = ci95_target_overlap,
        ci95_avg_width = ci95_avg_width,
        ci95_interval_score = ci95_interval_score,
    )

    return (
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        gating_weights = gating_weights,
        normal_predictions = normal_predictions,
    )
end

# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

function save_neural_ensemble(
    save_dir,
    spec,
    gating_config,
    gating_state,
    results,
    raw_spec,
)
    mkpath(save_dir)

    ds_name = string(typeof(spec.dataset).parameters[1])
    config_hash = bytes2hex(sha256(repr(raw_spec)))[1:8]
    filename = "$(ds_name)_h$(spec.horizon)_neural_ensemble_$(config_hash).jld2"
    path = joinpath(save_dir, filename)

    JLD2.jldsave(
        path;
        gating_config = gating_config,
        gating_parameters = gating_state.parameters,
        gating_states = gating_state.states,
        best_epoch = gating_state.best_epoch,
        best_monitor_loss = gating_state.best_monitor_loss,
        ensemble_metrics = results.ensemble_metrics,
        ensemble_mean = results.ensemble_mean,
        ensemble_std = results.ensemble_std,
        gating_weights = results.gating_weights,
        spec = (
            prediction_type = spec.prediction_type isa Univariate ? "univariate" :
                              "multivariate",
            column = spec.column,
            dataset = ds_name,
            dataset_path = spec.dataset_path,
            horizon = spec.horizon,
            experts = spec.experts,
            train_set = spec.train_set,
            feature_type = string(typeof(spec.feature_type)),
            selected_quantiles = spec.selected_quantiles,
            gating_layers = spec.gating_layers,
            gating_hidden_dim = spec.gating_hidden_dim,
        ),
        raw_spec = raw_spec,
    )

    @info "Saved neural ensemble model" path
    return path
end

function load_neural_ensemble(path::String)
    data = JLD2.load(path)
    return data
end

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

function run_neural_ensemble(spec::NeuralEnsembleSpecifier, raw_spec::Dict)
    data = before_neural_ensemble(spec)

    gating_cfg = (
        n_features = data.n_features,
        n_experts = data.n_total,
        layers = spec.gating_layers,
        hidden_dim = spec.gating_hidden_dim,
    )
    gating = build_gating_network(
        gating_cfg.n_features,
        gating_cfg.n_experts,
        gating_cfg.layers,
        gating_cfg.hidden_dim,
    )
    opt = Optimisers.Adam(spec.learning_rate)

    @info "Training gating network" n_features = data.n_features n_experts = data.n_total layers =
        spec.gating_layers hidden_dim = spec.gating_hidden_dim train_set = spec.train_set

    gating_state = if spec.train_set
        train_moe!(
            data.predictions_train_vec_moe,
            data.features_train,
            data.y_train_moe,
            data.predictions_val_vec_moe,
            data.features_val,
            data.y_val_moe,
            gating,
            opt;
            n_epochs = spec.n_epochs,
            patience = spec.patience,
            min_delta = spec.min_delta,
            monitor_label = "val",
        )
    else
        train_moe!(
            data.predictions_val_vec_moe,
            data.features_val,
            data.y_val_moe,
            data.predictions_train_vec_moe,
            data.features_train,
            data.y_train_moe,
            gating,
            opt;
            n_epochs = spec.n_epochs,
            patience = 1,
            min_delta = 1.0f-3,
            monitor_label = "train",
        )
    end

    @info "Best gating checkpoint" best_epoch = gating_state.best_epoch best_monitor_loss =
        gating_state.best_monitor_loss

    results = evaluate_neural_ensemble(
        spec.prediction_type,
        gating,
        gating_state.parameters,
        gating_state.states,
        data.predictions_test_vec,
        data.features_test,
        data.y_test_mat,
        data.col_idx,
    )

    @info "Neural ensemble metrics" results.ensemble_metrics...

    saved_path = save_neural_ensemble(
        spec.save_dir,
        spec,
        gating_cfg,
        gating_state,
        results,
        raw_spec,
    )

    return (
        ensemble_mean = results.ensemble_mean,
        ensemble_std = results.ensemble_std,
        ensemble_metrics = results.ensemble_metrics,
        gating_weights = results.gating_weights,
        gating_state = gating_state,
        saved_path = saved_path,
    )
end
