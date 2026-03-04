# ---------------------------------------------------------------------------
# Shared pipeline utilities used by both RxInfer and Neural Ensemble pipelines
# ---------------------------------------------------------------------------

using Lux
using Reactant
using Distributions
using Statistics

# prediction_type: univariate | multivariate
struct Univariate end
struct Multivariate end

function parse_prediction_type(s::String)
    s == "univariate" && return Univariate()
    s == "multivariate" && return Multivariate()
    error("Unknown prediction_type: $s")
end

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

reactant_device() =
    try
        Reactant.default_device()
    catch
        Lux.cpu_device()
    end
cpu_dev() = Lux.cpu_device()

# ---------------------------------------------------------------------------
# Metric constants
# ---------------------------------------------------------------------------

const ZSCORE_95 = 1.959963984540054
const ALPHA_95 = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _marginal_std(dist)
    s = std(dist)
    if s isa Number
        return [Float64(s)]
    elseif s isa AbstractVector
        return Float64.(s)
    elseif s isa AbstractMatrix
        return sqrt.(max.(diag(Matrix(s)), 0.0))
    end
    error("Unsupported std output type $(typeof(s)) for distribution $(typeof(dist))")
end

function predict_unscaled(model, ps, st, X_scaled; dev = reactant_device())
    Xd = dev(Float32.(X_scaled))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)
    model_c = dev isa ReactantDevice ? (@compile model(Xd, ps_d, st_test)) : model
    y_sc, _ = model_c(Xd, ps_d, st_test)
    return Array(cpu_dev()(y_sc))
end

# ---------------------------------------------------------------------------
# Ensemble metrics (RxInfer distribution-based)
# ---------------------------------------------------------------------------

function compute_ensemble_metrics(::Univariate, ensemble_preds, y_test)
    ensemble_mean = map(mean, ensemble_preds)
    ensemble_std = map(std, ensemble_preds)
    ci95_lower = ensemble_mean .- ZSCORE_95 .* ensemble_std
    ci95_upper = ensemble_mean .+ ZSCORE_95 .* ensemble_std
    ci95_target_overlap = mean((y_test .>= ci95_lower) .& (y_test .<= ci95_upper))
    ci95_avg_width = mean(ci95_upper .- ci95_lower)
    ci95_interval_score = mean(
        (ci95_upper .- ci95_lower) .+
        (2 / ALPHA_95) .* ((ci95_lower .- y_test) .* (y_test .< ci95_lower)) .+
        (2 / ALPHA_95) .* ((y_test .- ci95_upper) .* (y_test .> ci95_upper)),
    )
    ensemble_metrics = (
        mse = mse(ensemble_mean, y_test),
        mae = mae(ensemble_mean, y_test),
        rmse = rmse(ensemble_mean, y_test),
        r2 = r2(ensemble_mean, y_test),
        mape = mape(ensemble_mean, y_test),
        smape = smape(ensemble_mean, y_test),
        nll = mean(
            map((y_dist) -> logpdf(y_dist[2], y_dist[1]), zip(y_test, ensemble_preds)),
        ),
        ci95_target_overlap = ci95_target_overlap,
        ci95_avg_width = ci95_avg_width,
        ci95_interval_score = ci95_interval_score,
    )
    return (; ensemble_mean, ensemble_std, ensemble_metrics)
end

function compute_ensemble_metrics(::Multivariate, ensemble_preds, y_test)
    ensemble_mean = reduce(hcat, map(mean, ensemble_preds))
    ensemble_std = reduce(hcat, map(_marginal_std, ensemble_preds))
    ci95_lower = ensemble_mean .- ZSCORE_95 .* ensemble_std
    ci95_upper = ensemble_mean .+ ZSCORE_95 .* ensemble_std
    ci95_target_overlap = mean((y_test .>= ci95_lower) .& (y_test .<= ci95_upper))
    ci95_avg_width = mean(ci95_upper .- ci95_lower)
    ci95_interval_score = mean(
        (ci95_upper .- ci95_lower) .+
        (2 / ALPHA_95) .* ((ci95_lower .- y_test) .* (y_test .< ci95_lower)) .+
        (2 / ALPHA_95) .* ((y_test .- ci95_upper) .* (y_test .> ci95_upper)),
    )
    ensemble_metrics = (
        mse = mse_mv(ensemble_mean, y_test),
        mae = mae_mv(ensemble_mean, y_test),
        rmse = rmse_mv(ensemble_mean, y_test),
        r2 = r2_mv(ensemble_mean, y_test),
        mape = mape_mv(ensemble_mean, y_test),
        smape = smape_mv(ensemble_mean, y_test),
        nll = mean(
            map(
                (y_dist) -> logpdf(y_dist[2], y_dist[1]),
                zip(eachcol(y_test), ensemble_preds),
            ),
        ),
        ci95_target_overlap = ci95_target_overlap,
        ci95_avg_width = ci95_avg_width,
        ci95_interval_score = ci95_interval_score,
    )
    return (; ensemble_mean, ensemble_std, ensemble_metrics)
end

# ---------------------------------------------------------------------------
# Column index lookup
# ---------------------------------------------------------------------------

function find_column_index(feat_cols::Vector{String}, column::String)
    idx = findfirst(==(column), feat_cols)
    isnothing(idx) && error("Column \"$column\" not found. Available: $feat_cols")
    return idx
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
load_dataset(::Val{:traffic}, path::String) = load_ett(path)

# ---------------------------------------------------------------------------
# Expert prediction generation
# ---------------------------------------------------------------------------

function generate_expert_predictions(
    ::Univariate,
    experts,
    scaler,
    Xval_s,
    Xte_s,
    col_idx,
    selected_quantiles,
)
    n_model_forecasters = length(experts)
    n_quantile_forecasters = length(selected_quantiles)
    n_forecasters = n_model_forecasters + n_quantile_forecasters
    n_val = size(Xval_s, 3)
    n_test = size(Xte_s, 3)

    predictions_val = Matrix{Float64}(undef, n_forecasters, n_val)
    predictions_test = Matrix{Float64}(undef, n_forecasters, n_test)

    @info "Generating expert predictions" n_model_forecasters n_forecasters n_val n_test
    for (i, m) in enumerate(experts)
        model = build_model(m.model_type, m.config)
        yhat_val = predict_unscaled(model, m.parameters, m.states, Xval_s)
        yhat_te = predict_unscaled(model, m.parameters, m.states, Xte_s)
        predictions_val[i, :] = Float64.(yhat_val[col_idx, :])
        predictions_test[i, :] = Float64.(yhat_te[col_idx, :])
        @info "Expert ready" index = i model_type = m.model_type
    end

    if n_quantile_forecasters > 0
        x_last_val_scaled = Float64.(vec(Xval_s[col_idx, end, :]))
        for (offset, q_pct) in enumerate(selected_quantiles)
            idx = n_model_forecasters + offset
            q = quantile(x_last_val_scaled, q_pct / 100.0)
            predictions_val[idx, :] .= q
            predictions_test[idx, :] .= q
        end
        @info "Added constant quantile experts" quantiles = selected_quantiles
    else
        @info "No quantile experts selected"
    end

    return predictions_val, predictions_test
end

function generate_expert_predictions(
    ::Multivariate,
    experts,
    scaler,
    Xval_s,
    Xte_s,
    selected_quantiles,
)
    n_model_forecasters = length(experts)
    n_quantile_forecasters = length(selected_quantiles)
    n_forecasters = n_model_forecasters + n_quantile_forecasters
    n_val = size(Xval_s, 3)
    n_test = size(Xte_s, 3)
    d = size(Xval_s, 1)

    predictions_val = Matrix{Vector{Float64}}(undef, n_forecasters, n_val)
    predictions_test = Matrix{Vector{Float64}}(undef, n_forecasters, n_test)

    @info "Generating expert predictions (multivariate)" n_model_forecasters n_forecasters n_val n_test
    for (i, m) in enumerate(experts)
        model = build_model(m.model_type, m.config)
        yhat_val = predict_unscaled(model, m.parameters, m.states, Xval_s)
        yhat_te = predict_unscaled(model, m.parameters, m.states, Xte_s)
        for j = 1:n_val
            predictions_val[i, j] = Float64.(yhat_val[:, j])
        end
        for j = 1:n_test
            predictions_test[i, j] = Float64.(yhat_te[:, j])
        end
        @info "Expert ready" index = i model_type = m.model_type
    end

    if n_quantile_forecasters > 0
        x_last_val_scaled = Float64.(Xval_s[:, end, :])
        for (offset, q_pct) in enumerate(selected_quantiles)
            idx = n_model_forecasters + offset
            q_vec = [
                quantile(Float64.(view(x_last_val_scaled, k, :)), q_pct / 100.0) for k = 1:d
            ]
            q = vec(Float64.(reshape(q_vec, :, 1)[:, 1]))
            for j = 1:n_val
                predictions_val[idx, j] = copy(q)
            end
            for j = 1:n_test
                predictions_test[idx, j] = copy(q)
            end
        end
        @info "Added constant quantile experts (multivariate)" quantiles =
            selected_quantiles
    else
        @info "No quantile experts selected (multivariate)"
    end

    return predictions_val, predictions_test
end

# ---------------------------------------------------------------------------
# Three-split expert prediction generation (for neural ensemble pipeline)
# Returns 3D arrays: (n_model_forecasters, d, n_samples) per split
# ---------------------------------------------------------------------------

function generate_expert_predictions_three_splits(experts, Xtr_s, Xval_s, Xte_s)
    n_model_forecasters = length(experts)
    d = size(Xtr_s, 1)
    n_train = size(Xtr_s, 3)
    n_val = size(Xval_s, 3)
    n_test = size(Xte_s, 3)

    predictions_train = Array{Float64}(undef, n_model_forecasters, d, n_train)
    predictions_val = Array{Float64}(undef, n_model_forecasters, d, n_val)
    predictions_test = Array{Float64}(undef, n_model_forecasters, d, n_test)

    @info "Generating expert predictions (three splits)" n_model_forecasters n_train n_val n_test output_dim = d
    for (i, m) in enumerate(experts)
        model = build_model(m.model_type, m.config)
        yhat_tr = predict_unscaled(model, m.parameters, m.states, Xtr_s)
        yhat_val = predict_unscaled(model, m.parameters, m.states, Xval_s)
        yhat_te = predict_unscaled(model, m.parameters, m.states, Xte_s)

        predictions_train[i, :, :] = Float64.(yhat_tr)
        predictions_val[i, :, :] = Float64.(yhat_val)
        predictions_test[i, :, :] = Float64.(yhat_te)
        @info "Expert ready" index = i model_type = m.model_type
    end

    return predictions_train, predictions_val, predictions_test
end
