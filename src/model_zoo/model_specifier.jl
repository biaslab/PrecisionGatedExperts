using YAML

using Distributions
using Statistics
using JLD2
using Lux
using ProgressMeter
using Reactant

export run_experiment, predict_from_trained_ensemble

struct ExperimentSpecifier{P,M,D,F}
    prediction_type::P
    model_type::M
    column::Union{String,Nothing}
    horizon::Int
    dataset::D            # Val{:ETTh1}, Val{:electricity}, etc. — dispatches load_dataset
    dataset_path::String  # absolute path to the CSV
    experts::Vector{String}
    selected_quantiles::Vector{Float64}
    number_of_quantiles::Union{Int,Nothing}
    priors::Dict{Symbol,Any}
    inference_iterations::Int
    prediction_iterations::Int
    save_predictions::Bool
    subsample_size::Union{Int,Nothing}
    subsample_percentage::Union{Float64,Nothing}
    repeat_batch::Union{Int,Nothing}
    feature_type::F
end

# ---------------------------------------------------------------------------
# YAML → ExperimentSpecifier
# ---------------------------------------------------------------------------

function _normalize_selected_quantiles(raw_quantiles)
    quantiles_vec = Float64.(raw_quantiles)
    isempty(quantiles_vec) && return Float64[]

    all_in_unit_interval = all(0.0 <= q <= 1.0 for q in quantiles_vec)
    all_in_percent_range = all(0.0 <= q <= 100.0 for q in quantiles_vec)

    if all_in_unit_interval
        quantiles_vec .*= 100.0
    elseif !all_in_percent_range
        error("Quantiles must be in [0, 1] or [0, 100]. Got: $(quantiles_vec)")
    end

    all(0.0 < q < 100.0 for q in quantiles_vec) ||
        error("Quantiles must be strictly inside (0, 100). Got: $(quantiles_vec)")

    return unique(sort(quantiles_vec))
end

function _uniform_selected_quantiles(number_of_quantiles::Int)
    number_of_quantiles >= 0 || error("number_of_quantiles must be >= 0")
    number_of_quantiles == 0 && return Float64[]

    quantiles_unit = range(0.0, 1.0; length = number_of_quantiles + 2)[2:(end-1)]
    return 100.0 .* collect(quantiles_unit)
end

function _resolve_selected_quantiles(params::Dict)
    has_explicit_quantiles =
        haskey(params, "quantiles") || haskey(params, "selected_quantiles")
    number_of_quantiles = get(params, "number_of_quantiles", nothing)

    if has_explicit_quantiles
        raw =
            haskey(params, "quantiles") ? params["quantiles"] : params["selected_quantiles"]
        return _normalize_selected_quantiles(raw), nothing
    elseif !isnothing(number_of_quantiles)
        return _uniform_selected_quantiles(Int(number_of_quantiles)),
        Int(number_of_quantiles)
    else
        return [10.0, 90.0], 2
    end
end

function _parse_spec(config)
    p = config["params"]
    prediction_type = _parse_prediction_type(p["prediction_type"])
    model_type = _parse_model_type(p["model_type"])
    column = get(p, "column", nothing)
    horizon = p["horizon"]
    dataset = Val(Symbol(p["dataset"]))
    dataset_path = p["dataset_path"]
    experts = collect(String, p["experts"])
    selected_quantiles, number_of_quantiles = _resolve_selected_quantiles(p)
    n_forecasters = length(experts) + length(selected_quantiles)
    n_forecasters > 0 || error(
        "At least one forecaster is required. Provide experts, quantiles, or number_of_quantiles > 0.",
    )
    if haskey(p, "priors") &&
       haskey(p["priors"], "w") &&
       get(p["priors"]["w"], "break_symmetry_prior", false)
        @info "Break-symmetry prior requested in config" n_forecasters strength =
            get(p["priors"]["w"], "break_symmetry_strength", 0.01)
    end
    priors = parse_priors(model_type, p["priors"], n_forecasters)
    inference_iterations = p["inference_iterations"]
    prediction_iterations = p["prediction_iterations"]
    save_predictions = get(p, "save_predictions", false)
    subsample_size = get(p, "subsample_size", nothing)
    subsample_percentage = get(p, "subsample_percentage", nothing)
    repeat_batch = get(p, "repeat_batch", nothing)
    feature_type = parse_feature_type(get(p, "feature_type", "simple"))
    return ExperimentSpecifier(
        prediction_type,
        model_type,
        column,
        horizon,
        dataset,
        dataset_path,
        experts,
        selected_quantiles,
        number_of_quantiles,
        priors,
        inference_iterations,
        prediction_iterations,
        save_predictions,
        subsample_size,
        subsample_percentage,
        repeat_batch,
        feature_type,
    )
end

function run_experiment(path_to_yaml::String)
    config = YAML.load_file(path_to_yaml)
    spec = _parse_spec(config)
    results = run_experiment(spec)
    results_with_raw_spec = merge(results, (raw_spec = config,))
    results_dir = "final_results"
    mkpath(results_dir)
    ds_name = typeof(spec.dataset).parameters[1]
    model_name = model_type_name(spec.model_type)

    if spec.prediction_type isa Univariate
        fname = "$(ds_name)_h$(spec.horizon)_$(spec.column)_$(model_name)_$(hash(config)).jld2"
    else
        fname = "$(ds_name)_h$(spec.horizon)_multivariate_$(model_name)_$(hash(config)).jld2"
    end
    results_path = joinpath(results_dir, fname)
    save_data = if spec.save_predictions
        @info "predictions are saved"
        pairs(results_with_raw_spec)
    else
        @info "predictions are not saved"
        skipped_fields = [:ensemble_std, :ensemble_mean, :y_test, :predictions_test]
        (k => v for (k, v) in pairs(results_with_raw_spec) if !(k in skipped_fields))
    end
    JLD2.jldsave(results_path; save_data...)
    @info "Results saved" path=results_path save_predictions=spec.save_predictions

    return results_with_raw_spec
end

# ---------------------------------------------------------------------------
# Spec serialization helper (used by all pipeline functions)
# ---------------------------------------------------------------------------

function _save_spec(spec::ExperimentSpecifier; column = spec.column)
    return (
        prediction_type = model_prediction_name(spec.prediction_type),
        model_type = model_type_name(spec.model_type),
        column = column,
        horizon = spec.horizon,
        dataset = typeof(spec.dataset).parameters[1],
        dataset_path = spec.dataset_path,
        experts = spec.experts,
        selected_quantiles = spec.selected_quantiles,
        number_of_quantiles = spec.number_of_quantiles,
    )
end

# ---------------------------------------------------------------------------
# Generic pipeline hooks — defaults (models override in their pipeline.jl)
# ---------------------------------------------------------------------------

# build_rxinfer_model(pt, mt, n_forecasters, n_obs, priors) — required, no default

build_rxinfer_constraints(::Any, ::Any, _, _) = nothing
build_rxinfer_init(::Any, ::Any, _) = nothing
prepare_priors!(::Any, ::Any, _, _) = nothing
build_training_data(::Any, ::Any, y, features, predictions) =
    (y = y, features = features, predictions = predictions)
training_posterior_keys(::ModelType) = (:γ,)
prediction_prior_keys(mt::ModelType) =
    Tuple(k for k in training_posterior_keys(mt) if k !== :γ)
build_returnvars(::ModelType) = nothing

# model_results(pt, mt, training_posteriors, test_posteriors) — required, no default

function _compute_n_obs(spec, n_val)
    if !isnothing(spec.subsample_percentage)
        return round(Int, n_val * spec.subsample_percentage)
    else
        return something(spec.subsample_size, n_val)
    end
end

function _build_infer_kwargs(pt, mt, priors, prediction::Bool)
    kwargs = Dict{Symbol,Any}(:showprogress => true)
    kwargs[:constraints] = build_rxinfer_constraints(pt, mt, priors, prediction)
    kwargs[:initialization] = build_rxinfer_init(pt, mt, priors)
    kwargs[:returnvars] = build_returnvars(mt)
    return kwargs
end

function _expert_mse(::Univariate, predictions_val, y_val, i)
    return mse(predictions_val[i, :], y_val)
end

function _expert_mse(::Multivariate, predictions_val, y_val, i)
    pred_mat_i = reduce(hcat, predictions_val[i, :])
    Yval_mat = reduce(hcat, y_val)
    return mse_mv(pred_mat_i, Yval_mat)
end

function _log_validation_metrics(pt, mt, posteriors, predictions_val, y_val)
    n_forecasters = size(predictions_val, 1)
    γ = posteriors[:γ]
    γ_means = mean.(γ)
    is_matrix = ndims(γ_means) == 2

    model_name = model_type_name(mt)
    @info "Learned $(model_name) weights on validation"
    for i = 1:n_forecasters
        mse_i = _expert_mse(pt, predictions_val, y_val, i)
        if is_matrix
            avg_γ = mean(γ_means[i, :])
            @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
                round(mse_i; digits = 6)
        else
            @info "Expert $i" E_γ = round(γ_means[i]; digits = 4) val_MSE =
                round(mse_i; digits = 6)
        end
    end
end

function _save_spec_column(::Univariate, spec)
    return spec.column
end

function _save_spec_column(::Multivariate, _)
    return nothing
end

const TEST_PREDICTION_BATCH_SIZE = 250

_batch_slice(x::AbstractVector, batch_range) = x[batch_range]
_batch_slice(x::AbstractMatrix, batch_range) = x[:, batch_range]

function _merge_batched_posterior(existing, chunk)
    if existing isa AbstractVector && chunk isa AbstractVector
        return vcat(existing, chunk)
    elseif existing isa AbstractMatrix && chunk isa AbstractMatrix
        return hcat(existing, chunk)
    else
        return chunk
    end
end

function _predict_test_in_batches(
    pt,
    mt,
    prediction_priors;
    n_forecasters,
    n_test,
    predictions_test,
    features_test,
    prediction_iterations,
    batch_size = TEST_PREDICTION_BATCH_SIZE,
)
    ensemble_preds = Any[]
    test_posteriors = Dict{Symbol,Any}()
    n_batches = cld(n_test, batch_size)
    progress = Progress(
        n_batches;
        desc = "Test prediction batches",
        dt = 0.5,
    )

    for batch_start = 1:batch_size:n_test
        batch_end = min(batch_start + batch_size - 1, n_test)
        batch_range = batch_start:batch_end
        n_batch = length(batch_range)

        infer_batch = predict_with_model(
            pt,
            mt,
            prediction_priors;
            n_forecasters = n_forecasters,
            n_steps = n_batch,
            prediction_array = [missing for _ = 1:n_batch],
            predictions_test = _batch_slice(predictions_test, batch_range),
            features_test = _batch_slice(features_test, batch_range),
            prediction_iterations = prediction_iterations,
        )

        append!(ensemble_preds, infer_batch.predictions[:y][end])

        for k in training_posterior_keys(mt)
            if haskey(infer_batch.posteriors, k)
                batch_posterior = infer_batch.posteriors[k][end]
                if haskey(test_posteriors, k)
                    test_posteriors[k] =
                        _merge_batched_posterior(test_posteriors[k], batch_posterior)
                else
                    test_posteriors[k] = batch_posterior
                end
            end
        end

        ProgressMeter.next!(
            progress;
            showvalues = [(:batch_start, batch_start), (:batch_end, batch_end)],
        )
    end

    return ensemble_preds, test_posteriors
end

# ---------------------------------------------------------------------------
# Generic run_experiment — dispatches to model hooks
# ---------------------------------------------------------------------------

function run_experiment(spec::ExperimentSpecifier)
    pt, mt = spec.prediction_type, spec.model_type
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)
    n_obs = _compute_n_obs(spec, n_val)
    @info n_obs

    prepare_priors!(pt, mt, spec.priors, predictions_val)

    # Build model + train
    model = build_rxinfer_model(pt, mt, n_forecasters, n_obs, spec.priors)
    data = build_training_data(pt, mt, y_val, features_val, predictions_val)
    train_kwargs = _build_infer_kwargs(pt, mt, spec.priors, false)

    @info "Fitting $(model_type_name(mt)) ensemble on validation data" n_forecasters n_val
    result = run_training_rxinfer(spec, model, data; train_kwargs...)

    # Extract posteriors
    posteriors = Dict{Symbol,Any}(
        k => result.posteriors[k][end] for k in training_posterior_keys(mt)
    )
    free_energy = result.free_energy

    _log_validation_metrics(pt, mt, posteriors, predictions_val, y_val)

    # Prediction on test
    prediction_priors =
        Dict{Symbol,Any}(k => deepcopy(posteriors[k]) for k in prediction_prior_keys(mt))

    @info "Generating $(model_type_name(mt)) ensemble predictions on test" batch_size = TEST_PREDICTION_BATCH_SIZE
    ensemble_preds, test_posteriors = _predict_test_in_batches(
        pt,
        mt,
        prediction_priors;
        n_forecasters = n_forecasters,
        n_test = n_test,
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = spec.prediction_iterations,
    )

    # Metrics
    y_metrics = prepare_y_for_metrics(pt, y_test)
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(pt, ensemble_preds, y_metrics)
    @info "Ensemble test metrics" ensemble_metrics...

    # Build results: model-specific fields + common fields
    model_fields = model_results(pt, mt, posteriors, test_posteriors)

    return merge(
        model_fields,
        (
            free_energy = free_energy,
            ensemble_mean = ensemble_mean,
            ensemble_std = ensemble_std,
            ensemble_metrics = ensemble_metrics,
            predictions_test = predictions_test,
            y_test = y_metrics,
            spec = _save_spec(spec; column = _save_spec_column(pt, spec)),
        ),
    )
end

# ---------------------------------------------------------------------------
# Generic predict_with_model — dispatches to model hooks
# ---------------------------------------------------------------------------

function predict_with_model(
    pt,
    mt,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    model = build_rxinfer_model(pt, mt, n_forecasters, n_steps, priors)
    data = build_training_data(pt, mt, prediction_array, features_test, predictions_test)
    kwargs = _build_infer_kwargs(pt, mt, priors, true)
    return infer(;
        model = model,
        data = data,
        iterations = prediction_iterations,
        free_energy = false,
        kwargs...,
    )
end

# ---------------------------------------------------------------------------
# Load saved results → re-predict
# ---------------------------------------------------------------------------

function _dataset_val(x)
    return x isa Symbol ? Val(x) : Val(Symbol(x))
end

function _has_field(x, name::Symbol)
    return name in fieldnames(typeof(x))
end

function _saved_quantile_config(spec_saved)
    selected_quantiles = if _has_field(spec_saved, :selected_quantiles)
        Float64.(spec_saved.selected_quantiles)
    elseif _has_field(spec_saved, :quantiles)
        _normalize_selected_quantiles(spec_saved.quantiles)
    else
        [10.0, 90.0]
    end

    number_of_quantiles =
        if _has_field(spec_saved, :number_of_quantiles) &&
           !isnothing(spec_saved.number_of_quantiles)
            Int(spec_saved.number_of_quantiles)
        else
            nothing
        end
    return selected_quantiles, number_of_quantiles
end

function _spec_for_prediction_from_saved(saved, prediction_iterations::Int)
    haskey(saved, "spec") || error("Missing `spec` in saved results")
    spec_saved = saved["spec"]

    function _from_saved_fields()
        prediction_type = _parse_prediction_type(string(spec_saved.prediction_type))
        model_type = _parse_model_type(string(spec_saved.model_type))
        column = isnothing(spec_saved.column) ? nothing : String(spec_saved.column)
        dataset = _dataset_val(spec_saved.dataset)
        dataset_path = String(spec_saved.dataset_path)
        experts = collect(String, spec_saved.experts)
        selected_quantiles, number_of_quantiles = _saved_quantile_config(spec_saved)

        return ExperimentSpecifier(
            prediction_type,
            model_type,
            column,
            Int(spec_saved.horizon),
            dataset,
            dataset_path,
            experts,
            selected_quantiles,
            number_of_quantiles,
            Dict{Symbol,Any}(),
            1,
            prediction_iterations,
            false,
            nothing,
            nothing,
            nothing,
            SimpleFeatures(),
        )
    end

    if haskey(saved, "raw_spec")
        try
            spec_from_raw = _parse_spec(saved["raw_spec"])
            return ExperimentSpecifier(
                spec_from_raw.prediction_type,
                spec_from_raw.model_type,
                spec_from_raw.column,
                spec_from_raw.horizon,
                spec_from_raw.dataset,
                spec_from_raw.dataset_path,
                spec_from_raw.experts,
                spec_from_raw.selected_quantiles,
                spec_from_raw.number_of_quantiles,
                Dict{Symbol,Any}(),
                1,
                prediction_iterations,
                false,
                nothing,
                nothing,
                nothing,
                spec_from_raw.feature_type,
            )
        catch err
            @warn "Failed to parse raw_spec, falling back to saved spec fields" error =
                sprint(showerror, err)
            return _from_saved_fields()
        end
    end

    return _from_saved_fields()
end

prepare_y_test(::Univariate, y_test_all, n_steps) = Float64.(y_test_all)

function prepare_y_test(::Multivariate, y_test_all, n_steps)
    return y_test_all isa AbstractMatrix ? [Float64.(y_test_all[:, j]) for j = 1:n_steps] :
           y_test_all
end

function prepare_y_for_metrics(::Univariate, y_test)
    return y_test
end

function prepare_y_for_metrics(::Multivariate, y_test)
    return reduce(hcat, y_test)
end

"""
    predict_from_trained_ensemble(path_to_jld2; prediction_iterations=20, alpha=1.0)

Load a trained RxInfer ensemble from a saved results `.jld2` and run prediction on the
full available test set using the same dataset + expert checkpoints recorded in the file.
"""
function predict_from_trained_ensemble(
    path_to_jld2::AbstractString,
    prediction_iterations::Int = 20,
    alpha::Float64 = 1.0,
)
    isfile(path_to_jld2) || error("Results file not found: $path_to_jld2")
    prediction_iterations > 0 || error("prediction_iterations must be > 0")

    saved = JLD2.load(path_to_jld2)
    spec_for_data = _spec_for_prediction_from_saved(saved, prediction_iterations)

    _, y_test_all, _, predictions_test_all, _, features_test_all =
        before_rxinfer(spec_for_data)
    n_steps = length(y_test_all)

    prediction_type = spec_for_data.prediction_type
    model_type = spec_for_data.model_type

    y_test = prepare_y_test(prediction_type, y_test_all, n_steps)
    predictions_test = predictions_test_all
    features_test = features_test_all

    n_forecasters = size(predictions_test, 1)
    prediction_array = [missing for _ = 1:n_steps]

    priors = extract_prediction_priors(model_type, saved)
    prepare_priors!(prediction_type, model_type, priors, predictions_test)
    @info "Prediction start"
    infer_test = predict_with_model(
        prediction_type,
        model_type,
        priors;
        n_forecasters = n_forecasters,
        n_steps = n_steps,
        prediction_array = prediction_array,
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = prediction_iterations,
    )

    ensemble_preds = infer_test.predictions[:y][end]

    Y_for_metrics = prepare_y_for_metrics(prediction_type, y_test)

    ensemble_mean, ensemble_std, ensemble_metrics = compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics,
    )

    return (
        n_prediction_steps = n_steps,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        infer_result = infer_test,
    )
end

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

function _before_rxinfer_features(_, feature_type::UniWindowFeatures, X, col_idx)
    return make_features(feature_type, X, col_idx)
end

function _before_rxinfer_features(spec, feature_type, X, _)
    return make_features(feature_type, X, spec.dataset)
end

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

function before_rxinfer(spec::ExperimentSpecifier{Univariate})
    @info "Loading expert models" n = length(spec.experts)
    experts = map(load_jld2_model, spec.experts)

    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)
    col_idx = find_column_index(feat_cols, spec.column)

    if isempty(experts)
        seq_len = 96
        X3, Y2 = make_sequences(Xmat; seq_len = seq_len, horizon = spec.horizon)
        Xtr, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2)
        scaler = fit_scaler(Xtr)
        @info "No model experts; using dataset-only setup" seq_len horizon = spec.horizon
    else
        base_meta = experts[1].meta
        seq_len = Int(base_meta.seq_len)
        horizon = Int(base_meta.horizon)
        X3, Y2 = make_sequences(Xmat; seq_len = seq_len, horizon = horizon)
        split = base_meta.split
        _, _, Xval, Yval, Xte, Yte =
            train_val_test_split(X3, Y2; ratios = (split.train, split.val, split.test))
        scaler = base_meta.scaler
    end

    Xval_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Yval_s = scale_targets(scaler, Yval)
    Yte_s = scale_targets(scaler, Yte)

    y_val = Float64.(Yval_s[col_idx, :])
    y_test = Float64.(Yte_s[col_idx, :])

    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type,
        experts,
        scaler,
        Xval_s,
        Xte_s,
        col_idx,
        spec.selected_quantiles,
    )

    features_val = _before_rxinfer_features(spec, spec.feature_type, Xval_s, col_idx)
    features_test = _before_rxinfer_features(spec, spec.feature_type, Xte_s, col_idx)

    return (y_val, y_test, predictions_val, predictions_test, features_val, features_test)
end

function before_rxinfer(spec::ExperimentSpecifier{Multivariate})
    @info "Loading expert models" n = length(spec.experts)
    experts = map(load_jld2_model, spec.experts)

    Xmat, feat_cols = load_dataset(spec.dataset, spec.dataset_path)

    d = length(feat_cols)
    if isempty(experts)
        seq_len = 96
        X3, Y2 = make_sequences(Xmat; seq_len = seq_len, horizon = spec.horizon)
        Xtr, _, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2)
        scaler = fit_scaler(Xtr)
        @info "No model experts; using dataset-only setup (multivariate)" seq_len horizon =
            spec.horizon
    else
        base_meta = experts[1].meta
        seq_len = Int(base_meta.seq_len)
        horizon = Int(base_meta.horizon)
        X3, Y2 = make_sequences(Xmat; seq_len = seq_len, horizon = horizon)
        split = base_meta.split
        _, _, Xval, Yval, Xte, Yte =
            train_val_test_split(X3, Y2; ratios = (split.train, split.val, split.test))
        scaler = base_meta.scaler
        n_scaler = length(scaler.μ)
        n_feat = size(Xval, 1)
        n_scaler == n_feat || error(
            "Scaler dimension ($n_scaler) from expert model does not match " *
            "dataset features ($n_feat). The expert was likely trained on a " *
            "different dataset. Check that `dataset_path` in the YAML matches " *
            "the experts' training data.",
        )
    end

    Xval_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Yval_s = scale_targets(scaler, Yval)
    Yte_s = scale_targets(scaler, Yte)

    n_val = size(Xval, 3)
    n_test = size(Xte, 3)

    y_val = [Float64.(Yval_s[:, j]) for j = 1:n_val]
    y_test = [Float64.(Yte_s[:, j]) for j = 1:n_test]

    predictions_val, predictions_test = generate_expert_predictions(
        spec.prediction_type,
        experts,
        scaler,
        Xval_s,
        Xte_s,
        spec.selected_quantiles,
    )

    features_val = _before_rxinfer_features(spec, spec.feature_type, Xval_s, nothing)
    features_test = _before_rxinfer_features(spec, spec.feature_type, Xte_s, nothing)
    return (y_val, y_test, predictions_val, predictions_test, features_val, features_test)
end

# ---------------------------------------------------------------------------
# Training dispatcher
# ---------------------------------------------------------------------------

function run_training_rxinfer(spec, subsampled_data::Int, model, data; kwargs...)
    @show "Use subsampled data with sample size $(subsampled_data)"
    @info "Repeat each batch $(spec.repeat_batch) times"
    subsampled = (;
        (
            k => SubsampledData(v, subsampled_data, spec.repeat_batch) for
            (k, v) in pairs(data)
        )...
    )
    return infer(;
        model = model,
        data = subsampled,
        iterations = spec.inference_iterations,
        free_energy = true,
        kwargs...,
    )
end

function run_training_rxinfer(spec, subsample_percentage::Float64, model, data; kwargs...)
    @show "Use subsampled data with sample percentage $(subsample_percentage)"
    subsampled_data = round(Int, length(data.y)*subsample_percentage)
    @info "Equivalent to size of batch" subsampled_data
    return run_training_rxinfer(spec, subsampled_data, model, data; kwargs...)
end

function run_training_rxinfer(spec, ::Nothing, model, data; kwargs...)
    @show "Run inference on the full dataset"
    return infer(;
        model = model,
        data = data,
        iterations = spec.inference_iterations,
        free_energy = true,
        kwargs...,
    )
end

function run_training_rxinfer(spec, model, data; kwargs...)
    if !isnothing(spec.subsample_size)
        return run_training_rxinfer(spec, spec.subsample_size, model, data; kwargs...)
    elseif !isnothing(spec.subsample_percentage)
        return run_training_rxinfer(spec, spec.subsample_percentage, model, data; kwargs...)
    else
        return run_training_rxinfer(spec, nothing, model, data; kwargs...)
    end
end
