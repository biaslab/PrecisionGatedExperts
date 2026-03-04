using YAML

using Distributions
using Statistics
using JLD2
using Lux
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

function parse_model_type(s::String)
    s == "static" && return Static()
    s == "dynamic" && return Dynamic()
    s == "hierarchical" && return Hierarchical()
    s == "dynamic_noisy_observations" && return DynamicNoisyObservations()
    s == "deep" && return Deep()
    error("Unknown model_type: $s")
end

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
    prediction_type = parse_prediction_type(p["prediction_type"])
    model_type = parse_model_type(p["model_type"])
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
    model_name = lowercase(string(typeof(spec.model_type)))

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

function _parse_saved_prediction_type(s::AbstractString)
    occursin("Univariate", s) && return Univariate()
    occursin("Multivariate", s) && return Multivariate()
    error("Unknown saved prediction_type: $s")
end

function _parse_saved_model_type(s::AbstractString)
    occursin("Static", s) && return Static()
    occursin("Dynamic", s) && return Dynamic()
    occursin("Hierarchical", s) && return Hierarchical()
    occursin("DynamicNoisyObservations", s) && return DynamicNoisyObservations()
    occursin("Deep", s) && return Deep()
    error("Unknown saved model_type: $s")
end

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
        prediction_type = _parse_saved_prediction_type(string(spec_saved.prediction_type))
        model_type = _parse_saved_model_type(string(spec_saved.model_type))
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

# ---------------------------------------------------------------------------
# Prediction dispatch: extract_prediction_priors + predict_with_model
# ---------------------------------------------------------------------------

prepare_y_test(::Univariate, y_test_all, n_steps) = Float64.(y_test_all)

function prepare_y_test(::Multivariate, y_test_all, n_steps)
    return y_test_all isa AbstractMatrix ? [Float64.(y_test_all[:, j]) for j = 1:n_steps] :
           y_test_all
end

function extract_prediction_priors(::Static, saved, alpha)
    return Dict{Symbol,Any}(:γ => saved["γ_posteriors"])
end

function extract_prediction_priors(::Dynamic, saved, alpha)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :τ => saved["τ_posteriors"],
        :β => saved["β_posteriors"],
    )
end

function extract_prediction_priors(::Hierarchical, saved, alpha)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :τ => saved["τ_posteriors"],
        :ρ => saved["ρ_posteriors"],
        :α => alpha,
    )
end

function extract_prediction_priors(::DynamicNoisyObservations, saved, alpha)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :τ => saved["τ_posteriors"],
        :β => saved["β_posteriors"],
        :κ => saved["κ_posterior"],
    )
end

function extract_prediction_priors(::Deep, saved, alpha)
    return Dict{Symbol,Any}(
        :w => saved["w_posteriors"],
        :v => saved["v_posteriors"],
        :τ => saved["τ_posteriors"],
        :ρ => saved["ρ_posteriors"],
        :α => alpha,
    )
end

# --- Static ---

function predict_with_model(
    ::Univariate,
    ::Static,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = univariate_ensemble_precision_model(
            n_forecasters = n_forecasters,
            priors = priors,
        ),
        data = (y = prediction_array, X = predictions_test),
        iterations = prediction_iterations,
    )
end

function predict_with_model(
    ::Multivariate,
    ::Static,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = multivariate_ensemble_precision_model(
            n_forecasters = n_forecasters,
            priors = priors,
        ),
        data = (y = prediction_array, X = predictions_test),
        iterations = prediction_iterations,
    )
end

# --- Dynamic ---

function predict_with_model(
    ::Univariate,
    ::Dynamic,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = univariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs = n_steps,
            priors = priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = univariate_dynamic_ensemble_constraints(priors, true),
        initialization = univariate_dynamic_ensemble_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
end

function predict_with_model(
    ::Multivariate,
    ::Dynamic,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = multivariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs = n_steps,
            priors = priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = multivariate_dynamic_ensemble_constraints(priors, true),
        initialization = multivariate_dynamic_ensemble_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
end

# --- DynamicNoisyObservations ---

function predict_with_model(
    ::Univariate,
    ::DynamicNoisyObservations,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = univariate_dynamic_noisy_ensemble(
            n_forecasters = n_forecasters,
            n_obs = n_steps,
            priors = priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = univariate_dynamic_noisy_ensemble_constraints(priors, true),
        initialization = univariate_dynamic_noisy_ensemble_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
end

# --- Hierarchical ---

function predict_with_model(
    ::Univariate,
    ::Hierarchical,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs = n_steps,
            priors = priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = hierarchical_constraints(priors, true),
        initialization = hierarchical_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
end

function predict_with_model(
    ::Multivariate,
    ::Hierarchical,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = multivariate_hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs = n_steps,
            priors = priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = multivariate_hierarchical_constraints(priors, true),
        initialization = multivariate_hierarchical_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
end

# --- Deep ---

function predict_with_model(
    ::Univariate,
    ::Deep,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = deep_model(n_forecasters = n_forecasters, n_obs = n_steps, priors = priors),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = deep_constraints(),
        initialization = deep_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
end

function predict_with_model(
    ::Multivariate,
    ::Deep,
    priors;
    n_forecasters,
    n_steps,
    prediction_array,
    predictions_test,
    features_test,
    prediction_iterations,
)
    return infer(
        model = multivariate_deep_model(
            n_forecasters = n_forecasters,
            n_obs = n_steps,
            priors = priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = multivariate_deep_constraints(),
        initialization = multivariate_deep_init(priors),
        iterations = prediction_iterations,
        free_energy = false,
        showprogress = true,
    )
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

    y_test = prepare_y_test(prediction_type, y_test_all, n_steps)
    predictions_test = predictions_test_all
    features_test = features_test_all

    n_forecasters = size(predictions_test, 1)
    prediction_array = [missing for _ = 1:n_steps]

    priors = extract_prediction_priors(model_type, saved, alpha)
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

function run_experiment(spec::ExperimentSpecifier{Univariate,DynamicNoisyObservations})
    return run_dynamic_noisy_observations_univariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Univariate,Hierarchical})
    return run_hierarchical_univariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Multivariate,Hierarchical})
    return run_hierarchical_multivariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Univariate,Deep})
    return run_deep_univariate(spec)
end

function run_experiment(spec::ExperimentSpecifier{Multivariate,Deep})
    return run_deep_multivariate(spec)
end

function _before_rxinfer_features(_, feature_type::UniWindowFeatures, X, col_idx)
    return make_features(feature_type, X, col_idx)
end

function _before_rxinfer_features(spec, feature_type, X, _)
    return make_features(feature_type, X, spec.dataset)
end

function _before_rxinfer_features(
    ::ExperimentSpecifier{T,Static},
    feature_type,
    X,
    _,
) where {T}
    return nothing
end

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

# ---------------------------------------------------------------------------
# Static Univariate pipeline
# ---------------------------------------------------------------------------

function run_static_univariate(spec::ExperimentSpecifier{Univariate,Static})
    y_val, y_test, predictions_val, predictions_test, _, _ = before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)

    model = univariate_ensemble_precision_model(
        n_forecasters = n_forecasters,
        priors = spec.priors,
    )
    data = (y = y_val, X = predictions_val)

    @info "Fitting static ensemble on validation data"
    result = run_training_rxinfer(spec, model, data)

    free_energy = result.free_energy
    γ_posteriors = result.posteriors[:γ][end]
    γ_means = map(mean, γ_posteriors)
    weights = γ_means ./ sum(γ_means)

    @info "Learned precision weights"
    for i = 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        @info "Expert $i" E_γ = round(γ_means[i]; digits = 4) val_MSE =
            round(mse_i; digits = 6) weight = round(weights[i]; digits = 4)
    end

    @info "Generating ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(:γ => γ_posteriors)
    prediction_array = [missing for _ = 1:length(y_test)]
    infer_test = infer(
        model = univariate_ensemble_precision_model(
            n_forecasters = n_forecasters,
            priors = posterior_priors,
        ),
        data = (y = prediction_array, X = predictions_test),
        iterations = spec.prediction_iterations,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, y_test)

    results = (
        γ_posteriors = γ_posteriors,
        weights = weights,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = y_test,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = spec.column,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

# ---------------------------------------------------------------------------
# Static Multivariate pipeline
# ---------------------------------------------------------------------------
function run_static_multivariate(spec::ExperimentSpecifier{Multivariate,Static})
    y_val, y_test, predictions_val, predictions_test, _, _ = before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_test = length(y_test)
    d = length(y_val[1])

    model = multivariate_ensemble_precision_model(
        n_forecasters = n_forecasters,
        priors = spec.priors,
    )
    data = (y = y_val, X = predictions_val)

    @info "Fitting static multivariate ensemble on validation data" d n_forecasters
    result = run_training_rxinfer(spec, model, data; showprogress = true)

    free_energy = result.free_energy
    γ_posteriors = result.posteriors[:γ][end]
    γ_means = map(mean, γ_posteriors)
    weights = γ_means ./ sum(γ_means)

    # Per-expert validation MSE (multivariate)
    Yval_mat = reduce(hcat, y_val)
    @info "Learned precision weights"
    for i = 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])  # (d, n_val)
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        @info "Expert $i" E_γ = round(γ_means[i]; digits = 4) val_MSE =
            round(mse_i; digits = 6) weight = round(weights[i]; digits = 4)
    end

    # 5. Ensemble predictions on test
    @info "Generating ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(:γ => γ_posteriors)
    prediction_array = [missing for _ = 1:n_test]
    infer_test = infer(
        model = multivariate_ensemble_precision_model(
            n_forecasters = n_forecasters,
            priors = posterior_priors,
        ),
        data = (y = prediction_array, X = predictions_test),
        iterations = spec.prediction_iterations,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    Yte_mat = reduce(hcat, y_test)
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, Yte_mat)

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # 7. Save results
    results = (
        γ_posteriors = γ_posteriors,
        weights = weights,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = Yte_mat,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = nothing,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

function run_dynamic_univariate(spec::ExperimentSpecifier{Univariate,Dynamic})
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = univariate_dynamic_ensemble(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = spec.priors,
    )
    constraints = univariate_dynamic_ensemble_constraints(spec.priors, false)
    init = univariate_dynamic_ensemble_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting dynamic ensemble on validation data" n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    τ_posteriors = result.posteriors[:τ][end]
    β_posteriors = result.posteriors[:β][end]
    γ_posteriors = result.posteriors[:γ][end]  # (n_forecasters, n_val)

    γ_means_val = mean.(γ_posteriors)

    @info "Learned dynamic weights on validation"
    for i = 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end

    # 6. Ensemble predictions on test
    @info "Generating dynamic ensemble predictions on test"
    # deepcopy posteriors to prevent in-place mutation by LowRank product rules
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :τ => deepcopy(τ_posteriors),
        :β => deepcopy(β_posteriors),
    )
    prediction_array = [missing for _ = 1:n_test]

    infer_test = infer(
        model = univariate_dynamic_ensemble(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            priors = posterior_priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = univariate_dynamic_ensemble_constraints(posterior_priors, true),
        initialization = univariate_dynamic_ensemble_init(posterior_priors),
        iterations = spec.prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, y_test)

    # Extract dynamic precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors = w_posteriors,
        τ_posteriors = τ_posteriors,
        β_posteriors = β_posteriors,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = y_test,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = spec.column,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

# ---------------------------------------------------------------------------
# DynamicNoisyObservations Univariate pipeline
# ---------------------------------------------------------------------------

function run_dynamic_noisy_observations_univariate(
    spec::ExperimentSpecifier{Univariate,DynamicNoisyObservations},
)
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = univariate_dynamic_noisy_ensemble(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = spec.priors,
    )
    constraints = univariate_dynamic_noisy_ensemble_constraints(spec.priors, false)
    init = univariate_dynamic_noisy_ensemble_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting dynamic noisy observations ensemble on validation data" n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    τ_posteriors = result.posteriors[:τ][end]
    β_posteriors = result.posteriors[:β][end]
    κ_posterior = result.posteriors[:κ][end]
    γ_posteriors = result.posteriors[:γ][end]

    γ_means_val = mean.(γ_posteriors)

    @info "Learned dynamic noisy observations weights on validation"
    for i = 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end
    @info "Observation noise precision κ" E_κ = round(mean(κ_posterior); digits = 4)

    # Ensemble predictions on test
    @info "Generating dynamic noisy observations ensemble predictions on test"
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :τ => deepcopy(τ_posteriors),
        :β => deepcopy(β_posteriors),
        :κ => deepcopy(κ_posterior),
    )
    prediction_array = [missing for _ = 1:n_test]

    infer_test = infer(
        model = univariate_dynamic_noisy_ensemble(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            priors = posterior_priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = univariate_dynamic_noisy_ensemble_constraints(posterior_priors, true),
        initialization = univariate_dynamic_noisy_ensemble_init(posterior_priors),
        iterations = spec.prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, y_test)

    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics" ensemble_metrics...

    results = (
        w_posteriors = w_posteriors,
        τ_posteriors = τ_posteriors,
        β_posteriors = β_posteriors,
        κ_posterior = κ_posterior,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = y_test,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = spec.column,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

# ---------------------------------------------------------------------------
# Dynamic Multivariate pipeline
# ---------------------------------------------------------------------------

function run_dynamic_multivariate(spec::ExperimentSpecifier{Multivariate,Dynamic})
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)
    d = length(y_val[1])

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = multivariate_dynamic_ensemble(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = spec.priors,
    )
    constraints = multivariate_dynamic_ensemble_constraints(spec.priors, false)
    init = multivariate_dynamic_ensemble_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting dynamic multivariate ensemble on validation data" d n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    τ_posteriors = result.posteriors[:τ][end]
    β_posteriors = result.posteriors[:β][end]
    γ_posteriors = result.posteriors[:γ][end]

    γ_means_val = mean.(γ_posteriors)

    Yval_mat = reduce(hcat, y_val)
    @info "Learned dynamic weights on validation"
    for i = 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])  # (d, n_val)
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end

    # Ensemble predictions on test
    @info "Generating dynamic ensemble predictions on test"
    # deepcopy posteriors to prevent in-place mutation by LowRank product rules
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :τ => deepcopy(τ_posteriors),
        :β => deepcopy(β_posteriors),
    )

    infer_test = predict_with_model(
        spec.prediction_type,
        spec.model_type,
        posterior_priors;
        n_forecasters = n_forecasters,
        n_steps = n_test,
        prediction_array = [missing for _ = 1:n_test],
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = spec.prediction_iterations,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    Yte_mat = reduce(hcat, y_test)
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, Yte_mat)

    # Extract dynamic precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # Save results
    results = (
        w_posteriors = w_posteriors,
        τ_posteriors = τ_posteriors,
        β_posteriors = β_posteriors,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = Yte_mat,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = nothing,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

# ---------------------------------------------------------------------------
# Hierarchical Univariate pipeline
# ---------------------------------------------------------------------------

function run_hierarchical_univariate(spec::ExperimentSpecifier{Univariate,Hierarchical})
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = hierarchical_model(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = spec.priors,
    )
    constraints = hierarchical_constraints(spec.priors, false)
    init = hierarchical_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting hierarchical ensemble on validation data" n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    τ_posteriors = result.posteriors[:τ][end]
    ρ_posteriors = result.posteriors[:ρ][end]
    γ_posteriors = result.posteriors[:γ][end]  # (n_forecasters, n_val)

    γ_means_val = mean.(γ_posteriors)

    @info "Learned hierarchical weights on validation"
    for i = 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end

    # 6. Ensemble predictions on test
    @info "Generating hierarchical ensemble predictions on test"
    # deepcopy posteriors to prevent in-place mutation by LowRank product rules
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :τ => deepcopy(τ_posteriors),
        :ρ => deepcopy(ρ_posteriors),
        :α => spec.priors[:α],
    )
    prediction_array = [missing for _ = 1:n_test]

    infer_test = infer(
        model = hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            priors = posterior_priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = hierarchical_constraints(posterior_priors, true),
        initialization = hierarchical_init(posterior_priors),
        iterations = spec.prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, y_test)

    # Extract precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors = w_posteriors,
        τ_posteriors = τ_posteriors,
        ρ_posteriors = ρ_posteriors,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = y_test,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = spec.column,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

# ---------------------------------------------------------------------------
# Hierarchical Multivariate pipeline
# ---------------------------------------------------------------------------

function run_hierarchical_multivariate(spec::ExperimentSpecifier{Multivariate,Hierarchical})
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)
    d = length(y_val[1])

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = multivariate_hierarchical_model(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = spec.priors,
    )
    constraints = multivariate_hierarchical_constraints(spec.priors, false)
    init = multivariate_hierarchical_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting hierarchical multivariate ensemble on validation data" d n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    τ_posteriors = result.posteriors[:τ][end]
    ρ_posteriors = result.posteriors[:ρ][end]
    γ_posteriors = result.posteriors[:γ][end]

    γ_means_val = mean.(γ_posteriors)
    Yval_mat = reduce(hcat, y_val)
    @info "Learned hierarchical weights on validation"
    for i = 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end

    @info "Generating hierarchical ensemble predictions on test"
    # deepcopy posteriors to prevent in-place mutation by LowRank product rules
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :τ => deepcopy(τ_posteriors),
        :ρ => deepcopy(ρ_posteriors),
        :α => spec.priors[:α],
    )
    prediction_array = [missing for _ = 1:n_test]

    infer_test = infer(
        model = multivariate_hierarchical_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            priors = posterior_priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = multivariate_hierarchical_constraints(posterior_priors, true),
        initialization = multivariate_hierarchical_init(posterior_priors),
        iterations = spec.prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    Yte_mat = reduce(hcat, y_test)
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, Yte_mat)

    # Extract precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # Save results
    results = (
        w_posteriors = w_posteriors,
        τ_posteriors = τ_posteriors,
        ρ_posteriors = ρ_posteriors,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = Yte_mat,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = nothing,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

function run_deep_multivariate(spec::ExperimentSpecifier{Multivariate,Deep})
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)
    d = length(y_val[1])

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = multivariate_deep_model(
        n_forecasters = n_forecasters,
        n_obs = n_obs,
        priors = spec.priors,
    )
    constraints = multivariate_deep_constraints()
    init = multivariate_deep_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting deep multivariate ensemble on validation data" d n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    v_posteriors = result.posteriors[:v][end]
    τ_posteriors = result.posteriors[:τ][end]
    ρ_posteriors = result.posteriors[:ρ][end]
    γ_posteriors = result.posteriors[:γ][end]

    γ_means_val = mean.(γ_posteriors)

    Yval_mat = reduce(hcat, y_val)
    @info "Learned deep weights on validation"
    for i = 1:n_forecasters
        pred_mat_i = reduce(hcat, predictions_val[i, :])
        mse_i = mse_mv(pred_mat_i, Yval_mat)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end

    # Ensemble predictions on test
    @info "Generating deep ensemble predictions on test"
    # deepcopy posteriors to prevent in-place mutation by LowRank product rules
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :v => deepcopy(v_posteriors),
        :τ => deepcopy(τ_posteriors),
        :ρ => deepcopy(ρ_posteriors),
        :α => spec.priors[:α],
    )
    prediction_array = [missing for _ = 1:n_test]

    infer_test = infer(
        model = multivariate_deep_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            priors = posterior_priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = multivariate_deep_constraints(),
        initialization = multivariate_deep_init(posterior_priors),
        iterations = spec.prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    Yte_mat = reduce(hcat, y_test)
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, Yte_mat)

    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics (multivariate)" ensemble_metrics...

    # Save results
    results = (
        w_posteriors = w_posteriors,
        v_posteriors = v_posteriors,
        τ_posteriors = τ_posteriors,
        ρ_posteriors = ρ_posteriors,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = Yte_mat,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = nothing,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end

# ---------------------------------------------------------------------------
# Deep Univariate pipeline
# ---------------------------------------------------------------------------

function run_deep_univariate(spec::ExperimentSpecifier{Univariate,Deep})
    y_val, y_test, predictions_val, predictions_test, features_val, features_test =
        before_rxinfer(spec)
    n_forecasters = size(predictions_val, 1)
    n_val = length(y_val)
    n_test = length(y_test)

    if !isnothing(spec.subsample_percentage)
        n_obs = round(Int, n_val*spec.subsample_percentage)
    else
        n_obs = something(spec.subsample_size, n_val)
    end
    @info n_obs

    model = deep_model(n_forecasters = n_forecasters, n_obs = n_obs, priors = spec.priors)
    constraints = deep_constraints()
    init = deep_init(spec.priors)
    data = (y = y_val, features = features_val, predictions = predictions_val)

    @info "Fitting deep ensemble on validation data" n_forecasters n_val
    result = run_training_rxinfer(
        spec,
        model,
        data;
        constraints = constraints,
        initialization = init,
        showprogress = true,
    )

    free_energy = result.free_energy
    w_posteriors = result.posteriors[:w][end]
    v_posteriors = result.posteriors[:v][end]
    τ_posteriors = result.posteriors[:τ][end]
    ρ_posteriors = result.posteriors[:ρ][end]
    γ_posteriors = result.posteriors[:γ][end]

    γ_means_val = mean.(γ_posteriors)

    @info "Learned deep weights on validation"
    for i = 1:n_forecasters
        mse_i = mse(predictions_val[i, :], y_val)
        avg_γ = mean(γ_means_val[i, :])
        @info "Expert $i" avg_E_γ = round(avg_γ; digits = 4) val_MSE =
            round(mse_i; digits = 6)
    end

    # 6. Ensemble predictions on test
    @info "Generating deep ensemble predictions on test"
    # deepcopy posteriors to prevent in-place mutation by LowRank product rules
    posterior_priors = Dict{Symbol,Any}(
        :w => deepcopy(w_posteriors),
        :v => deepcopy(v_posteriors),
        :τ => deepcopy(τ_posteriors),
        :ρ => deepcopy(ρ_posteriors),
        :α => spec.priors[:α],
    )
    prediction_array = [missing for _ = 1:n_test]

    infer_test = infer(
        model = deep_model(
            n_forecasters = n_forecasters,
            n_obs = n_test,
            priors = posterior_priors,
        ),
        data = (
            y = prediction_array,
            features = features_test,
            predictions = predictions_test,
        ),
        constraints = deep_constraints(),
        initialization = deep_init(posterior_priors),
        iterations = spec.prediction_iterations,
        free_energy = false,
        showprogress = true,
    )

    ensemble_preds = infer_test.predictions[:y][end]
    (; ensemble_mean, ensemble_std, ensemble_metrics) =
        compute_ensemble_metrics(spec.prediction_type, ensemble_preds, y_test)

    # Extract precision weights on test
    γ_test_posteriors = infer_test.posteriors[:γ][end]
    γ_means_test = mean.(γ_test_posteriors)

    @info "Ensemble test metrics" ensemble_metrics...

    # 8. Save results
    results = (
        w_posteriors = w_posteriors,
        v_posteriors = v_posteriors,
        τ_posteriors = τ_posteriors,
        ρ_posteriors = ρ_posteriors,
        γ_test = γ_means_test,
        free_energy = free_energy,
        ensemble_mean = ensemble_mean,
        ensemble_std = ensemble_std,
        ensemble_metrics = ensemble_metrics,
        predictions_test = predictions_test,
        y_test = y_test,
        spec = (
            prediction_type = string(typeof(spec.prediction_type)),
            model_type = string(typeof(spec.model_type)),
            column = spec.column,
            horizon = spec.horizon,
            dataset = typeof(spec.dataset).parameters[1],
            dataset_path = spec.dataset_path,
            experts = spec.experts,
            selected_quantiles = spec.selected_quantiles,
            number_of_quantiles = spec.number_of_quantiles,
        ),
    )

    return results
end
