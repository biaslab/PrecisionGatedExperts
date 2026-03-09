# ---------------------------------------------------------------------------
# Neural Ensemble Specifier — YAML config + entry points
# ---------------------------------------------------------------------------

using YAML
using JLD2
using SHA

export run_neural_ensemble_experiment, predict_from_trained_neural_ensemble

# ---------------------------------------------------------------------------
# YAML → NeuralEnsembleSpecifier
# ---------------------------------------------------------------------------

function _parse_neural_ensemble_spec(config::Dict)
    params = config["params"]

    prediction_type = _parse_prediction_type(params["prediction_type"])
    column = get(params, "column", nothing)
    if prediction_type isa Univariate && column === nothing
        error("column is required when prediction_type is univariate")
    end

    dataset_name = Symbol(params["dataset"])
    dataset = Val{dataset_name}()
    dataset_path = params["dataset_path"]

    experts = String.(params["experts"])
    isempty(experts) && error("experts list must not be empty")

    train_set = params["train_set"]

    feature_type = parse_feature_type(params["feature_type"])

    selected_quantiles = Float64.(params["quantiles"])

    gating = params["gating"]
    gating_layers = gating["layers"]
    gating_hidden_dim = gating["hidden_dim"]
    n_epochs = gating["n_epochs"]
    patience = gating["patience"]
    min_delta = Float32(gating["min_delta"])
    learning_rate = Float32(gating["learning_rate"])

    save_dir = params["save_dir"]

    # Infer horizon from the first expert model
    first_expert = load_jld2_model(experts[1])
    horizon = Int(first_expert.meta.horizon)

    return NeuralEnsembleSpecifier(
        prediction_type,
        column,
        dataset,
        dataset_path,
        horizon,
        experts,
        train_set,
        gating_layers,
        gating_hidden_dim,
        n_epochs,
        patience,
        min_delta,
        learning_rate,
        feature_type,
        selected_quantiles,
        save_dir,
    )
end

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

function run_neural_ensemble_experiment(path_to_yaml::String)
    config = YAML.load_file(path_to_yaml)
    spec = _parse_neural_ensemble_spec(config)

    @info "Running neural ensemble experiment" dataset = spec.dataset horizon = spec.horizon prediction_type = spec.prediction_type train_set = spec.train_set
    results = run_neural_ensemble(spec, config)

    return results
end

function run_neural_ensemble_experiment(spec::NeuralEnsembleSpecifier)
    raw_spec = Dict(
        "params" => Dict(
            "pipeline" => "neural_ensemble",
            "prediction_type" => spec.prediction_type isa Univariate ? "univariate" : "multivariate",
            "column" => spec.column,
            "dataset" => string(typeof(spec.dataset).parameters[1]),
            "dataset_path" => spec.dataset_path,
            "experts" => spec.experts,
            "train_set" => spec.train_set,
            "feature_type" => string(typeof(spec.feature_type)),
            "quantiles" => spec.selected_quantiles,
            "gating" => Dict(
                "layers" => spec.gating_layers,
                "hidden_dim" => spec.gating_hidden_dim,
                "n_epochs" => spec.n_epochs,
                "patience" => spec.patience,
                "min_delta" => spec.min_delta,
                "learning_rate" => spec.learning_rate,
            ),
            "save_dir" => spec.save_dir,
        ),
    )
    return run_neural_ensemble(spec, raw_spec)
end

# ---------------------------------------------------------------------------
# Predict from trained model
# ---------------------------------------------------------------------------

function predict_from_trained_neural_ensemble(path_to_jld2::String)
    saved = load_neural_ensemble(path_to_jld2)

    gating_cfg = saved["gating_config"]
    gating = build_gating_network(gating_cfg.n_features, gating_cfg.n_experts, gating_cfg.layers, gating_cfg.hidden_dim)

    gating_ps = saved["gating_parameters"]
    gating_st = saved["gating_states"]

    raw_spec = saved["raw_spec"]
    spec = _parse_neural_ensemble_spec(raw_spec)

    @info "Loaded neural ensemble model" path = path_to_jld2

    data = before_neural_ensemble(spec)

    results = evaluate_neural_ensemble(
        spec.prediction_type, gating,
        gating_ps, gating_st,
        data.predictions_test_vec, data.features_test,
        data.y_test_mat, data.col_idx,
    )

    @info "Prediction metrics" results.ensemble_metrics...

    return (
        ensemble_mean = results.ensemble_mean,
        ensemble_std = results.ensemble_std,
        ensemble_metrics = results.ensemble_metrics,
        gating_weights = results.gating_weights,
        normal_predictions = results.normal_predictions,
        prediction_type = spec.prediction_type,
        experts = spec.experts,
        selected_quantiles = spec.selected_quantiles,
        n_forecasters = data.n_total,
        y_test_mat = data.y_test_mat,
        col_idx = data.col_idx,
    )
end
