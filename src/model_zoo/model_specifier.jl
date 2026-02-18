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

# model_type: static | dynamic | hierarchical
struct Static end
struct Dynamic end
struct Hierarchical end

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

function parse_priors(::Static, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()
    if haskey(cfg, "γ")
        γ_cfg = cfg["γ"]
        shape = get(γ_cfg, "shape", 1.0)
        rate  = get(γ_cfg, "rate", 1e-12)
        priors[:γ] = [GammaShapeRate(shape, rate) for _ in 1:n_forecasters]
    end
    return priors
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

# ---------------------------------------------------------------------------
# Dataset loading — dispatch on Val{:dataset_name}
# Add new methods for other dataset formats, e.g.:
#   load_dataset(::Val{:electricity}, path::String) = ...
# ---------------------------------------------------------------------------

load_dataset(::Val{:ETTh1}, path::String) = load_ett(path)
load_dataset(::Val{:ETTh2}, path::String) = load_ett(path)

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
