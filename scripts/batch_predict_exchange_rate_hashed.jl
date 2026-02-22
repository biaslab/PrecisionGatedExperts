#!/usr/bin/env julia

using ProbabilisticEnsembling
using CSV
using DataFrames
using JLD2

const DEFAULT_RESULTS_DIR = "final_results"
const DEFAULT_GRID_CSV = "results/hier_hyper_grid.csv"
const DEFAULT_OUT_CSV = "results/exchange_rate_hashed_predictions_steps_1_2_8.csv"
const PREDICTION_ITERATIONS = [1, 2, 8]

function parse_horizon(filename::AbstractString)
    m = match(r"_h(\d+)_", filename)
    isnothing(m) && return missing
    return parse(Int, m.captures[1])
end

function parse_hash(filename::AbstractString)
    m = match(r"_([0-9]+)\.jld2$", filename)
    isnothing(m) && return missing
    return m.captures[1]
end

function list_exchange_rate_hashed_models(results_dir::AbstractString)
    pat = r"^exchange_rate_.*probabilisticensembling\.hierarchical_[0-9]+\.jld2$"
    return sort(filter(f -> occursin(pat, f), readdir(results_dir)))
end

function load_training_metrics(path::AbstractString)
    data = JLD2.load(path)
    metrics = data["ensemble_metrics"]
    return (mse = Float64(metrics.mse), mae = Float64(metrics.mae))
end

function build_param_lookup(grid_csv::AbstractString)
    df = CSV.read(grid_csv, DataFrame)
    df = filter(r -> r.dataset == "exchange_rate" && r.status == "ok", df)
    return df
end

function match_params(lookup::DataFrame, horizon::Int, mse::Float64, mae::Float64)
    rows = filter(r -> r.horizon == horizon, lookup)
    nrow(rows) == 0 && return (
        w_scale = missing,
        subsample_percentage = missing,
        match_error = missing,
        grid_mse = missing,
        grid_mae = missing,
    )

    best_idx = 0
    best_err = Inf
    for (i, r) in enumerate(eachrow(rows))
        err = abs(Float64(r.mse) - mse) + abs(Float64(r.mae) - mae)
        if err < best_err
            best_err = err
            best_idx = i
        end
    end
    best = rows[best_idx, :]
    return (
        w_scale = Float64(best.w_scale),
        subsample_percentage = Float64(best.subsample_percentage),
        match_error = best_err,
        grid_mse = Float64(best.mse),
        grid_mae = Float64(best.mae),
    )
end

function run_one_model(model_path::AbstractString, pred_iters::Int)
    out = ProbabilisticEnsembling.predict_from_trained_ensemble(model_path, pred_iters)
    m = out.ensemble_metrics
    return (
        n_prediction_steps = Int(out.n_prediction_steps),
        mse = Float64(m.mse),
        mae = Float64(m.mae),
        rmse = Float64(m.rmse),
        r2 = Float64(m.r2),
        mape = Float64(m.mape),
        smape = Float64(m.smape),
    )
end

function main()
    results_dir = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_RESULTS_DIR
    grid_csv = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_GRID_CSV
    out_csv = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_OUT_CSV

    isdir(results_dir) || error("Results directory not found: $results_dir")
    isfile(grid_csv) || error("Grid CSV not found: $grid_csv")

    lookup = build_param_lookup(grid_csv)
    files = list_exchange_rate_hashed_models(results_dir)
    isempty(files) && error("No hashed exchange_rate hierarchical files found in: $results_dir")

    rows = DataFrame(
        model_file = String[],
        model_path = String[],
        model_hash = Union{Missing, String}[],
        horizon = Union{Missing, Int}[],
        training_mse = Float64[],
        training_mae = Float64[],
        w_scale = Union{Missing, Float64}[],
        subsample_percentage = Union{Missing, Float64}[],
        config_match_error = Union{Missing, Float64}[],
        grid_prediction_iterations = Int[],
        grid_mse_pred4 = Union{Missing, Float64}[],
        grid_mae_pred4 = Union{Missing, Float64}[],
        prediction_iterations = Int[],
        n_prediction_steps = Union{Missing, Int}[],
        mse = Union{Missing, Float64}[],
        mae = Union{Missing, Float64}[],
        rmse = Union{Missing, Float64}[],
        r2 = Union{Missing, Float64}[],
        mape = Union{Missing, Float64}[],
        smape = Union{Missing, Float64}[],
        status = String[],
        error = String[],
    )

    total = length(files) * length(PREDICTION_ITERATIONS)
    idx = 0

    for file in files
        full_path = joinpath(results_dir, file)
        horizon = parse_horizon(file)
        model_hash = parse_hash(file)

        train_metrics = load_training_metrics(full_path)
        param_match = horizon isa Missing ? (
            w_scale = missing,
            subsample_percentage = missing,
            match_error = missing,
            grid_mse = missing,
            grid_mae = missing,
        ) :
                      match_params(lookup, horizon, train_metrics.mse, train_metrics.mae)

        for pred_iters in PREDICTION_ITERATIONS
            idx += 1
            println("[$idx/$total] $file | prediction_iterations=$pred_iters")
            try
                pred = run_one_model(full_path, pred_iters)
                push!(
                    rows,
                    (
                        model_file = file,
                        model_path = full_path,
                        model_hash = model_hash,
                        horizon = horizon,
                        training_mse = train_metrics.mse,
                        training_mae = train_metrics.mae,
                        w_scale = param_match.w_scale,
                        subsample_percentage = param_match.subsample_percentage,
                        config_match_error = param_match.match_error,
                        grid_prediction_iterations = 4,
                        grid_mse_pred4 = param_match.grid_mse,
                        grid_mae_pred4 = param_match.grid_mae,
                        prediction_iterations = pred_iters,
                        n_prediction_steps = pred.n_prediction_steps,
                        mse = pred.mse,
                        mae = pred.mae,
                        rmse = pred.rmse,
                        r2 = pred.r2,
                        mape = pred.mape,
                        smape = pred.smape,
                        status = "ok",
                        error = "",
                    ),
                )
            catch err
                push!(
                    rows,
                    (
                        model_file = file,
                        model_path = full_path,
                        model_hash = model_hash,
                        horizon = horizon,
                        training_mse = train_metrics.mse,
                        training_mae = train_metrics.mae,
                        w_scale = param_match.w_scale,
                        subsample_percentage = param_match.subsample_percentage,
                        config_match_error = param_match.match_error,
                        grid_prediction_iterations = 4,
                        grid_mse_pred4 = param_match.grid_mse,
                        grid_mae_pred4 = param_match.grid_mae,
                        prediction_iterations = pred_iters,
                        n_prediction_steps = missing,
                        mse = missing,
                        mae = missing,
                        rmse = missing,
                        r2 = missing,
                        mape = missing,
                        smape = missing,
                        status = "error",
                        error = sprint(showerror, err),
                    ),
                )
                @warn "Prediction failed" file pred_iters err
            end

            mkpath(dirname(out_csv))
            CSV.write(out_csv, rows)
        end
    end

    println("Done. Wrote CSV to: $out_csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
