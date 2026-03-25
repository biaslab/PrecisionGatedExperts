#!/usr/bin/env julia

using CSV
using DataFrames
using JLD2

using ProbabilisticEnsembling

const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results_vae")
const OUTPUT_FILE = joinpath(@__DIR__, "..", "paper", "results_quantiles", "quantile_performance.csv")
const MODEL_DIRS = [
    "static",
    "dynamic",
    "dynamic_diagonal",
    "noisy_experts",
    "noisy_diagonal",
]

function collect_result_files()
    files = String[]
    for dir_name in MODEL_DIRS
        dir = joinpath(RESULTS_DIR, dir_name)
        isdir(dir) || continue
        for name in readdir(dir)
            endswith(name, ".jld2") || continue
            push!(files, joinpath(dir, name))
        end
    end
    sort!(files)
    return files
end

function model_dir_to_type(path::String)
    dir_name = basename(dirname(path))
    return dir_name == "noisy_diagonal" ? "noisy_experts_diagonal" : dir_name
end

function export_file_quantile_performance(path::String)
    saved = JLD2.load(path)
    spec = ProbabilisticEnsembling._spec_for_prediction_from_saved(saved, 1)

    y_val_all, y_test_all, _, predictions_test, _, _ =
        ProbabilisticEnsembling.before_rxinfer(spec)
    n_steps = length(y_test_all)
    y_test = ProbabilisticEnsembling.prepare_y_test(spec.prediction_type, y_test_all, n_steps)
    y_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(spec.prediction_type, y_test)

    metrics = ProbabilisticEnsembling.export_quantile_performance(
        spec.prediction_type,
        predictions_test,
        y_metrics,
        spec.selected_quantiles,
    )

    dataset = string(typeof(spec.dataset).parameters[1])
    prediction_type = ProbabilisticEnsembling.model_prediction_name(spec.prediction_type)
    model_type = model_dir_to_type(path)
    column = isnothing(spec.column) ? missing : spec.column

    return [
        (
            dataset = dataset,
            horizon = spec.horizon,
            prediction_type = prediction_type,
            column = column,
            model_type = model_type,
            source_file = basename(path),
            quantile = row.quantile,
            forecaster_index = row.index,
            mse = row.mse,
            mae = row.mae,
        ) for row in metrics
    ]
end

function main()
    rows = NamedTuple[]
    for path in collect_result_files()
        append!(rows, export_file_quantile_performance(path))
    end

    mkpath(dirname(OUTPUT_FILE))
    CSV.write(OUTPUT_FILE, DataFrame(rows))
    println("Saved quantile performance to $(OUTPUT_FILE)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
