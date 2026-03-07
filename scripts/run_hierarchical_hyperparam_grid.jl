#!/usr/bin/env julia

using ProbabilisticEnsembling
using YAML
using CSV
using DataFrames

const DATASETS = ["traffic", "exchange_rate"]
const HORIZONS = [96, 192, 336, 720]
const ITERS = [1, 5, 10]

function session_path(dataset::String, horizon::Int)
    return joinpath("sessions", "dynamic", "dynamic_$(dataset)_$(horizon).yaml")
end

function one_run(dataset::String, horizon::Int, inference_iterations::Int)
    path = session_path(dataset, horizon)
    isfile(path) || error("Session file not found: $path")

    config = YAML.load_file(path)
    params = config["params"]
    params["inference_iterations"] = inference_iterations

    tmp_dir = mktempdir()
    tmp_yaml = joinpath(tmp_dir, "session.yaml")
    YAML.write_file(tmp_yaml, config)

    results = ProbabilisticEnsembling.run_experiment(tmp_yaml)
    metrics = results.ensemble_metrics
    nll = hasproperty(metrics, :nll) ? Float64(metrics.nll) : missing

    return (
        mse = Float64(metrics.mse),
        mae = Float64(metrics.mae),
        nll = nll,
    )
end

function main()
    out_path = length(ARGS) >= 1 ? ARGS[1] : "results/dynamic_hyperparam_grid_3.csv"
    mkpath(dirname(out_path))

    rows = DataFrame(
        dataset = String[],
        horizon = Int[],
        inference_iterations = Int[],
        mse = Union{Missing, Float64}[],
        mae = Union{Missing, Float64}[],
        nll = Union{Missing, Float64}[],
        status = String[],
        error = String[],
    )

    total = length(DATASETS) * length(HORIZONS) * length(ITERS)
    run_idx = 0

    for dataset in DATASETS
        for horizon in HORIZONS
            for inference_iterations in ITERS
                run_idx += 1
                println("[$run_idx/$total] dataset=$dataset horizon=$horizon inference_iterations=$inference_iterations")
                try
                    run_out = one_run(dataset, horizon, inference_iterations)
                    push!(
                        rows,
                        (
                            dataset = dataset,
                            horizon = horizon,
                            inference_iterations = inference_iterations,
                            mse = run_out.mse,
                            mae = run_out.mae,
                            nll = run_out.nll,
                            status = "ok",
                            error = "",
                        ),
                    )
                catch err
                    push!(
                        rows,
                        (
                            dataset = dataset,
                            horizon = horizon,
                            inference_iterations = inference_iterations,
                            mse = missing,
                            mae = missing,
                            nll = missing,
                            status = "error",
                            error = sprint(showerror, err),
                        ),
                    )
                    @warn "Run failed" dataset horizon inference_iterations err
                end

                CSV.write(out_path, rows)
            end
        end
    end

    println("Finished. CSV written to: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
