#!/usr/bin/env julia

using ProbabilisticEnsembling
using YAML
using CSV
using DataFrames
using Plots

const DATASETS = ["ETTh1", "ETTh2", "exchange_rate"]
const HORIZONS = [96, 192, 336, 720]
const W_SCALES = [0.1, 1.0]
const TAU_RATES = [1e-3, 1.0]
const SUBSAMPLE_SIZES = [10, 20]

function session_path(dataset::String, horizon::Int)
    return joinpath("sessions", "hierarchical", "hierarchical_$(dataset)_$(horizon).yaml")
end

function get_tau_prior(priors::Dict)
    if haskey(priors, "τ")
        return priors["τ"]
    elseif haskey(priors, "tau")
        return priors["tau"]
    else
        error("Could not find τ (or tau) prior in YAML priors.")
    end
end

function one_run(dataset::String, horizon::Int, w_scale::Float64, tau_rate::Float64, subsample_size::Int)
    path = session_path(dataset, horizon)
    isfile(path) || error("Session file not found: $path")

    config = YAML.load_file(path)
    params = config["params"]
    priors = params["priors"]
    tau_prior = get_tau_prior(priors)
    w_prior = priors["w"]

    w_prior["scale"] = w_scale
    tau_prior["rate"] = tau_rate
    params["subsample_size"] = subsample_size

    tmp_dir = mktempdir()
    tmp_yaml = joinpath(tmp_dir, "session.yaml")
    YAML.write_file(tmp_yaml, config)
    results = ProbabilisticEnsembling.run_experiment(tmp_yaml)
    metrics = results.ensemble_metrics

    return (
        mse = Float64(metrics.mse),
        mae = Float64(metrics.mae),
        results = results,
    )
end

function param_tag(v::Real)
    s = string(v)
    s = replace(s, "." => "p")
    s = replace(s, "-" => "m")
    return s
end

function save_prediction_plot(results, dataset::String, horizon::Int, w_scale::Float64, tau_rate::Float64, subsample_size::Int)
    if !(dataset in ("ETTh1", "ETTh2"))
        return nothing
    end

    y_test = collect(results.y_test)
    y_pred = collect(results.ensemble_mean)
    x = 1:length(y_test)
    plt = plot(
        x,
        y_test;
        label = "True",
        linewidth = 2,
        color = :black,
        linestyle = :dash,
        xlabel = "t",
        ylabel = "OT",
        title = "Hierarchical prediction: $(dataset), h=$(horizon)",
    )
    plot!(plt, x, y_pred; label = "Ensemble", linewidth = 2, color = :dodgerblue)

    plots_dir = joinpath("results", "hierarchical_prediction_plots", dataset, "h$(horizon)")
    mkpath(plots_dir)
    filename = "pred_ws$(param_tag(w_scale))_tr$(param_tag(tau_rate))_ss$(subsample_size).png"
    out_path = joinpath(plots_dir, filename)
    savefig(plt, out_path)
    return out_path
end

function main()
    out_path = length(ARGS) >= 1 ? ARGS[1] : "results/hierarchical_hyperparam_grid.csv"
    mkpath(dirname(out_path))

    rows = DataFrame(
        dataset = String[],
        horizon = Int[],
        w_scale = Float64[],
        tau_rate = Float64[],
        subsample_size = Int[],
        mse = Union{Missing, Float64}[],
        mae = Union{Missing, Float64}[],
        status = String[],
        error = String[],
    )

    total = length(DATASETS) * length(HORIZONS) * length(W_SCALES) * length(TAU_RATES) * length(SUBSAMPLE_SIZES)
    run_idx = 0

    for dataset in DATASETS
        for horizon in HORIZONS
            for w_scale in W_SCALES
                for tau_rate in TAU_RATES
                    for subsample_size in SUBSAMPLE_SIZES
                        run_idx += 1
                        println("[$run_idx/$total] dataset=$dataset horizon=$horizon w.scale=$w_scale τ.rate=$tau_rate subsample_size=$subsample_size")
                        try
                            run_out = one_run(dataset, horizon, w_scale, tau_rate, subsample_size)
                            save_prediction_plot(run_out.results, dataset, horizon, w_scale, tau_rate, subsample_size)
                            push!(
                                rows,
                                (
                                    dataset = dataset,
                                    horizon = horizon,
                                    w_scale = w_scale,
                                    tau_rate = tau_rate,
                                    subsample_size = subsample_size,
                                    mse = run_out.mse,
                                    mae = run_out.mae,
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
                                    w_scale = w_scale,
                                    tau_rate = tau_rate,
                                    subsample_size = subsample_size,
                                    mse = missing,
                                    mae = missing,
                                    status = "error",
                                    error = sprint(showerror, err),
                                ),
                            )
                            @warn "Run failed" dataset horizon w_scale tau_rate subsample_size err
                        end

                        CSV.write(out_path, rows)
                    end
                end
            end
        end
    end

    println("Finished. CSV written to: $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
