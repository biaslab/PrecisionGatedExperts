#!/usr/bin/env julia

"""
Run dynamic neural ensemble for selected datasets/horizons and write CSV summaries.

Usage:
    julia scripts/run_dynamic_neural_ensemble_batch.jl
"""

const DATASETS = ["ETTh1", "ETTh2", "exchange_rate"]
const HORIZONS = [720]
const MODEL_TYPES = ["MLP", "CNN", "LSTM"]

function find_model_file(models_dir::AbstractString, dataset::AbstractString, horizon::Int, model_type::AbstractString)
    direct = joinpath(models_dir, "$(dataset)_h$(horizon)_$(model_type)_enzyme.jld2")
    with_seq = joinpath(models_dir, "$(dataset)_h$(horizon)_s$(horizon)_$(model_type)_enzyme.jld2")

    # Rule:
    # - horizon 96: use only checkpoints without explicit seq_len suffix.
    # - other horizons: use only checkpoints with _s<horizon>_ suffix.
    if horizon == 96
        if isfile(direct)
            return direct
        end
        return nothing
    else
        if isfile(with_seq)
            return with_seq
        end
        return nothing
    end
end

function parse_metric_lines(output::AbstractString)
    forecasters = Vector{NamedTuple{(:index, :path, :mse, :mae), Tuple{Int, String, Float64, Float64}}}()
    average = nothing
    dynamic = nothing

    for line in split(output, '\n')
        startswith(line, "METRIC|") || continue
        parts = split(line, '|')
        fields = Dict{String, String}()
        for part in parts[2:end]
            kv = split(part, '='; limit=2)
            if length(kv) == 2
                fields[kv[1]] = kv[2]
            end
        end

        kind = get(fields, "kind", "")
        if kind == "forecaster"
            idx = parse(Int, fields["index"])
            path = fields["path"]
            mse = parse(Float64, fields["mse"])
            mae = parse(Float64, fields["mae"])
            push!(forecasters, (index=idx, path=path, mse=mse, mae=mae))
        elseif kind == "average"
            average = (mse=parse(Float64, fields["mse"]), mae=parse(Float64, fields["mae"]))
        elseif kind == "dynamic"
            dynamic = (mse=parse(Float64, fields["mse"]), mae=parse(Float64, fields["mae"]))
        end
    end

    return forecasters, average, dynamic
end

function write_result_csv(file::AbstractString, model_names::Vector{String}, model_metrics, average, dynamic, horizon::Int)
    header = vcat(["index"], model_names, ["average", "dynamic"])

    mse_values = [string(m.mse) for m in model_metrics]
    mae_values = [string(m.mae) for m in model_metrics]
    horizon_values = fill(string(horizon), length(model_names) + 2)

    open(file, "w") do io
        println(io, join(header, ","))
        println(io, join(vcat(["mse"], mse_values, [string(average.mse), string(dynamic.mse)]), ","))
        println(io, join(vcat(["mae"], mae_values, [string(average.mae), string(dynamic.mae)]), ","))
        println(io, join(vcat(["horizon"], horizon_values), ","))
    end
end

function run_one(root::AbstractString, models_dir::AbstractString, results_dir::AbstractString, dataset::String, horizon::Int)
    model_paths = String[]
    model_names = String[]

    for model_type in MODEL_TYPES
        path = find_model_file(models_dir, dataset, horizon, model_type)
        if path === nothing
            @warn "Missing model checkpoint" dataset horizon model_type
            continue
        end
        @info "Found model checkpoint" dataset horizon model_type path
        push!(model_paths, path)
        push!(model_names, model_type)
    end

    if length(model_paths) < 2
        @warn "Not enough model checkpoints for ensemble (need at least 2)" dataset horizon found=length(model_paths)
        return
    end

    @info "Running ensemble task" dataset horizon models=model_paths

    script = joinpath(root, "scripts", "dynamic_neural_ensemble_rxinfer.jl")
    julia_bin = Base.julia_cmd().exec[1]
    cmd = Cmd(vcat([julia_bin, "--project=$(root)", script], model_paths))

    io = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd), stdout=io, stderr=io))
    output = String(take!(io))

    if proc.exitcode != 0
        @warn "Dynamic ensemble run failed" dataset horizon exitcode=proc.exitcode
        println(output)
        return
    end

    forecasters, average, dynamic = parse_metric_lines(output)

    if average === nothing || dynamic === nothing
        @warn "Missing parsed ensemble metrics" dataset horizon
        println(output)
        return
    end

    if length(forecasters) < length(model_paths)
        @warn "Missing parsed forecaster metrics" dataset horizon expected=length(model_paths) got=length(forecasters)
        println(output)
        return
    end

    sort!(forecasters; by=x -> x.index)
    model_metrics = Vector{Any}()
    model_names_csv = String[]

    # Keep user-selected neural models in fixed order first.
    for model_type in model_names
        needle = "_$(model_type)_enzyme.jld2"
        idx = findfirst(f -> occursin(needle, f.path), forecasters)
        if idx === nothing
            @warn "Missing parsed model metric" dataset horizon model_type
            println(output)
            return
        end
        push!(model_metrics, forecasters[idx])
        push!(model_names_csv, model_type)
    end

    # Add constant baselines if present in dynamic script output.
    q10_idx = findfirst(f -> f.path == "const_q10_train", forecasters)
    if q10_idx !== nothing
        push!(model_metrics, forecasters[q10_idx])
        push!(model_names_csv, "const_q10_train")
    end

    q90_idx = findfirst(f -> f.path == "const_q90_train", forecasters)
    if q90_idx !== nothing
        push!(model_metrics, forecasters[q90_idx])
        push!(model_names_csv, "const_q90_train")
    end

    out_file = joinpath(results_dir, "dynamic_neural_ensemble_$(dataset)_h$(horizon).csv")
    write_result_csv(out_file, model_names_csv, model_metrics, average, dynamic, horizon)
    @info "Saved result CSV" file=out_file
end

function main()
    root = normpath(joinpath(@__DIR__, ".."))
    models_dir = joinpath(root, "models")
    results_dir = joinpath(root, "results")
    mkpath(results_dir)

    for dataset in DATASETS
        for horizon in HORIZONS
            @info "Running batch item" dataset horizon
            run_one(root, models_dir, results_dir, dataset, horizon)
        end
    end

    @info "Batch run completed" results_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
