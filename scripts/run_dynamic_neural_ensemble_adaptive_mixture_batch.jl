#!/usr/bin/env julia

"""
Run adaptive-mixture dynamic neural ensemble for selected datasets/horizons.
Produces one CSV per dataset+horizon:
- softmax_moe_<dataset>_h<horizon>.csv

Usage:
    julia scripts/run_dynamic_neural_ensemble_adaptive_mixture_batch.jl
"""

const DATASETS = ["ETTh2", "exchange_rate", "ETTh1"]
const HORIZONS = [720, 96, 192, 336 ]

function discover_model_files(models_dir::AbstractString, dataset::AbstractString, horizon::Int)
    pattern = "$(dataset)_h$(horizon)_"
    paths = filter(f -> startswith(basename(f), pattern) && contains(basename(f), "_s"),
                   readdir(models_dir; join = true))
    paths = filter(f -> endswith(f, ".jld2"), paths)
    sort!(paths)
    return paths
end

function parse_metric_lines(output::AbstractString)
    forecasters = Vector{NamedTuple{(:index, :path, :mse, :mae), Tuple{Int, String, Float64, Float64}}}()
    average = nothing
    dynamic = nothing

    extract_number(s::AbstractString) = begin
        m = match(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)
        m === nothing && error("Cannot parse numeric value from: $(s)")
        return parse(Float64, m.match)
    end

    metric_records = String[]
    for line in split(output, '\n')
        for m in eachmatch(r"METRIC\|[^\r\n]*", line)
            push!(metric_records, m.match)
        end
    end

    for record in metric_records
        parts = split(record, '|')
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
            mse = extract_number(fields["mse"])
            mae = extract_number(fields["mae"])
            push!(forecasters, (index=idx, path=path, mse=mse, mae=mae))
        elseif kind == "average"
            average = (mse=extract_number(fields["mse"]), mae=extract_number(fields["mae"]))
        elseif kind == "dynamic"
            dynamic = (mse=extract_number(fields["mse"]), mae=extract_number(fields["mae"]))
        end
    end

    return forecasters, average, dynamic
end

function write_result_csv(file::AbstractString, dynamic_train, dynamic_val, horizon::Int)
    open(file, "w") do io
        println(io, "index,softmax_train,softmax_val")
        println(io, "mse,$(dynamic_train.mse),$(dynamic_val.mse)")
        println(io, "mae,$(dynamic_train.mae),$(dynamic_val.mae)")
        println(io, "horizon,$(horizon),$(horizon)")
    end
end

function run_one_mode(root::AbstractString, dataset::String, horizon::Int, model_paths::Vector{String}, train_set::Bool)
    script = joinpath(root, "scripts", "dynamic_neural_ensemble_adaptive_mixture_local_experts.jl")
    julia_bin = Base.julia_cmd().exec[1]
    train_set_str = train_set ? "true" : "false"
    cmd = Cmd(vcat([julia_bin, "--project=$(root)", script, "--train_set", train_set_str], model_paths))

    mode_label = train_set ? "softmax_train" : "softmax_val"
    @info "Running adaptive-mixture mode" dataset horizon mode=mode_label models=model_paths

    io = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd), stdout=io, stderr=io))
    output = String(take!(io))

    if proc.exitcode != 0
        @warn "Adaptive-mixture run failed" dataset horizon mode=mode_label exitcode=proc.exitcode
        println(output)
        return
    end

    _, average, dynamic = parse_metric_lines(output)
    if average === nothing || dynamic === nothing
        @warn "Missing parsed ensemble metrics" dataset horizon mode=mode_label
        println(output)
        return
    end

    return (average=average, dynamic=dynamic)
end

function run_one(root::AbstractString, models_dir::AbstractString, results_dir::AbstractString, dataset::String, horizon::Int)
    model_paths = discover_model_files(models_dir, dataset, horizon)

    if length(model_paths) < 2
        @warn "Not enough model checkpoints for ensemble (need at least 2)" dataset horizon found=length(model_paths)
        return
    end

    model_names = [splitext(basename(path))[1] for path in model_paths]
    @info "Found model checkpoints" dataset horizon count=length(model_paths) model_names

    train_res = run_one_mode(root, dataset, horizon, model_paths, true)
    val_res = run_one_mode(root, dataset, horizon, model_paths, false)
    if train_res === nothing || val_res === nothing
        @warn "Skipping combined CSV because one mode failed" dataset horizon
        return
    end

    out_file = joinpath(results_dir, "softmax_moe_$(dataset)_h$(horizon).csv")
    write_result_csv(out_file, train_res.dynamic, val_res.dynamic, horizon)
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
