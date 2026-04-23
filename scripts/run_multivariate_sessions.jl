#!/usr/bin/env julia

using ProbabilisticEnsembling
using YAML

const SESSIONS_ROOT = normpath(joinpath(@__DIR__, "..", "sessions"))
const FAILED_LIST_PATH = normpath(joinpath(@__DIR__, "..", "failed_multivariate_sessions.txt"))
const DEFAULT_RESULTS_DIR = normpath(joinpath(@__DIR__, "..", "final_results"))

function parse_flags(args::Vector{String})
    dry_run = false
    continue_on_error = false
    skip_existing = true
    results_dir = DEFAULT_RESULTS_DIR
    skip_list_file = nothing

    for arg in args
        if arg == "--dry-run"
            dry_run = true
        elseif arg == "--continue-on-error"
            continue_on_error = true
        elseif arg == "--no-skip-existing"
            skip_existing = false
        elseif startswith(arg, "--results-dir=")
            results_dir = normpath(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--skip-list-file=")
            skip_list_file = normpath(split(arg, "=", limit = 2)[2])
        else
            error("Unknown argument: $arg")
        end
    end

    return (; dry_run, continue_on_error, skip_existing, results_dir, skip_list_file)
end

function all_session_files(root::String)
    files = String[]
    for (dir, _, names) in walkdir(root)
        for name in names
            endswith(lowercase(name), ".yaml") || continue
            push!(files, normpath(joinpath(dir, name)))
        end
    end
    sort!(files)
    return files
end

function yaml_files_in_dir(dir::String)
    isdir(dir) || return String[]
    files = [
        normpath(joinpath(dir, f)) for f in readdir(dir)
        if endswith(lowercase(f), ".yaml")
    ]
    sort!(files)
    return files
end

function collect_vae_sessions(root::String)
    out = String[]
    for (dir, _, files) in walkdir(root)
        rel = replace(relpath(dir, root), "\\" => "/")
        # Only */vae/*.yaml folders; neural_ensemble is handled separately.
        if endswith(rel, "/vae") || rel == "vae"
            if startswith(rel, "neural_ensemble")
                continue
            end
            for f in files
                endswith(lowercase(f), ".yaml") || continue
                push!(out, normpath(joinpath(dir, f)))
            end
        end
    end
    sort!(out)
    return out
end

function is_multivariate_session(path::String)
    cfg = YAML.load_file(path)
    params = get(cfg, "params", nothing)
    params isa AbstractDict || return false
    pred = get(params, "prediction_type", nothing)
    pred === nothing && return false
    return lowercase(String(pred)) == "multivariate"
end

function infer_horizon_from_experts(params)::Union{Int,Nothing}
    experts = get(params, "experts", nothing)
    experts isa AbstractVector || return nothing
    isempty(experts) && return nothing
    first_expert = String(first(experts))
    m = match(r"_h(\d+)_", basename(first_expert))
    return isnothing(m) ? nothing : parse(Int, m.captures[1])
end

function session_result_prefix(path::String)::Union{String,Nothing}
    cfg = YAML.load_file(path)
    params = get(cfg, "params", nothing)
    params isa AbstractDict || return nothing
    dataset = get(params, "dataset", nothing)
    prediction_type = get(params, "prediction_type", nothing)
    dataset === nothing && return nothing
    prediction_type === nothing && return nothing

    ds = String(dataset)
    pt = lowercase(String(prediction_type))
    rel = replace(relpath(path, SESSIONS_ROOT), "\\" => "/")
    horizon = get(params, "horizon", nothing)
    h = horizon === nothing ? infer_horizon_from_experts(params) : Int(horizon)
    h === nothing && return nothing

    if startswith(rel, "neural_ensemble/")
        return "$(ds)_h$(h)_neural_ensemble_"
    end

    if pt == "multivariate"
        model_type = get(params, "model_type", nothing)
        model_type === nothing && return nothing
        return "$(ds)_h$(h)_multivariate_$(String(model_type))"
    end

    return nothing
end

function has_matching_result(path::String, result_files::Vector{String})
    prefix = session_result_prefix(path)
    prefix === nothing && return false
    return any(f -> startswith(f, prefix) && endswith(f, ".jld2"), result_files)
end

function read_skip_list(path::String)
    isfile(path) || error("Skip list file not found: $path")
    names = String[]
    for raw in readlines(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        push!(names, basename(line))
    end
    return names
end

function run_session(path::String, dry_run::Bool)
    rel = replace(relpath(path, SESSIONS_ROOT), "\\" => "/")
    if startswith(rel, "neural_ensemble/")
        println("[neural] $rel")
        dry_run || run_neural_ensemble_experiment(path)
    else
        println("[ensemble] $rel")
        dry_run || run_experiment(path)
    end
end

function main(args::Vector{String})
    flags = parse_flags(args)

    # Match run_all_vae_sessions scope first:
    # 1) VAE feature sessions under */vae/
    # 2) static sessions
    # 3) neural_ensemble sessions
    candidate_sessions = vcat(
        collect_vae_sessions(SESSIONS_ROOT),
        yaml_files_in_dir(joinpath(SESSIONS_ROOT, "static")),
        yaml_files_in_dir(joinpath(SESSIONS_ROOT, "neural_ensemble")),
    )

    # Then keep only multivariate configs.
    selected = [f for f in candidate_sessions if is_multivariate_session(f)]

    skipped = String[]
    if flags.skip_list_file !== nothing
        skip_names = read_skip_list(flags.skip_list_file)
        skipped = [f for f in selected if has_matching_result(f, skip_names)]
    elseif flags.skip_existing
        result_files = isdir(flags.results_dir) ? readdir(flags.results_dir) : String[]
        skipped = [f for f in selected if has_matching_result(f, result_files)]
    end
    skipped_set = Set(skipped)
    to_run = (flags.skip_list_file !== nothing || flags.skip_existing) ? [f for f in selected if !(f in skipped_set)] : selected

    println("Sessions root: $SESSIONS_ROOT")
    println("Multivariate sessions found: $(length(selected))")
    if flags.skip_list_file !== nothing
        println("Skip list file: $(flags.skip_list_file)")
        println("Matched in skip list (skipped): $(length(skipped))")
        println("Will run: $(length(to_run))")
    elseif flags.skip_existing
        println("Results dir: $(flags.results_dir)")
        println("Already completed (skipped): $(length(skipped))")
        println("Will run: $(length(to_run))")
    end
    flags.dry_run && println("Dry run mode: no sessions will be executed")

    failures = String[]
    for session in to_run
        try
            run_session(session, flags.dry_run)
        catch err
            push!(failures, session)
            @error "Session failed" session error = sprint(showerror, err)
            flags.continue_on_error || rethrow(err)
        end
    end

    if isempty(failures)
        println("Done: all multivariate sessions completed.")
    else
        open(FAILED_LIST_PATH, "w") do io
            for f in failures
                println(io, replace(relpath(f, SESSIONS_ROOT), "\\" => "/"))
            end
        end
        println("Done with failures: $(length(failures))")
        println("Failed session list saved to: $FAILED_LIST_PATH")
        for f in failures
            println("  - $(replace(relpath(f, SESSIONS_ROOT), "\\" => "/"))")
        end
    end
end

main(ARGS)
