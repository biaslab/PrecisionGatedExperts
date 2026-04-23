#!/usr/bin/env julia

using ProbabilisticEnsembling
using YAML

const NEURAL_SESSIONS_DIR = normpath(joinpath(@__DIR__, "..", "sessions", "neural_ensemble"))
const FAILED_LIST_PATH = normpath(joinpath(@__DIR__, "..", "failed_neural_l1_h64_sessions.txt"))
const TARGET_LAYERS = 1
const TARGET_HIDDEN_DIM = 64

function parse_flags(args::Vector{String})
    dry_run = "--dry-run" in args
    continue_on_error = "--continue-on-error" in args
    return (; dry_run, continue_on_error)
end

function neural_session_files(dir::String)
    isdir(dir) || error("Neural sessions directory not found: $dir")
    files = [
        normpath(joinpath(dir, f)) for f in readdir(dir)
        if endswith(lowercase(f), ".yaml")
    ]
    sort!(files)
    return files
end

function override_gating_config(config::Dict)
    params = get(config, "params", nothing)
    params isa AbstractDict || error("Missing `params` in config")
    gating = get(params, "gating", nothing)
    gating isa AbstractDict || error("Missing `params.gating` in config")

    gating["layers"] = TARGET_LAYERS
    gating["hidden_dim"] = TARGET_HIDDEN_DIM
    return config
end

function run_with_override(session_path::String, tmp_dir::String, dry_run::Bool)
    rel = replace(relpath(session_path, NEURAL_SESSIONS_DIR), "\\" => "/")
    println("[neural] $rel  (layers=$TARGET_LAYERS, hidden_dim=$TARGET_HIDDEN_DIM)")

    dry_run && return

    cfg = YAML.load_file(session_path)
    cfg = override_gating_config(cfg)

    tmp_path = joinpath(tmp_dir, basename(session_path))
    YAML.write_file(tmp_path, cfg)
    run_neural_ensemble_experiment(tmp_path)
end

function main(args::Vector{String})
    flags = parse_flags(args)
    sessions = neural_session_files(NEURAL_SESSIONS_DIR)

    println("Neural sessions dir: $NEURAL_SESSIONS_DIR")
    println("Sessions found: $(length(sessions))")
    println("Override: gating.layers=$TARGET_LAYERS, gating.hidden_dim=$TARGET_HIDDEN_DIM")
    flags.dry_run && println("Dry run mode: no sessions will be executed")

    failures = String[]

    mktempdir() do tmp_dir
        for session in sessions
            try
                run_with_override(session, tmp_dir, flags.dry_run)
            catch err
                push!(failures, session)
                @error "Session failed" session error = sprint(showerror, err)
                flags.continue_on_error || rethrow(err)
            end
        end
    end

    if isempty(failures)
        println("Done: all neural sessions completed.")
    else
        open(FAILED_LIST_PATH, "w") do io
            for f in failures
                println(io, replace(relpath(f, NEURAL_SESSIONS_DIR), "\\" => "/"))
            end
        end
        println("Done with failures: $(length(failures))")
        println("Failed session list saved to: $FAILED_LIST_PATH")
        for f in failures
            println("  - $(replace(relpath(f, NEURAL_SESSIONS_DIR), "\\" => "/"))")
        end
    end
end

main(ARGS)

