#!/usr/bin/env julia

using ProbabilisticEnsembling

const SESSIONS_ROOT = normpath(joinpath(@__DIR__, "..", "sessions"))
const FAILED_LIST_PATH = normpath(joinpath(@__DIR__, "..", "failed_sessions.txt"))

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
        # Include only */vae/*.yaml folders, excluding neural_ensemble (handled separately).
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

function parse_flags(args::Vector{String})
    dry_run = "--dry-run" in args
    continue_on_error = "--continue-on-error" in args
    return (; dry_run, continue_on_error)
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

    static_sessions = yaml_files_in_dir(joinpath(SESSIONS_ROOT, "static"))
    vae_sessions = collect_vae_sessions(SESSIONS_ROOT)
    neural_sessions = yaml_files_in_dir(joinpath(SESSIONS_ROOT, "neural_ensemble"))

    # VAE-feature sessions + static sessions + neural ensemble sessions.
    # Static has no feature variants; neural ensemble is treated as VAE-only.
    all_sessions = vcat(vae_sessions, static_sessions, neural_sessions)

    println("Sessions root: $SESSIONS_ROOT")
    println("VAE sessions: $(length(vae_sessions))")
    println("Static sessions: $(length(static_sessions))")
    println("Neural sessions: $(length(neural_sessions))")
    println("Total: $(length(all_sessions))")
    flags.dry_run && println("Dry run mode: no sessions will be executed")

    failures = String[]
    for session in all_sessions
        try
            run_session(session, flags.dry_run)
        catch err
            push!(failures, session)
            @error "Session failed" session error = sprint(showerror, err)
            flags.continue_on_error || rethrow(err)
        end
    end

    if isempty(failures)
        println("Done: all sessions completed.")
    else
        println("Done with failures: $(length(failures))")
        open(FAILED_LIST_PATH, "w") do io
            for f in failures
                println(io, replace(relpath(f, SESSIONS_ROOT), "\\" => "/"))
            end
        end
        println("Failed session list saved to: $FAILED_LIST_PATH")
        for f in failures
            println("  - $(replace(relpath(f, SESSIONS_ROOT), "\\" => "/"))")
        end
    end
end

main(ARGS)
