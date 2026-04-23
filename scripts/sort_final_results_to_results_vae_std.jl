#!/usr/bin/env julia

const SRC_DIR = normpath(joinpath(@__DIR__, "..", "final_results"))
const DST_ROOT = normpath(joinpath(@__DIR__, "..", "paper", "results_vae_std"))

function parse_flags(args::Vector{String})
    dry_run = "--dry-run" in args
    move_files = "--move" in args
    return (; dry_run, move_files)
end

function destination_subdir(filename::String)::Union{String,Nothing}
    # Keep this aligned with existing results_vae layout.
    if occursin("_noisy_experts_diagonal_", filename)
        return "noisy_diagonal"
    elseif occursin("_noisy_experts_", filename)
        return "noisy_experts"
    elseif occursin("_dynamic_diagonal_", filename)
        return "dynamic_diagonal"
    elseif occursin("_dynamic_", filename)
        return "dynamic"
    elseif occursin("_static_", filename)
        return "static"
    elseif occursin("_neural_ensemble_big_", filename)
        return nothing
    elseif occursin("_neural_ensemble_", filename)
        return nothing
    end
    return nothing
end

function main(args::Vector{String})
    flags = parse_flags(args)

    isdir(SRC_DIR) || error("Source directory not found: $SRC_DIR")
    mkpath(DST_ROOT)

    files = sort(filter(f -> endswith(lowercase(f), ".jld2"), readdir(SRC_DIR)))
    println("Source dir: $SRC_DIR")
    println("Destination root: $DST_ROOT")
    println("JLD2 files found: $(length(files))")
    println("Mode: " * (flags.move_files ? "move" : "copy"))
    flags.dry_run && println("Dry run mode: no files will be modified")

    copied = 0
    skipped = 0
    unknown = String[]

    for f in files
        sub = destination_subdir(f)
        if sub === nothing
            push!(unknown, f)
            skipped += 1
            continue
        end

        dst_dir = joinpath(DST_ROOT, sub)
        src = joinpath(SRC_DIR, f)
        dst = joinpath(dst_dir, f)

        println("[$sub] $f")
        if !flags.dry_run
            mkpath(dst_dir)
            if flags.move_files
                mv(src, dst; force = true)
            else
                cp(src, dst; force = true)
            end
        end
        copied += 1
    end

    println("Done.")
    println("Processed: $(copied)")
    println("Skipped (unknown pattern): $(skipped)")
    if !isempty(unknown)
        println("Unknown filenames:")
        for f in unknown
            println("  - $f")
        end
    end
end

main(ARGS)
