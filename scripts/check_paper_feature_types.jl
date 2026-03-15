using JLD2

function get_feature_type(data)
    if haskey(data, "raw_spec")
        raw_spec = data["raw_spec"]
        if raw_spec isa AbstractDict
            if haskey(raw_spec, "params") && raw_spec["params"] isa AbstractDict &&
               haskey(raw_spec["params"], "feature_type")
                return string(raw_spec["params"]["feature_type"])
            elseif haskey(raw_spec, "feature_type")
                return string(raw_spec["feature_type"])
            end
        end
    end

    if haskey(data, "spec")
        spec = data["spec"]
        if spec isa NamedTuple && hasproperty(spec, :feature_type)
            return string(getproperty(spec, :feature_type))
        elseif spec isa AbstractDict && haskey(spec, "feature_type")
            return string(spec["feature_type"])
        end
    end

    return nothing
end

function expected_feature_type(path::AbstractString)
    norm = replace(path, '\\' => '/')

    if startswith(norm, "paper/results_vae/")
        return "vae"
    elseif startswith(norm, "paper/results/")
        if occursin("/ETTh1_", norm) || occursin("/ETTh2_", norm)
            return "uniwindow"
        elseif occursin("/exchange_rate_", norm)
            return "window"
        end
    end

    return nothing
end

function report_root(root::String)
    println("ROOT=$root")

    dirs = String[]
    for (dir, _, files) in walkdir(root)
        any(endswith(file, ".jld2") for file in files) || continue
        push!(dirs, dir)
    end
    sort!(dirs)

    for dir in dirs
        counts = Dict{String,Int}()
        bad = String[]
        missing = String[]
        total = 0

        for file in sort(readdir(dir))
            endswith(file, ".jld2") || continue
            path = joinpath(dir, file)
            total += 1

            feature_type = get_feature_type(JLD2.load(path))
            if isnothing(feature_type)
                push!(missing, file)
                continue
            end

            counts[feature_type] = get(counts, feature_type, 0) + 1
            expected = expected_feature_type(path)

            if !isnothing(expected) && feature_type != expected
                push!(bad, "$file => $feature_type (expected $expected)")
            elseif root == "paper/results" && isnothing(expected) &&
                   !(feature_type in ("window", "uniwindow"))
                push!(bad, "$file => $feature_type (expected window or uniwindow)")
            end
        end

        type_summary = isempty(counts) ? "(none)" :
                       join(["$k:$(counts[k])" for k in sort!(collect(keys(counts)))], ", ")

        println("DIR=$dir")
        println("TOTAL=$total")
        println("TYPES=$type_summary")
        println("BAD=$(length(bad))")
        for entry in bad
            println("  $entry")
        end
        println("MISSING=$(length(missing))")
        for entry in missing
            println("  $entry")
        end
    end
end

report_root("paper/results")
report_root("paper/results_vae")
