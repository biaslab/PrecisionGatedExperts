using CSV
using DataFrames
using Distributions
using JLD2
using Statistics

using ProbabilisticEnsembling

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results_vae")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "..", "paper", "results_quantiles")
const DEFAULT_DATASET_PATHS = Dict(
    "ETTh1" => "data/ETTh1.csv",
    "ETTh2" => "data/ETTh2.csv",
    "electricity" => "data/electricity.csv",
    "exchange_rate" => "data/exchange_rate.csv",
    "traffic" => "data/traffic.csv",
)
const DEFAULT_HORIZONS = [96, 192, 336, 720]
const DEFAULT_MODEL_TYPES = [
    "static",
    "dynamic",
    "dynamic_diagonal",
    "noisy_experts",
    "noisy_experts_diagonal",
    "neural_ensemble",
    "neural_ensemble_big",
]
const MODEL_DIR = Dict(
    "noisy_experts" => "noisy_experts",
    "noisy_experts_diagonal" => "noisy_diagonal",
)
const DATASETS = [
    ("ETTh1", :univariate),
    ("ETTh2", :univariate),
    ("electricity", :multivariate),
    ("exchange_rate", :multivariate),
    ("traffic", :multivariate),
]

function maybe_get(obj, key::Symbol)
    if obj isa AbstractDict
        if haskey(obj, String(key))
            return obj[String(key)]
        elseif haskey(obj, key)
            return obj[key]
        end
        return nothing
    end
    return hasproperty(obj, key) ? getproperty(obj, key) : nothing
end

function print_usage()
    println(
        """
        Usage:
          julia --project=. scripts/export_quantile_performance.jl [options]

        Options:
          --quantiles=0.1,0.5,0.9     Quantiles in (0,1). Default: 0.1:0.1:0.9
          --datasets=ETTh1,traffic    Restrict to selected datasets
          --models=static,dynamic     Restrict to selected model types
          --horizons=96,192,336,720   Restrict to selected horizons
          --results-dir=PATH          Results root. Default: paper/results_vae
          --output-dir=PATH           Output directory. Default: paper/results_quantiles
          --seq-len=96                Sequence length used for reconstruction
          --split=0.6,0.2,0.2         Train/val/test split ratios
          --original-scale            Report losses on the original data scale
          --help                      Show this message
        """,
    )
end

function parse_csv_arg(raw::AbstractString, converter)
    isempty(strip(raw)) && return []
    return [converter(strip(part)) for part in split(raw, ',') if !isempty(strip(part))]
end

function parse_args(args)
    options = Dict{String,Any}(
        "quantiles" => collect(0.1:0.1:0.9),
        "datasets" => [ds for (ds, _) in DATASETS],
        "models" => copy(DEFAULT_MODEL_TYPES),
        "horizons" => copy(DEFAULT_HORIZONS),
        "results_dir" => DEFAULT_RESULTS_DIR,
        "output_dir" => DEFAULT_OUTPUT_DIR,
        "seq_len" => 96,
        "split" => (0.6, 0.2, 0.2),
        "original_scale" => false,
    )

    for arg in args
        if arg == "--help"
            print_usage()
            exit(0)
        elseif arg == "--original-scale"
            options["original_scale"] = true
        elseif startswith(arg, "--quantiles=")
            options["quantiles"] = parse_csv_arg(split(arg, "=", limit = 2)[2], x -> parse(Float64, x))
        elseif startswith(arg, "--datasets=")
            options["datasets"] = parse_csv_arg(split(arg, "=", limit = 2)[2], String)
        elseif startswith(arg, "--models=")
            options["models"] = parse_csv_arg(split(arg, "=", limit = 2)[2], String)
        elseif startswith(arg, "--horizons=")
            options["horizons"] = parse_csv_arg(split(arg, "=", limit = 2)[2], x -> parse(Int, x))
        elseif startswith(arg, "--results-dir=")
            options["results_dir"] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--output-dir=")
            options["output_dir"] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--seq-len=")
            options["seq_len"] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--split=")
            parts = parse_csv_arg(split(arg, "=", limit = 2)[2], x -> parse(Float64, x))
            length(parts) == 3 || error("--split must have exactly three comma-separated values")
            options["split"] = (parts[1], parts[2], parts[3])
        else
            error("Unknown argument: $arg")
        end
    end

    isempty(options["quantiles"]) && error("At least one quantile is required")
    all(0.0 < q < 1.0 for q in options["quantiles"]) ||
        error("Quantiles must lie strictly inside (0, 1): $(options["quantiles"])")

    return options
end

function result_filename(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    if startswith(model_type, "neural_ensemble")
        return "$(dataset)_h$(horizon)_neural_ensemble.jld2"
    elseif pred_type == :univariate
        return "$(dataset)_h$(horizon)_OT_$(model_type).jld2"
    else
        return "$(dataset)_h$(horizon)_multivariate_$(model_type).jld2"
    end
end

function resolve_result_path(results_dir::String, dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    dir_name = get(MODEL_DIR, model_type, model_type)
    dir = joinpath(results_dir, dir_name)
    isdir(dir) || return nothing

    fname = result_filename(dataset, horizon, pred_type, model_type)
    direct = joinpath(dir, fname)
    if isfile(direct)
        return direct
    end

    prefix = replace(fname, ".jld2" => "")
    matches = filter(f -> startswith(f, prefix) && endswith(f, ".jld2"), readdir(dir))
    isempty(matches) && return nothing
    sort!(matches)
    return joinpath(dir, last(matches))
end

function saved_dataset_path(saved, dataset::String)
    if haskey(saved, "spec")
        spec = saved["spec"]
        dataset_path = maybe_get(spec, :dataset_path)
        if !isnothing(dataset_path) && !isempty(String(dataset_path))
            return String(dataset_path)
        end
    end
    if haskey(saved, "raw_spec")
        raw_spec = saved["raw_spec"]
        params = maybe_get(raw_spec, :params)
        dataset_path = isnothing(params) ? nothing : maybe_get(params, :dataset_path)
        if !isnothing(dataset_path)
            return String(dataset_path)
        end
    end
    return get(DEFAULT_DATASET_PATHS, dataset) do
        error("No dataset path available for dataset=$dataset")
    end
end

function saved_column(saved)
    if haskey(saved, "spec")
        spec = saved["spec"]
        column = maybe_get(spec, :column)
        return something(column, nothing)
    end
    if haskey(saved, "raw_spec")
        raw_spec = saved["raw_spec"]
        params = maybe_get(raw_spec, :params)
        return isnothing(params) ? nothing : maybe_get(params, :column)
    end
    return nothing
end

function rebuild_test_targets(dataset::String, pred_type::Symbol, horizon::Int, dataset_path::String, seq_len::Int, split)
    Xmat, feat_cols = ProbabilisticEnsembling.load_ett(dataset_path)
    X3, Y2 = ProbabilisticEnsembling.make_sequences(Xmat; seq_len = seq_len, horizon = horizon)
    Xtr, _, _, _, _, Yte =
        ProbabilisticEnsembling.train_val_test_split(X3, Y2; ratios = split)
    scaler = ProbabilisticEnsembling.fit_scaler(Xtr)
    Yte_scaled = ProbabilisticEnsembling.scale_targets(scaler, Yte)
    return (Yte = Float64.(Yte), Yte_scaled = Float64.(Yte_scaled), scaler = scaler, feat_cols = feat_cols)
end

function extract_target_arrays(saved, dataset::String, pred_type::Symbol, horizon::Int, options)
    rebuilt = rebuild_test_targets(
        dataset,
        pred_type,
        horizon,
        saved_dataset_path(saved, dataset),
        options["seq_len"],
        options["split"],
    )

    column_name = saved_column(saved)
    if pred_type == :univariate
        column_name === nothing && error("No target column stored for dataset=$dataset horizon=$horizon")
        col_idx = ProbabilisticEnsembling.find_column_index(rebuilt.feat_cols, String(column_name))
        y_scaled = vec(rebuilt.Yte_scaled[col_idx, :])
        y_original = vec(rebuilt.Yte[col_idx, :])
        μ = Float64(rebuilt.scaler.μ[col_idx])
        σ = Float64(rebuilt.scaler.σ[col_idx])
        return (y_scaled = y_scaled, y_original = y_original, μ = μ, σ = σ, col_idx = col_idx)
    end

    μ = Float64.(rebuilt.scaler.μ)
    σ = Float64.(rebuilt.scaler.σ)
    return (
        y_scaled = rebuilt.Yte_scaled,
        y_original = rebuilt.Yte,
        μ = μ,
        σ = σ,
        col_idx = nothing,
    )
end

function extract_mean_std(saved, pred_type::Symbol, col_idx)
    mean_saved = saved["ensemble_mean"]
    std_saved = saved["ensemble_std"]

    if pred_type == :univariate
        if ndims(mean_saved) == 1
            return (mean = Float64.(mean_saved), std = Float64.(std_saved))
        end
        isnothing(col_idx) && error("col_idx is required for matrix-valued univariate outputs")
        return (
            mean = vec(Float64.(mean_saved[col_idx, :])),
            std = vec(Float64.(std_saved[col_idx, :])),
        )
    end

    return (mean = Float64.(mean_saved), std = Float64.(std_saved))
end

pinball_loss(y, qhat, q) = mean(max.(q .* (y .- qhat), (q - 1.0) .* (y .- qhat)))

function calibration_gap(y, qhat, q)
    return mean(y .<= qhat) - q
end

function quantile_predictions(mean_pred, std_pred, quantile_level)
    z = quantile(Normal(), quantile_level)
    safe_std = max.(std_pred, eps(Float64))
    return mean_pred .+ z .* safe_std
end

function original_scale_predictions(qhat_scaled, μ, σ, pred_type::Symbol)
    if pred_type == :univariate
        return qhat_scaled .* σ .+ μ
    end
    return qhat_scaled .* reshape(σ, :, 1) .+ reshape(μ, :, 1)
end

function model_label(model_type::String)
    labels = Dict(
        "static" => "Static",
        "dynamic" => "Dynamic",
        "dynamic_diagonal" => "Dynamic Diagonal",
        "noisy_experts" => "Noisy Experts",
        "noisy_experts_diagonal" => "Noisy Experts Diagonal",
        "neural_ensemble" => "MoE",
        "neural_ensemble_big" => "MoE Big",
    )
    return get(labels, model_type, model_type)
end

function build_quantile_rows(options)
    rows = NamedTuple[]
    selected_datasets = Set(String.(options["datasets"]))
    selected_models = Set(String.(options["models"]))
    selected_horizons = Set(Int.(options["horizons"]))
    quantiles = Float64.(options["quantiles"])

    for (dataset, pred_type) in DATASETS
        dataset in selected_datasets || continue
        for horizon in DEFAULT_HORIZONS
            horizon in selected_horizons || continue
            for model_type in DEFAULT_MODEL_TYPES
                model_type in selected_models || continue

                path = resolve_result_path(options["results_dir"], dataset, horizon, pred_type, model_type)
                if isnothing(path)
                    @warn "Skipping missing result" dataset horizon model_type
                    continue
                end

                saved = JLD2.load(path)
                target_data = extract_target_arrays(saved, dataset, pred_type, horizon, options)
                preds = extract_mean_std(saved, pred_type, target_data.col_idx)

                for q in quantiles
                    qhat_scaled = quantile_predictions(preds.mean, preds.std, q)
                    if options["original_scale"]
                        qhat = original_scale_predictions(qhat_scaled, target_data.μ, target_data.σ, pred_type)
                        y = target_data.y_original
                        scale_label = "original"
                    else
                        qhat = qhat_scaled
                        y = target_data.y_scaled
                        scale_label = "scaled"
                    end

                    push!(
                        rows,
                        (
                            dataset = dataset,
                            horizon = horizon,
                            model_type = model_type,
                            model_label = model_label(model_type),
                            quantile = q,
                            pinball_loss = pinball_loss(y, qhat, q),
                            calibration_gap = calibration_gap(y, qhat, q),
                            empirical_coverage = mean(y .<= qhat),
                            n_targets = length(y),
                            result_path = path,
                            scale = scale_label,
                        ),
                    )
                end
            end
        end
    end

    return DataFrame(rows)
end

function build_average_rows(df::DataFrame)
    return combine(
        groupby(df, [:dataset, :model_type, :model_label, :quantile, :scale]),
        :pinball_loss => mean => :avg_pinball_loss,
        :calibration_gap => mean => :avg_calibration_gap,
        :empirical_coverage => mean => :avg_empirical_coverage,
        nrow => :n_rows,
    )
end

function main(args)
    options = parse_args(args)
    mkpath(options["output_dir"])

    df = build_quantile_rows(options)
    isempty(df) && error("No quantile rows were produced. Check filters and result paths.")

    sort!(df, [:dataset, :horizon, :model_type, :quantile])
    avg_df = build_average_rows(df)
    sort!(avg_df, [:dataset, :model_type, :quantile])

    quantile_tag = join(string.(round.(Int, 100 .* options["quantiles"])), "-")
    scale_tag = options["original_scale"] ? "original" : "scaled"
    detail_path = joinpath(
        options["output_dir"],
        "quantile_performance_$(scale_tag)_q$(quantile_tag).csv",
    )
    average_path = joinpath(
        options["output_dir"],
        "quantile_performance_average_$(scale_tag)_q$(quantile_tag).csv",
    )

    CSV.write(detail_path, df)
    CSV.write(average_path, avg_df)

    println("Saved detailed quantile metrics to: $detail_path")
    println("Saved averaged quantile metrics to: $average_path")
end

main(ARGS)
