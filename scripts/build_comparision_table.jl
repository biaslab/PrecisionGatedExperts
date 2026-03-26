using JLD2
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results_vae")
const HORIZONS = [96, 192, 336, 720]
const SELECTED_METRICS = [:mse, :nll]
const ENSEMBLE_MODEL_TYPES = ["static", "dynamic", "dynamic_diagonal", "noisy_experts", "noisy_experts_diagonal", "neural_ensemble", "neural_ensemble_big"]
const ENSEMBLE_MODEL_LABELS = Dict(
    "static" => "Static",
    "dynamic" => "Dyn.",
    "dynamic_diagonal" => "Dyn. Diag.",
    "noisy_experts" => "Noisy",
    "noisy_experts_diagonal" => "Noisy Diag.",
    "neural_ensemble" => "MoE",
    "neural_ensemble_big" => "MoE Big",
)
# Map model type to its actual directory name under results_vae/
const MODEL_DIR = Dict(
    "noisy_experts_diagonal" => "noisy_diagonal",
    "noisy_experts" => "noisy_experts",
)
const BASELINE_MODELS = ["CNN", "DLinear", "LSTM", "MLP", "NConv"]

const BASELINE_METRICS = Dict(
    "exchange_rate" => Dict(
        96 => Dict("CNN" => (mse = 2.455, mae = 1.187), "DLinear" => (mse = 0.157, mae = 0.306), "LSTM" => (mse = 1.150, mae = 0.881), "MLP" => (mse = 0.183, mae = 0.317), "NConv" => (mse = 0.180, mae = 0.314)),
        192 => Dict("CNN" => (mse = 3.559, mae = 1.448), "DLinear" => (mse = 0.325, mae = 0.446), "LSTM" => (mse = 2.589, mae = 1.337), "MLP" => (mse = 0.400, mae = 0.478), "NConv" => (mse = 0.410, mae = 0.479)),
        336 => Dict("CNN" => (mse = 2.295, mae = 1.184), "DLinear" => (mse = 0.558, mae = 0.606), "LSTM" => (mse = 1.536, mae = 1.023), "MLP" => (mse = 0.804, mae = 0.680), "NConv" => (mse = 0.820, mae = 0.680)),
        720 => Dict("CNN" => (mse = 4.902, mae = 1.706), "DLinear" => (mse = 1.495, mae = 0.984), "LSTM" => (mse = 1.982, mae = 1.147), "MLP" => (mse = 2.187, mae = 1.166), "NConv" => (mse = 2.426, mae = 1.225)),
    ),
    "ETTh1" => Dict(
        96 => Dict("CNN" => (mse = 0.456272, mae = 0.586459), "DLinear" => (mse = 0.156445, mae = 0.312482), "LSTM" => (mse = 0.411891, mae = 0.535926), "MLP" => (mse = 0.147123, mae = 0.300812), "NConv" => (mse = 0.180725, mae = 0.333047)),
        192 => Dict("CNN" => (mse = 0.248275, mae = 0.398232), "DLinear" => (mse = 0.161983, mae = 0.318343), "LSTM" => (mse = 0.278308, mae = 0.419464), "MLP" => (mse = 0.136831, mae = 0.295188), "NConv" => (mse = 0.177978, mae = 0.323494)),
        336 => Dict("CNN" => (mse = 0.297750, mae = 0.448504), "DLinear" => (mse = 0.137354, mae = 0.308821), "LSTM" => (mse = 0.230511, mae = 0.377467), "MLP" => (mse = 0.135473, mae = 0.291139), "NConv" => (mse = 0.173684, mae = 0.328243)),
        720 => Dict("CNN" => (mse = 1.006215, mae = 0.904852), "DLinear" => (mse = 0.290496, mae = 0.461657), "LSTM" => (mse = 0.275422, mae = 0.409312), "MLP" => (mse = 0.215503, mae = 0.361101), "NConv" => (mse = 0.309745, mae = 0.439577)),
    ),
    "ETTh2" => Dict(
        96 => Dict("CNN" => (mse = 1.795751, mae = 1.114136), "DLinear" => (mse = 0.312674, mae = 0.448080), "LSTM" => (mse = 0.850844, mae = 0.751669), "MLP" => (mse = 0.338792, mae = 0.464811), "NConv" => (mse = 0.426582, mae = 0.504736)),
        192 => Dict("CNN" => (mse = 1.016528, mae = 0.771749), "DLinear" => (mse = 0.269895, mae = 0.414489), "LSTM" => (mse = 0.854724, mae = 0.751242), "MLP" => (mse = 0.319791, mae = 0.454525), "NConv" => (mse = 0.413613, mae = 0.503851)),
        336 => Dict("CNN" => (mse = 1.109364, mae = 0.853351), "DLinear" => (mse = 0.269398, mae = 0.413311), "LSTM" => (mse = 0.848074, mae = 0.715418), "MLP" => (mse = 0.344999, mae = 0.470280), "NConv" => (mse = 0.423463, mae = 0.507681)),
        720 => Dict("CNN" => (mse = 1.956659, mae = 1.227688), "DLinear" => (mse = 0.384535, mae = 0.515972), "LSTM" => (mse = 1.232485, mae = 0.966198), "MLP" => (mse = 0.680617, mae = 0.660704), "NConv" => (mse = 0.684699, mae = 0.655226)),
    ),
    "electricity" => Dict(
        96 => Dict("CNN" => (mse = 0.350, mae = 0.432), "DLinear" => (mse = 0.176, mae = 0.267), "LSTM" => (mse = 0.396, mae = 0.443), "MLP" => (mse = 0.178, mae = 0.266), "NConv" => (mse = 0.348, mae = 0.363)),
        192 => Dict("CNN" => (mse = 0.388, mae = 0.462), "DLinear" => (mse = 0.251, mae = 0.326), "LSTM" => (mse = 0.335, mae = 0.403), "MLP" => (mse = 0.260, mae = 0.324), "NConv" => (mse = 0.307, mae = 0.345)),
        336 => Dict("CNN" => (mse = 0.364, mae = 0.434), "DLinear" => (mse = 0.223, mae = 0.314), "LSTM" => (mse = 0.337, mae = 0.407), "MLP" => (mse = 0.229, mae = 0.310), "NConv" => (mse = 0.248, mae = 0.319)),
        720 => Dict("CNN" => (mse = 0.418, mae = 0.462), "DLinear" => (mse = 0.344, mae = 0.404), "LSTM" => (mse = 0.425, mae = 0.462), "MLP" => (mse = 0.370, mae = 0.407), "NConv" => (mse = 0.462, mae = 0.449)),
    ),
    "traffic" => Dict(
        96 => Dict("CNN" => (mse = 0.743, mae = 0.422), "DLinear" => (mse = 0.475, mae = 0.308), "LSTM" => (mse = 0.845, mae = 0.469), "MLP" => (mse = 0.478, mae = 0.305), "NConv" => (mse = 1.205, mae = 0.596)),
        192 => Dict("CNN" => (mse = 0.737, mae = 0.423), "DLinear" => (mse = 0.710, mae = 0.431), "LSTM" => (mse = 0.817, mae = 0.443), "MLP" => (mse = 0.716, mae = 0.430), "NConv" => (mse = 0.831, mae = 0.453)),
        336 => Dict("CNN" => (mse = 0.730, mae = 0.403), "DLinear" => (mse = 0.547, mae = 0.345), "LSTM" => (mse = 0.740, mae = 0.407), "MLP" => (mse = 0.556, mae = 0.337), "NConv" => (mse = 0.622, mae = 0.344)),
        720 => Dict("CNN" => (mse = 0.800, mae = 0.447), "DLinear" => (mse = 0.814, mae = 0.473), "LSTM" => (mse = 1.055, mae = 0.576), "MLP" => (mse = 0.820, mae = 0.466), "NConv" => (mse = 1.191, mae = 0.588)),
    ),
)

# Dataset name => (display name, prediction_type)
# univariate files: {dataset}_h{H}_OT_{model}.jld2
# multivariate files: {dataset}_h{H}_multivariate_{model}.jld2
const DATASETS = [
    ("ETTh1", "ETTh1", :univariate),
    ("ETTh2", "ETTh2", :univariate),
    ("exchange_rate", "Exchange rate", :multivariate),
    ("electricity", "Electricity", :multivariate),
    ("traffic", "Traffic", :multivariate),
]

function result_filename(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    if startswith(model_type, "neural_ensemble")
        return "$(dataset)_h$(horizon)_neural_ensemble.jld2"
    elseif pred_type == :univariate
        return "$(dataset)_h$(horizon)_OT_$(model_type).jld2"
    else
        return "$(dataset)_h$(horizon)_multivariate_$(model_type).jld2"
    end
end

function load_metrics(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    fname = result_filename(dataset, horizon, pred_type, model_type)
    dir_name = get(MODEL_DIR, model_type, model_type)
    fpath = joinpath(RESULTS_DIR, dir_name, fname)
    if !isfile(fpath)
        # Fall back to prefix-matching (handles files with hash suffixes)
        dir = joinpath(RESULTS_DIR, dir_name)
        !isdir(dir) && return nothing
        prefix = replace(fname, ".jld2" => "")
        matches = filter(f -> startswith(f, prefix) && endswith(f, ".jld2"), readdir(dir))
        isempty(matches) && return nothing
        fpath = joinpath(dir, first(matches))
    end
    data = JLD2.load(fpath)
    metrics = data["ensemble_metrics"]
    return Dict(
        :mse => clean_metric(get_metric_value(metrics, :mse)),
        :mae => clean_metric(get_metric_value(metrics, :mae)),
        :nll => clean_metric(negate_if_present(get_metric_value(metrics, :nll))),
        :r2 => clean_metric(get_metric_value(metrics, :r2)),
        :smape => clean_metric(get_metric_value(metrics, :smape)),
        :ci95_interval_score => clean_metric(get_metric_value(metrics, :ci95_interval_score)),
    )
end

function best_baseline_metrics(dataset::String, horizon::Int)
    by_dataset = get(BASELINE_METRICS, dataset, nothing)
    by_dataset === nothing && return nothing
    by_horizon = get(by_dataset, horizon, nothing)
    by_horizon === nothing && return nothing

    best_model = nothing
    best_metrics = nothing
    for model in BASELINE_MODELS
        metrics = get(by_horizon, model, nothing)
        metrics === nothing && continue
        if best_metrics === nothing || metrics.mse < best_metrics.mse
            best_model = model
            best_metrics = metrics
        end
    end

    best_metrics === nothing && return nothing
    return Dict(
        :model => best_model,
        :mse => clean_metric(best_metrics.mse),
        :mae => clean_metric(best_metrics.mae),
        :nll => nothing,
        :r2 => nothing,
        :smape => nothing,
        :ci95_interval_score => nothing,
    )
end

clean_metric(val) = (val === nothing || !isfinite(val)) ? nothing : val
negate_if_present(val) = val === nothing ? nothing : -val
get_metric_value(metrics, key::Symbol) = hasproperty(metrics, key) ? getproperty(metrics, key) : nothing

function fmt(val)
    if val === nothing
        return "\$\\times\$"
    end
    abs_val = abs(val)
    if abs_val == 0.0
        return "0.000"
    elseif abs_val >= 1000.0
        exp = floor(Int, log10(abs_val))
        coeff = val / 10.0^exp
        return "\$" * @sprintf("%.3f", coeff) * "\\!\\times\\!10^{$exp}\$"
    elseif abs_val >= 100.0
        return @sprintf("%.1f", val)
    elseif abs_val >= 10.0
        return @sprintf("%.2f", val)
    else
        return @sprintf("%.3f", val)
    end
end

function best_and_second(vals, larger_is_better::Bool)
    unique_sorted = sort(unique(filter(isfinite, vals)); rev = larger_is_better)
    isempty(unique_sorted) && return (nothing, nothing)
    return (unique_sorted[1], length(unique_sorted) >= 2 ? unique_sorted[2] : nothing)
end

function highlight_metric(val, best_val, second_val)
    rendered = fmt(val)
    if val !== nothing && best_val !== nothing && val == best_val
        return "\\textbf{$rendered}"
    elseif val !== nothing && second_val !== nothing && val == second_val
        return "\\underline{\\textcolor{blue}{$rendered}}"
    end
    return rendered
end

function selected_metric_specs()
    metric_catalog = Dict(
        :mse => (label = "MSE", larger_is_better = false),
        :mae => (label = "MAE", larger_is_better = false),
        :nll => (label = "NLL", larger_is_better = false),
        :r2 => (label = "R2", larger_is_better = true),
        :smape => (label = "sMAPE", larger_is_better = false),
        :ci95_interval_score => (label = "CI95 Int. Score", larger_is_better = false),
    )
    isempty(SELECTED_METRICS) && error("SELECTED_METRICS must contain at least one metric.")

    specs = NamedTuple{(:key,:label,:larger_is_better),Tuple{Symbol,String,Bool}}[]
    for metric in SELECTED_METRICS
        haskey(metric_catalog, metric) || error("Unsupported metric in SELECTED_METRICS: $(metric). Supported metrics: $(collect(keys(metric_catalog))).")
        push!(specs, (key = metric, label = metric_catalog[metric].label, larger_is_better = metric_catalog[metric].larger_is_better))
    end
    return specs
end

function build_latex_table()
    table_models = vcat(ENSEMBLE_MODEL_TYPES, ["best_baseline"])
    table_model_labels = merge(copy(ENSEMBLE_MODEL_LABELS), Dict("best_baseline" => "Best"))
    metric_names = selected_metric_specs()
    horizon_groups = vcat(string.(HORIZONS), ["Avg"])

    metric_dict_type = Dict{Symbol,Union{Nothing,Float64}}
    all_data = Dict{Tuple{String,Int,String}, Union{Nothing,metric_dict_type}}()
    for (ds, _, pred_type) in DATASETS
        for h in HORIZONS
            for mt in ENSEMBLE_MODEL_TYPES
                metrics = load_metrics(ds, h, pred_type, mt)
                if metrics === nothing
                    all_data[(ds, h, mt)] = nothing
                else
                    all_data[(ds, h, mt)] = metrics
                end
            end
        end
    end

    col_spec = "lll" * repeat(" c", length(table_models))
    header_models = join([table_model_labels[mt] for mt in table_models], " & ")

    lines = String[]
    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\scriptsize")
    push!(lines, "\\setlength{\\tabcolsep}{2.5pt}")
    push!(lines, "\\caption{MSE / MAE / NLL for Static, Dynamic (Dyn.), Dynamic Diagonal (Dyn. Diag.), Noisy Experts (Noisy), Noisy Diagonal (Noisy Diag.), Mixture of Experts (MoE), and MoE Big ensembles, compared against the best baseline model selected by lowest baseline MSE for each dataset and horizon. Rows are organized by dataset, horizon, and metric; models remain as columns. An average block is included for each dataset. \\textbf{Bold} indicates the best result; {\\underline{\\textcolor{blue}{blue underlined}}} indicates the second best. Non-finite values are treated as missing.}")
    push!(lines, "\\label{tab:ensemble_comparison}")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")
    push!(lines, "Dataset & H & Metric & $header_models \\\\")
    push!(lines, "\\midrule")

    for (di, (ds, display_name, _)) in enumerate(DATASETS)
        dataset_row_span = length(horizon_groups) * length(metric_names)
        for (hi, horizon_label) in enumerate(horizon_groups)
            per_model_metrics = Dict{String,metric_dict_type}()
            for mt in ENSEMBLE_MODEL_TYPES
                if horizon_label == "Avg"
                    model_metrics = [
                        all_data[(ds, h, mt)] for h in HORIZONS
                        if all_data[(ds, h, mt)] !== nothing
                    ]
                    per_model_metrics[mt] = Dict(
                        metric.key => let vals = [m[metric.key] for m in model_metrics if get(m, metric.key, nothing) !== nothing]
                            isempty(vals) ? nothing : sum(vals) / length(vals)
                        end for metric in metric_names
                    )
                else
                    h = parse(Int, horizon_label)
                    metrics = all_data[(ds, h, mt)]
                    per_model_metrics[mt] = metrics === nothing ? Dict(metric.key => nothing for metric in metric_names) : Dict(metric.key => get(metrics, metric.key, nothing) for metric in metric_names)
                end
            end
            if horizon_label == "Avg"
                baseline_metrics = [
                    best_baseline_metrics(ds, h) for h in HORIZONS
                    if best_baseline_metrics(ds, h) !== nothing
                ]
                per_model_metrics["best_baseline"] = Dict(
                    metric.key => let vals = [m[metric.key] for m in baseline_metrics if get(m, metric.key, nothing) !== nothing]
                        isempty(vals) ? nothing : sum(vals) / length(vals)
                    end for metric in metric_names
                )
            else
                h = parse(Int, horizon_label)
                baseline = best_baseline_metrics(ds, h)
                per_model_metrics["best_baseline"] = baseline === nothing ? Dict(metric.key => nothing for metric in metric_names) : Dict(metric.key => get(baseline, metric.key, nothing) for metric in metric_names)
            end

            for (mi, metric) in enumerate(metric_names)
                values = [per_model_metrics[mt][metric.key] for mt in table_models if per_model_metrics[mt][metric.key] !== nothing]
                best_val, second_val = best_and_second(values, metric.larger_is_better)

                dataset_cell = (hi == 1 && mi == 1) ? "\\multirow{$dataset_row_span}{*}{$display_name}" : ""
                horizon_cell = mi == 1 ? "\\multirow{$(length(metric_names))}{*}{$horizon_label}" : ""

                row_cells = String[]
                for mt in table_models
                    model_metrics = per_model_metrics[mt]
                    push!(row_cells, highlight_metric(model_metrics[metric.key], best_val, second_val))
                end

                push!(lines, "$dataset_cell & $horizon_cell & $(metric.label) & $(join(row_cells, " & ")) \\\\")
            end

            if hi < length(horizon_groups)
                push!(lines, "\\cmidrule(lr){2-$(3 + length(table_models))}")
            end
        end

        if di < length(DATASETS)
            push!(lines, "\\midrule")
        end
    end

    push!(lines, "\\bottomrule")
    push!(lines, "\\end{tabular}")
    push!(lines, "\\end{table}")

    return join(lines, "\n")
end

latex = build_latex_table()
println(latex)
