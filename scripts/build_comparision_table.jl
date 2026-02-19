using JLD2
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "..", "final_results")
const HORIZONS = [96, 192, 336, 720]
const MODEL_TYPES = ["static", "dynamic", "hierarchical"]
const MODEL_LABELS = Dict(
    "static" => "Static",
    "dynamic" => "Dyn.",
    "hierarchical" => "Hier.",
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
    if pred_type == :univariate
        return "$(dataset)_h$(horizon)_OT_$(model_type).jld2"
    else
        return "$(dataset)_h$(horizon)_multivariate_$(model_type).jld2"
    end
end

function load_metrics(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    fname = result_filename(dataset, horizon, pred_type, model_type)
    fpath = joinpath(RESULTS_DIR, fname)
    if !isfile(fpath)
        return nothing
    end
    data = JLD2.load(fpath)
    metrics = data["ensemble_metrics"]
    return (mse = metrics.mse, mae = metrics.mae)
end

function fmt(val)
    if val === nothing
        return "\$\\times\$"
    end
    return @sprintf("%.3f", val)
end

function build_latex_table()
    n_models = length(MODEL_TYPES)
    # Collect all data first to identify best values per row
    all_data = Dict{Tuple{String,Int,String}, Union{Nothing,NamedTuple{(:mse,:mae), Tuple{Float64,Float64}}}}()
    for (ds, _, pred_type) in DATASETS
        for h in HORIZONS
            for mt in MODEL_TYPES
                all_data[(ds, h, mt)] = load_metrics(ds, h, pred_type, mt)
            end
        end
    end

    col_spec = "ll" * repeat(" cc", n_models)
    cmidrules = join(
        ["\\cmidrule(lr){$(2i+1)-$(2i+2)}" for i in 1:n_models],
    )

    header_models = join(
        ["\\multicolumn{2}{c}{$(MODEL_LABELS[mt])}" for mt in MODEL_TYPES],
        " &\n",
    )

    subheader = join(["MSE & MAE" for _ in MODEL_TYPES], " & ")

    lines = String[]
    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\scriptsize")
    push!(lines, "\\setlength{\\tabcolsep}{2.5pt}")
    push!(lines, "\\caption{MSE / MAE for Static, Dynamic (Dyn.), and Hierarchical (Hier.) ensembles.}")
    push!(lines, "\\label{tab:ensemble_comparison}")
    push!(lines, "\\begin{tabular}{$col_spec}")
    push!(lines, "\\toprule")
    push!(lines, "\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{H} &")
    push!(lines, "$header_models \\\\")
    push!(lines, "$cmidrules")
    push!(lines, "& & $subheader \\\\")
    push!(lines, "\\midrule")

    for (di, (ds, display_name, pred_type)) in enumerate(DATASETS)
        push!(lines, "\\multirow{$(length(HORIZONS) + 1)}{*}{$display_name}")

        # Collect metrics for avg computation
        avg_mse = Dict(mt => Float64[] for mt in MODEL_TYPES)
        avg_mae = Dict(mt => Float64[] for mt in MODEL_TYPES)

        for h in HORIZONS
            row_mse = Dict{String,Union{Nothing,Float64}}()
            row_mae = Dict{String,Union{Nothing,Float64}}()
            for mt in MODEL_TYPES
                m = all_data[(ds, h, mt)]
                row_mse[mt] = m === nothing ? nothing : m.mse
                row_mae[mt] = m === nothing ? nothing : m.mae
                if m !== nothing
                    push!(avg_mse[mt], m.mse)
                    push!(avg_mae[mt], m.mae)
                end
            end

            # Find best (min) MSE and MAE across models for this row
            valid_mse = [v for v in values(row_mse) if v !== nothing]
            valid_mae = [v for v in values(row_mae) if v !== nothing]
            best_mse = isempty(valid_mse) ? nothing : minimum(valid_mse)
            best_mae = isempty(valid_mae) ? nothing : minimum(valid_mae)

            cells = String[]
            for mt in MODEL_TYPES
                mse_str = fmt(row_mse[mt])
                mae_str = fmt(row_mae[mt])
                if row_mse[mt] !== nothing && row_mse[mt] == best_mse
                    mse_str = "\\textbf{$mse_str}"
                end
                if row_mae[mt] !== nothing && row_mae[mt] == best_mae
                    mae_str = "\\textbf{$mae_str}"
                end
                push!(cells, "$mse_str & $mae_str")
            end

            push!(lines, "& $h  & $(join(cells, " & ")) \\\\")
        end

        # Avg row
        avg_cells = String[]
        avg_mse_vals = Dict{String,Union{Nothing,Float64}}()
        avg_mae_vals = Dict{String,Union{Nothing,Float64}}()
        for mt in MODEL_TYPES
            if isempty(avg_mse[mt])
                avg_mse_vals[mt] = nothing
                avg_mae_vals[mt] = nothing
            else
                avg_mse_vals[mt] = sum(avg_mse[mt]) / length(avg_mse[mt])
                avg_mae_vals[mt] = sum(avg_mae[mt]) / length(avg_mae[mt])
            end
        end

        valid_avg_mse = [v for v in values(avg_mse_vals) if v !== nothing]
        valid_avg_mae = [v for v in values(avg_mae_vals) if v !== nothing]
        best_avg_mse = isempty(valid_avg_mse) ? nothing : minimum(valid_avg_mse)
        best_avg_mae = isempty(valid_avg_mae) ? nothing : minimum(valid_avg_mae)

        for mt in MODEL_TYPES
            mse_str = fmt(avg_mse_vals[mt])
            mae_str = fmt(avg_mae_vals[mt])
            if avg_mse_vals[mt] !== nothing && avg_mse_vals[mt] == best_avg_mse
                mse_str = "\\textbf{$mse_str}"
            end
            if avg_mae_vals[mt] !== nothing && avg_mae_vals[mt] == best_avg_mae
                mae_str = "\\textbf{$mae_str}"
            end
            push!(avg_cells, "$mse_str & $mae_str")
        end

        push!(lines, "& Avg & $(join(avg_cells, " & ")) \\\\")

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
