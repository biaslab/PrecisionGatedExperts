using JLD2
using Printf

const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results")
const HORIZONS = [96, 192, 336, 720]
const ENSEMBLE_MODEL_TYPES = ["static", "dynamic", "neural_ensemble"]
const ENSEMBLE_MODEL_LABELS = Dict(
    "static" => "Static",
    "dynamic" => "Dyn.",
    "neural_ensemble" => "MoE",
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
        96 => Dict("CNN" => (mse = 0.900, mae = 0.734), "DLinear" => (mse = 0.504, mae = 0.484), "LSTM" => (mse = 0.762, mae = 0.646), "MLP" => (mse = 0.516, mae = 0.493), "NConv" => (mse = 0.665, mae = 0.547)),
        192 => Dict("CNN" => (mse = 1.594, mae = 0.966), "DLinear" => (mse = 0.553, mae = 0.523), "LSTM" => (mse = 0.964, mae = 0.725), "MLP" => (mse = 0.578, mae = 0.528), "NConv" => (mse = 0.779, mae = 0.610)),
        336 => Dict("CNN" => (mse = 1.717, mae = 0.958), "DLinear" => (mse = 0.611, mae = 0.560), "LSTM" => (mse = 1.189, mae = 0.812), "MLP" => (mse = 0.643, mae = 0.564), "NConv" => (mse = 0.776, mae = 0.625)),
        720 => Dict("CNN" => (mse = 1.186, mae = 0.846), "DLinear" => (mse = 0.794, mae = 0.688), "LSTM" => (mse = 1.204, mae = 0.832), "MLP" => (mse = 0.910, mae = 0.724), "NConv" => (mse = 1.084, mae = 0.787)),
    ),
    "ETTh2" => Dict(
        96 => Dict("CNN" => (mse = 2.111, mae = 0.955), "DLinear" => (mse = 0.293, mae = 0.368), "LSTM" => (mse = 0.987, mae = 0.774), "MLP" => (mse = 0.300, mae = 0.364), "NConv" => (mse = 0.336, mae = 0.388)),
        192 => Dict("CNN" => (mse = 2.274, mae = 1.099), "DLinear" => (mse = 0.345, mae = 0.410), "LSTM" => (mse = 1.347, mae = 0.792), "MLP" => (mse = 0.354, mae = 0.401), "NConv" => (mse = 0.417, mae = 0.434)),
        336 => Dict("CNN" => (mse = 2.359, mae = 1.198), "DLinear" => (mse = 0.428, mae = 0.473), "LSTM" => (mse = 1.517, mae = 0.952), "MLP" => (mse = 0.421, mae = 0.445), "NConv" => (mse = 0.479, mae = 0.465)),
        720 => Dict("CNN" => (mse = 1.826, mae = 1.052), "DLinear" => (mse = 0.639, mae = 0.585), "LSTM" => (mse = 1.155, mae = 0.884), "MLP" => (mse = 0.695, mae = 0.578), "NConv" => (mse = 0.759, mae = 0.596)),
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
    if pred_type == :univariate
        return "$(dataset)_h$(horizon)_OT_$(model_type).jld2"
    else
        return "$(dataset)_h$(horizon)_multivariate_$(model_type).jld2"
    end
end

function load_metrics(dataset::String, horizon::Int, pred_type::Symbol, model_type::String)
    if model_type == "neural_ensemble"
        return load_neural_ensemble_metrics(dataset, horizon, pred_type)
    end
    fname = result_filename(dataset, horizon, pred_type, model_type)
    fpath = joinpath(RESULTS_DIR, model_type, fname)
    if !isfile(fpath)
        return nothing
    end
    data = JLD2.load(fpath)
    metrics = data["ensemble_metrics"]
    return (mse = metrics.mse, mae = metrics.mae, nll = -metrics.nll)
end

function load_neural_ensemble_metrics(dataset::String, horizon::Int, pred_type::Symbol)
    dir = joinpath(RESULTS_DIR, "neural_ensemble")
    !isdir(dir) && return nothing
    prefix = "$(dataset)_h$(horizon)_neural_ensemble"
    matches = filter(f -> startswith(f, prefix) && endswith(f, ".jld2"), readdir(dir))
    isempty(matches) && return nothing
    fpath = joinpath(dir, first(matches))
    data = JLD2.load(fpath)
    metrics = data["ensemble_metrics"]
    return (mse = metrics.mse, mae = metrics.mae, nll = -metrics.nll)
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
    return (model = best_model, mse = best_metrics.mse, mae = best_metrics.mae, nll = nothing)
end

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

function build_latex_table()
    n_ensemble_models = length(ENSEMBLE_MODEL_TYPES)
    # Collect all data first to identify best values per row
    all_data = Dict{Tuple{String,Int,String}, Union{Nothing,NamedTuple{(:mse,:mae,:nll), Tuple{Float64,Float64,Float64}}}}()
    for (ds, _, pred_type) in DATASETS
        for h in HORIZONS
            for mt in ENSEMBLE_MODEL_TYPES
                all_data[(ds, h, mt)] = load_metrics(ds, h, pred_type, mt)
            end
        end
    end

    col_spec = "ll" * join([" ccc" for _ in ENSEMBLE_MODEL_TYPES], " @{\\hskip 6pt}") * " @{\\hskip 6pt} ccc"
    cmidrules = join(
        vcat(
            ["\\cmidrule(lr){$(3i)-$(3i+2)}" for i in 1:n_ensemble_models],
            ["\\cmidrule(lr){$(3n_ensemble_models+3)-$(3n_ensemble_models+5)}"],
        ),
        " ",
    )

    header_models = join(
        vcat(
            ["\\multicolumn{3}{c}{$(ENSEMBLE_MODEL_LABELS[mt])}" for mt in ENSEMBLE_MODEL_TYPES],
            ["\\multicolumn{3}{c}{Best}"],
        ),
        " & ",
    )

    subheader = join(
        vcat(
            ["MSE & MAE & NLL" for _ in ENSEMBLE_MODEL_TYPES],
            ["MSE & MAE & Model"],
        ),
        " & ",
    )

    lines = String[]
    push!(lines, "\\begin{table}[t]")
    push!(lines, "\\centering")
    push!(lines, "\\scriptsize")
    push!(lines, "\\setlength{\\tabcolsep}{2.5pt}")
    push!(lines, "\\caption{MSE / MAE / NLL for Static, Dynamic (Dyn.), and Mixture of Experts (MoE) ensembles, compared against the best baseline model selected by lowest baseline MSE for each dataset and horizon.}")
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
        avg_mse = Dict(mt => Float64[] for mt in ENSEMBLE_MODEL_TYPES)
        avg_mae = Dict(mt => Float64[] for mt in ENSEMBLE_MODEL_TYPES)
        avg_nll = Dict(mt => Float64[] for mt in ENSEMBLE_MODEL_TYPES)
        best_avg_mse = Float64[]
        best_avg_mae = Float64[]
        best_avg_models = String[]

        for h in HORIZONS
            row_mse = Dict{String,Union{Nothing,Float64}}()
            row_mae = Dict{String,Union{Nothing,Float64}}()
            row_nll = Dict{String,Union{Nothing,Float64}}()
            for mt in ENSEMBLE_MODEL_TYPES
                m = all_data[(ds, h, mt)]
                row_mse[mt] = m === nothing ? nothing : m.mse
                row_mae[mt] = m === nothing ? nothing : m.mae
                row_nll[mt] = m === nothing ? nothing : m.nll
                if m !== nothing
                    push!(avg_mse[mt], m.mse)
                    push!(avg_mae[mt], m.mae)
                    push!(avg_nll[mt], m.nll)
                end
            end

            best_baseline = best_baseline_metrics(ds, h)
            best_baseline_mse = best_baseline === nothing ? nothing : best_baseline.mse
            best_baseline_mae = best_baseline === nothing ? nothing : best_baseline.mae
            best_baseline_model = best_baseline === nothing ? "--" : best_baseline.model
            if best_baseline !== nothing
                push!(best_avg_mse, best_baseline.mse)
                push!(best_avg_mae, best_baseline.mae)
                push!(best_avg_models, best_baseline.model)
            end

            # Find best (min) MSE, MAE, and NLL across models for this row
            valid_mse = [v for v in vcat(collect(values(row_mse)), [best_baseline_mse]) if v !== nothing]
            valid_mae = [v for v in vcat(collect(values(row_mae)), [best_baseline_mae]) if v !== nothing]
            valid_nll = [v for v in values(row_nll) if v !== nothing]
            best_mse = isempty(valid_mse) ? nothing : minimum(valid_mse)
            best_mae = isempty(valid_mae) ? nothing : minimum(valid_mae)
            best_nll = isempty(valid_nll) ? nothing : minimum(valid_nll)

            cells = String[]
            for mt in ENSEMBLE_MODEL_TYPES
                mse_str = fmt(row_mse[mt])
                mae_str = fmt(row_mae[mt])
                nll_str = fmt(row_nll[mt])
                if row_mse[mt] !== nothing && row_mse[mt] == best_mse
                    mse_str = "\\textbf{$mse_str}"
                end
                if row_mae[mt] !== nothing && row_mae[mt] == best_mae
                    mae_str = "\\textbf{$mae_str}"
                end
                if row_nll[mt] !== nothing && row_nll[mt] == best_nll
                    nll_str = "\\textbf{$nll_str}"
                end
                push!(cells, "$mse_str & $mae_str & $nll_str")
            end
            best_mse_str = fmt(best_baseline_mse)
            best_mae_str = fmt(best_baseline_mae)
            if best_baseline_mse !== nothing && best_baseline_mse == best_mse
                best_mse_str = "\\textbf{$best_mse_str}"
            end
            if best_baseline_mae !== nothing && best_baseline_mae == best_mae
                best_mae_str = "\\textbf{$best_mae_str}"
            end
            push!(cells, "$best_mse_str & $best_mae_str & $best_baseline_model")

            push!(lines, "& $h  & $(join(cells, " & ")) \\\\")
        end

        # Avg row
        avg_cells = String[]
        avg_mse_vals = Dict{String,Union{Nothing,Float64}}()
        avg_mae_vals = Dict{String,Union{Nothing,Float64}}()
        avg_nll_vals = Dict{String,Union{Nothing,Float64}}()
        for mt in ENSEMBLE_MODEL_TYPES
            if isempty(avg_mse[mt])
                avg_mse_vals[mt] = nothing
                avg_mae_vals[mt] = nothing
                avg_nll_vals[mt] = nothing
            else
                avg_mse_vals[mt] = sum(avg_mse[mt]) / length(avg_mse[mt])
                avg_mae_vals[mt] = sum(avg_mae[mt]) / length(avg_mae[mt])
                avg_nll_vals[mt] = sum(avg_nll[mt]) / length(avg_nll[mt])
            end
        end
        avg_best_mse = isempty(best_avg_mse) ? nothing : sum(best_avg_mse) / length(best_avg_mse)
        avg_best_mae = isempty(best_avg_mae) ? nothing : sum(best_avg_mae) / length(best_avg_mae)
        avg_best_model = isempty(best_avg_models) ? "--" : (length(unique(best_avg_models)) == 1 ? only(unique(best_avg_models)) : "varies")

        valid_avg_mse = [v for v in vcat(collect(values(avg_mse_vals)), [avg_best_mse]) if v !== nothing]
        valid_avg_mae = [v for v in vcat(collect(values(avg_mae_vals)), [avg_best_mae]) if v !== nothing]
        valid_avg_nll = [v for v in values(avg_nll_vals) if v !== nothing]
        best_avg_mse = isempty(valid_avg_mse) ? nothing : minimum(valid_avg_mse)
        best_avg_mae = isempty(valid_avg_mae) ? nothing : minimum(valid_avg_mae)
        best_avg_nll = isempty(valid_avg_nll) ? nothing : minimum(valid_avg_nll)

        for mt in ENSEMBLE_MODEL_TYPES
            mse_str = fmt(avg_mse_vals[mt])
            mae_str = fmt(avg_mae_vals[mt])
            nll_str = fmt(avg_nll_vals[mt])
            if avg_mse_vals[mt] !== nothing && avg_mse_vals[mt] == best_avg_mse
                mse_str = "\\textbf{$mse_str}"
            end
            if avg_mae_vals[mt] !== nothing && avg_mae_vals[mt] == best_avg_mae
                mae_str = "\\textbf{$mae_str}"
            end
            if avg_nll_vals[mt] !== nothing && avg_nll_vals[mt] == best_avg_nll
                nll_str = "\\textbf{$nll_str}"
            end
            push!(avg_cells, "$mse_str & $mae_str & $nll_str")
        end
        avg_best_mse_str = fmt(avg_best_mse)
        avg_best_mae_str = fmt(avg_best_mae)
        if avg_best_mse !== nothing && avg_best_mse == best_avg_mse
            avg_best_mse_str = "\\textbf{$avg_best_mse_str}"
        end
        if avg_best_mae !== nothing && avg_best_mae == best_avg_mae
            avg_best_mae_str = "\\textbf{$avg_best_mae_str}"
        end
        push!(avg_cells, "$avg_best_mse_str & $avg_best_mae_str & $avg_best_model")

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
