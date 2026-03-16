using JLD2
using Printf
using Plots; gr()
using Plots.PlotMeasures

# ─── Configuration ────────────────────────────────────────────────────────────
const RESULTS_DIR = joinpath(@__DIR__, "..", "paper", "results_vae")
const FIGURES_DIR = joinpath(@__DIR__, "..", "paper", "figures")
const HORIZONS = [96, 192, 336, 720]

# Datasets to include in the radar chart (change this list to add/remove datasets)
# Each entry: (internal_name, display_label, prediction_type)
const DATASETS_DEFAULT = [
    ("ETTh1",         "ETTh1",         :univariate),
    ("ETTh2",         "ETTh2",         :univariate),
    ("exchange_rate", "Exchange Rate", :multivariate),
    ("electricity",   "Electricity",   :multivariate),
    ("traffic",       "Traffic",       :multivariate),  # uncomment for 5-axis pentagon
]

const ENSEMBLE_MODEL_TYPES = [
    "static", "dynamic", "dynamic_diagonal",
    "noisy_experts", "noisy_experts_diagonal",
    "neural_ensemble", "neural_ensemble_big",
]
const ENSEMBLE_MODEL_LABELS = Dict(
    "static"                  => "Static",
    "dynamic"                 => "Dyn.",
    "dynamic_diagonal"        => "Dyn. Diag.",
    "noisy_experts"           => "Noisy",
    "noisy_experts_diagonal"  => "Noisy Diag.",
    "neural_ensemble"         => "MoE",
    "neural_ensemble_big"     => "MoE Big",
)
const MODEL_DIR = Dict(
    "noisy_experts_diagonal" => "noisy_diagonal",
    "noisy_experts"          => "noisy_experts",
)

const MODEL_COLORS = Dict(
    "static"                  => :royalblue,
    "dynamic"                 => :orangered,
    "dynamic_diagonal"        => :forestgreen,
    "noisy_experts"           => :darkorchid,
    "noisy_experts_diagonal"  => :goldenrod,
    "neural_ensemble"         => :deeppink,
    "neural_ensemble_big"     => :saddlebrown,
)

# Cap extreme values (e.g., MoE NLL can be 1e22+)
const VALUE_CAP = 1e40

# Models to exclude from axis range calculation (they distort the scale)
# Their values will be clamped to the axis range instead
const RANGE_EXCLUDE_MODELS = Set{String}()  # include all models in axis range (log compresses MoE NLL)

const BASELINE_MODELS = ["CNN", "DLinear", "LSTM", "MLP", "NConv"]
const BASELINE_METRICS = Dict(
    "exchange_rate" => Dict(
        96  => Dict("CNN" => (mse=2.455, mae=1.187), "DLinear" => (mse=0.157, mae=0.306), "LSTM" => (mse=1.150, mae=0.881), "MLP" => (mse=0.183, mae=0.317), "NConv" => (mse=0.180, mae=0.314)),
        192 => Dict("CNN" => (mse=3.559, mae=1.448), "DLinear" => (mse=0.325, mae=0.446), "LSTM" => (mse=2.589, mae=1.337), "MLP" => (mse=0.400, mae=0.478), "NConv" => (mse=0.410, mae=0.479)),
        336 => Dict("CNN" => (mse=2.295, mae=1.184), "DLinear" => (mse=0.558, mae=0.606), "LSTM" => (mse=1.536, mae=1.023), "MLP" => (mse=0.804, mae=0.680), "NConv" => (mse=0.820, mae=0.680)),
        720 => Dict("CNN" => (mse=4.902, mae=1.706), "DLinear" => (mse=1.495, mae=0.984), "LSTM" => (mse=1.982, mae=1.147), "MLP" => (mse=2.187, mae=1.166), "NConv" => (mse=2.426, mae=1.225)),
    ),
    "ETTh1" => Dict(
        96  => Dict("CNN" => (mse=0.900, mae=0.734), "DLinear" => (mse=0.504, mae=0.484), "LSTM" => (mse=0.762, mae=0.646), "MLP" => (mse=0.516, mae=0.493), "NConv" => (mse=0.665, mae=0.547)),
        192 => Dict("CNN" => (mse=1.594, mae=0.966), "DLinear" => (mse=0.553, mae=0.523), "LSTM" => (mse=0.964, mae=0.725), "MLP" => (mse=0.578, mae=0.528), "NConv" => (mse=0.779, mae=0.610)),
        336 => Dict("CNN" => (mse=1.717, mae=0.958), "DLinear" => (mse=0.611, mae=0.560), "LSTM" => (mse=1.189, mae=0.812), "MLP" => (mse=0.643, mae=0.564), "NConv" => (mse=0.776, mae=0.625)),
        720 => Dict("CNN" => (mse=1.186, mae=0.846), "DLinear" => (mse=0.794, mae=0.688), "LSTM" => (mse=1.204, mae=0.832), "MLP" => (mse=0.910, mae=0.724), "NConv" => (mse=1.084, mae=0.787)),
    ),
    "ETTh2" => Dict(
        96  => Dict("CNN" => (mse=2.111, mae=0.955), "DLinear" => (mse=0.293, mae=0.368), "LSTM" => (mse=0.987, mae=0.774), "MLP" => (mse=0.300, mae=0.364), "NConv" => (mse=0.336, mae=0.388)),
        192 => Dict("CNN" => (mse=2.274, mae=1.099), "DLinear" => (mse=0.345, mae=0.410), "LSTM" => (mse=1.347, mae=0.792), "MLP" => (mse=0.354, mae=0.401), "NConv" => (mse=0.417, mae=0.434)),
        336 => Dict("CNN" => (mse=2.359, mae=1.198), "DLinear" => (mse=0.428, mae=0.473), "LSTM" => (mse=1.517, mae=0.952), "MLP" => (mse=0.421, mae=0.445), "NConv" => (mse=0.479, mae=0.465)),
        720 => Dict("CNN" => (mse=1.826, mae=1.052), "DLinear" => (mse=0.639, mae=0.585), "LSTM" => (mse=1.155, mae=0.884), "MLP" => (mse=0.695, mae=0.578), "NConv" => (mse=0.759, mae=0.596)),
    ),
    "electricity" => Dict(
        96  => Dict("CNN" => (mse=0.350, mae=0.432), "DLinear" => (mse=0.176, mae=0.267), "LSTM" => (mse=0.396, mae=0.443), "MLP" => (mse=0.178, mae=0.266), "NConv" => (mse=0.348, mae=0.363)),
        192 => Dict("CNN" => (mse=0.388, mae=0.462), "DLinear" => (mse=0.251, mae=0.326), "LSTM" => (mse=0.335, mae=0.403), "MLP" => (mse=0.260, mae=0.324), "NConv" => (mse=0.307, mae=0.345)),
        336 => Dict("CNN" => (mse=0.364, mae=0.434), "DLinear" => (mse=0.223, mae=0.314), "LSTM" => (mse=0.337, mae=0.407), "MLP" => (mse=0.229, mae=0.310), "NConv" => (mse=0.248, mae=0.319)),
        720 => Dict("CNN" => (mse=0.418, mae=0.462), "DLinear" => (mse=0.344, mae=0.404), "LSTM" => (mse=0.425, mae=0.462), "MLP" => (mse=0.370, mae=0.407), "NConv" => (mse=0.462, mae=0.449)),
    ),
    "traffic" => Dict(
        96  => Dict("CNN" => (mse=0.743, mae=0.422), "DLinear" => (mse=0.475, mae=0.308), "LSTM" => (mse=0.845, mae=0.469), "MLP" => (mse=0.478, mae=0.305), "NConv" => (mse=1.205, mae=0.596)),
        192 => Dict("CNN" => (mse=0.737, mae=0.423), "DLinear" => (mse=0.710, mae=0.431), "LSTM" => (mse=0.817, mae=0.443), "MLP" => (mse=0.716, mae=0.430), "NConv" => (mse=0.831, mae=0.453)),
        336 => Dict("CNN" => (mse=0.730, mae=0.403), "DLinear" => (mse=0.547, mae=0.345), "LSTM" => (mse=0.740, mae=0.407), "MLP" => (mse=0.556, mae=0.337), "NConv" => (mse=0.622, mae=0.344)),
        720 => Dict("CNN" => (mse=0.800, mae=0.447), "DLinear" => (mse=0.814, mae=0.473), "LSTM" => (mse=1.055, mae=0.576), "MLP" => (mse=0.820, mae=0.466), "NConv" => (mse=1.191, mae=0.588)),
    ),
)

# ─── Data Loading ─────────────────────────────────────────────────────────────

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
        dir = joinpath(RESULTS_DIR, dir_name)
        !isdir(dir) && return nothing
        prefix = replace(fname, ".jld2" => "")
        matches = filter(f -> startswith(f, prefix) && endswith(f, ".jld2"), readdir(dir))
        isempty(matches) && return nothing
        fpath = joinpath(dir, first(matches))
    end
    data = JLD2.load(fpath)
    metrics = data["ensemble_metrics"]
    return (mse=metrics.mse, mae=metrics.mae, nll=-metrics.nll)
end

clean_val(v, cap) = (v === nothing || !isfinite(v)) ? nothing : min(log(abs(v)), log(cap))

function best_baseline_metric(dataset::String, metric::Symbol)
    by_dataset = get(BASELINE_METRICS, dataset, nothing)
    by_dataset === nothing && return nothing
    vals = Float64[]
    for h in HORIZONS
        by_horizon = get(by_dataset, h, nothing)
        by_horizon === nothing && continue
        for model in BASELINE_MODELS
            m = get(by_horizon, model, nothing)
            m === nothing && continue
            v = hasfield(typeof(m), metric) ? getfield(m, metric) : nothing
            v !== nothing && isfinite(v) && push!(vals, v)
        end
    end
    isempty(vals) ? nothing : minimum(vals)
end

function fmt_tick(v)
    abs_v = abs(v)
    if abs_v >= 1000
        exp = floor(Int, log10(abs_v))
        coeff = v / 10.0^exp
        return @sprintf("%.1f×10^%d", coeff, exp)
    elseif abs_v >= 100
        return @sprintf("%.0f", v)
    elseif abs_v >= 10
        return @sprintf("%.1f", v)
    elseif abs_v >= 1
        return @sprintf("%.2f", v)
    else
        return @sprintf("%.3f", v)
    end
end

# ─── Radar Polygon Area (Shoelace in Polar) ───────────────────────────────────

function radar_polygon_area(radii, angles)
    n = length(radii)
    area = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        area += radii[i] * radii[j] * sin(angles[j] - angles[i])
    end
    return abs(area) / 2.0
end

# ─── Main Radar Chart Builder ─────────────────────────────────────────────────

function build_radar_chart(metric::Symbol;
                           datasets = DATASETS_DEFAULT,
                           cap = VALUE_CAP)
    N = length(datasets)
    dataset_labels = [d[2] for d in datasets]

    # 1. Compute average metric per model × dataset
    raw_data = Dict{String, Vector{Union{Nothing,Float64}}}()
    for mt in ENSEMBLE_MODEL_TYPES
        vals = Union{Nothing,Float64}[]
        for (ds, _, pred_type) in datasets
            horizon_vals = Float64[]
            for h in HORIZONS
                m = load_metrics(ds, h, pred_type, mt)
                m === nothing && continue
                v = getfield(m, metric)
                cv = clean_val(v, cap)
                cv !== nothing && push!(horizon_vals, cv)
            end
            push!(vals, isempty(horizon_vals) ? nothing : sum(horizon_vals) / length(horizon_vals))
        end
        raw_data[mt] = vals
    end

    # 2. Get baseline values for axis range (not plotted)
    baseline_vals = Union{Nothing,Float64}[]
    for (ds, _, _) in datasets
        bv = best_baseline_metric(ds, metric)
        push!(baseline_vals, bv)
    end

    # 3. Per-axis range: [best (min), worst (max)]
    #    Exclude RANGE_EXCLUDE_MODELS from range calc to avoid scale distortion (e.g., MoE NLL)
    axis_best  = fill(Inf, N)
    axis_worst = fill(-Inf, N)
    for i in 1:N
        for mt in ENSEMBLE_MODEL_TYPES
            mt in RANGE_EXCLUDE_MODELS && continue
            v = raw_data[mt][i]
            v === nothing && continue
            axis_best[i]  = min(axis_best[i], v)
            axis_worst[i] = max(axis_worst[i], v)
        end
        # Include baseline in range
        bv = baseline_vals[i]
        if bv !== nothing
            axis_best[i]  = min(axis_best[i], bv)
            axis_worst[i] = max(axis_worst[i], bv)
        end
    end
    # Add 10% padding to worst so lines don't sit at origin
    for i in 1:N
        span = axis_worst[i] - axis_best[i]
        if span ≈ 0.0
            axis_worst[i] = axis_best[i] + 1.0
        else
            axis_worst[i] += 0.10 * span
        end
    end

    # 4. Normalize: lower-is-better → best=1 (outer), worst=0 (center)
    #    Clamp values outside the range (e.g., MoE models with huge NLL)
    function normalize_val(v, i)
        v === nothing && return 0.0
        n = (v - axis_worst[i]) / (axis_best[i] - axis_worst[i])
        return clamp(n, 0.0, 1.0)
    end

    normed = Dict{String, Vector{Float64}}()
    for mt in ENSEMBLE_MODEL_TYPES
        normed[mt] = [normalize_val(raw_data[mt][i], i) for i in 1:N]
    end

    # 5. Find best model (largest total area)
    θ_raw = LinRange(0, 2π, N + 1)[1:N] |> collect
    areas = Dict{String, Float64}()
    for mt in ENSEMBLE_MODEL_TYPES
        areas[mt] = radar_polygon_area(normed[mt], θ_raw)
    end
    max_area = radar_polygon_area(ones(N), θ_raw)
    best_model = argmax(areas)

    # 6. Print area table
    metric_upper = uppercase(string(metric))
    println("\n--- Radar Polygon Areas ($metric_upper) ---")
    @printf("%-20s %10s %10s\n", "Model", "Area", "% of max")
    println("-"^42)
    sorted_models = sort(collect(ENSEMBLE_MODEL_TYPES), by=mt -> -areas[mt])
    for mt in sorted_models
        label = ENSEMBLE_MODEL_LABELS[mt]
        @printf("%-20s %10.4f %9.1f%%\n", label, areas[mt], 100 * areas[mt] / max_area)
    end
    @printf("\n%-20s %10.4f %10s\n", "Max possible", max_area, "100.0%")

    # Also print raw average values per dataset
    println("\n--- Raw Average $metric_upper per Dataset ---")
    header = @sprintf("%-20s", "Model")
    for dl in dataset_labels
        header *= @sprintf(" %14s", dl)
    end
    println(header)
    println("-"^(20 + 15 * N))
    for mt in sorted_models
        row = @sprintf("%-20s", ENSEMBLE_MODEL_LABELS[mt])
        for i in 1:N
            v = raw_data[mt][i]
            row *= @sprintf(" %14s", v === nothing ? "—" : fmt_tick(v))
        end
        println(row)
    end

    # 7. Build polar plot
    θ = LinRange(0, 2π, N + 1) |> collect  # N+1 to close the polygon

    # Plot all models (log transform compresses scale enough for all to be visible)
    plotted_models = copy(ENSEMBLE_MODEL_TYPES)

    p = plot(
        size = (950, 900),
        proj = :polar,
        lims = (0, 1.18),
        xaxis = false,
        yaxis = false,
        grid = true,
        gridalpha = 0.25,
        gridlinewidth = 0.5,
        bg = :white,
        left_margin = 30mm,
        right_margin = 25mm,
        top_margin = 15mm,
        bottom_margin = 20mm,
        title = "Log $metric_upper — Average over Horizons",
        titlefontsize = 14,
        legend = :outertopright,
        legendfontsize = 10,
    )

    # Draw models (best model last for z-ordering)
    best_plotted = best_model in plotted_models ? best_model : (isempty(plotted_models) ? best_model : plotted_models[1])
    draw_order = [mt for mt in plotted_models if mt != best_plotted]
    push!(draw_order, best_plotted)

    for mt in draw_order
        R = vcat(normed[mt], [normed[mt][1]])  # close polygon
        lw = mt == best_plotted ? 3.0 : 1.5
        fa = mt == best_plotted ? 0.12 : 0.0
        plot!(p, θ, R,
            proj = :polar,
            linewidth = lw,
            color = MODEL_COLORS[mt],
            fill = fa > 0 ? (true, fa, MODEL_COLORS[mt]) : false,
            label = ENSEMBLE_MODEL_LABELS[mt],
            markershape = :none,
        )
    end

    # 8. Add axis labels (dataset names) — placed well outside the plot area
    for i in 1:N
        angle = 2π * (i - 1) / N
        angle_deg = mod(rad2deg(angle), 360)

        # Determine alignment based on quadrant
        if angle_deg < 10 || angle_deg > 350       # right
            label_r = 1.38; ha = :left;   va = :center
        elseif angle_deg < 80                       # upper-right
            label_r = 1.35; ha = :left;   va = :bottom
        elseif angle_deg < 100                      # top
            label_r = 1.40; ha = :center; va = :bottom
        elseif angle_deg < 170                      # upper-left
            label_r = 1.35; ha = :right;  va = :bottom
        elseif angle_deg < 190                      # left
            label_r = 1.38; ha = :right;  va = :center
        elseif angle_deg < 260                      # lower-left
            label_r = 1.35; ha = :right;  va = :top
        elseif angle_deg < 280                      # bottom
            label_r = 1.40; ha = :center; va = :top
        else                                        # lower-right
            label_r = 1.35; ha = :left;   va = :top
        end

        x = label_r * cos(angle)
        y = label_r * sin(angle)
        annotate!(p, x, y, text(dataset_labels[i], 12, ha, va, :bold))
    end

    # 9. Per-axis numeric tick labels (along each spoke, offset slightly)
    for i in 1:N
        angle = 2π * (i - 1) / N
        for frac in [0.25, 0.5, 0.75, 1.0]
            raw_tick = axis_worst[i] + frac * (axis_best[i] - axis_worst[i])
            label = fmt_tick(raw_tick)
            # Offset labels slightly off the spoke
            angle_offset = 0.10
            r_pos = frac
            x = r_pos * cos(angle + angle_offset)
            y = r_pos * sin(angle + angle_offset)
            annotate!(p, x, y, text(label, 6, :left, :gray40))
        end
    end

    # 10. Save
    mkpath(FIGURES_DIR)
    outpath = joinpath(FIGURES_DIR, "radar_$(metric).pdf")
    savefig(p, outpath)
    println("\nSaved: $outpath")

    outpath_png = joinpath(FIGURES_DIR, "radar_$(metric).png")
    savefig(p, outpath_png)
    println("Saved: $outpath_png")

    return p
end

# ─── Generate Charts ──────────────────────────────────────────────────────────
build_radar_chart(:mse)
build_radar_chart(:nll)
