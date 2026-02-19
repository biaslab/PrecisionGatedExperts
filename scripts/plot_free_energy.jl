using Pkg
Pkg.activate(; temp=true)
Pkg.add(["JLD2", "CairoMakie"]; io=devnull)

using JLD2
using CairoMakie

const RESULTS_DIR = joinpath(@__DIR__, "..", "final_results")
const HORIZONS = [96, 192, 336, 720]
const MODEL_TYPES = ["static", "dynamic", "hierarchical"]
const COLORS = Dict("static" => :royalblue, "dynamic" => :firebrick, "hierarchical" => :forestgreen)
const LABELS = Dict("static" => "Static", "dynamic" => "Dynamic", "hierarchical" => "Hierarchical")

function load_fe(horizon, model_type)
    fname = "exchange_rate_h$(horizon)_multivariate_$(model_type).jld2"
    fpath = joinpath(RESULTS_DIR, fname)
    !isfile(fpath) && return nothing
    data = JLD2.load(fpath)
    haskey(data, "free_energy") || return nothing
    fe = data["free_energy"]
    (fe isa AbstractVector && !isempty(fe)) || return nothing
    return Float64.(fe)
end

# --- Full convergence plot (all iters) ---
fig = Figure(size=(1200, 900), fontsize=14)

for (col, h) in enumerate(HORIZONS)
    ax_full = Axis(fig[1, col];
        title="H = $h (full)",
        xlabel=col == 1 ? "Iteration" : "",
        ylabel=col == 1 ? "Free Energy" : "",
        xgridvisible=false, ygridvisible=true)

    ax_zoom = Axis(fig[2, col];
        title="H = $h (last 100)",
        xlabel="Iteration",
        ylabel=col == 1 ? "Free Energy" : "",
        xgridvisible=false, ygridvisible=true)

    for mt in MODEL_TYPES
        fe = load_fe(h, mt)
        fe === nothing && continue
        n = length(fe)

        lines!(ax_full, 1:n, fe; color=COLORS[mt], linewidth=1.5, label=LABELS[mt])

        if n > 100
            start = n - 100
            lines!(ax_zoom, start:n, fe[start:end]; color=COLORS[mt], linewidth=1.5, label=LABELS[mt])
        end
    end

    # Static is way different scale, so separate axes won't help much.
    # Just let Makie auto-scale.
end

Legend(fig[0, :], fig.content[1]; orientation=:horizontal, framevisible=false, tellheight=true)
Label(fig[-1, :], "Exchange Rate — Free Energy Convergence"; fontsize=18, font=:bold, tellwidth=false)

outpath = joinpath(@__DIR__, "..", "viz", "free_energy_convergence.png")
mkpath(dirname(outpath))
save(outpath, fig; px_per_unit=2)
println("Saved: $outpath")

# --- Per-model separate plots (dynamic & hierarchical only, since static is flat) ---
for mt in ["dynamic", "hierarchical"]
    fig2 = Figure(size=(1200, 400), fontsize=14)

    for (col, h) in enumerate(HORIZONS)
        fe = load_fe(h, mt)
        fe === nothing && continue
        n = length(fe)

        ax = Axis(fig2[1, col];
            title="$(LABELS[mt]) H=$h",
            xlabel="Iteration",
            ylabel=col == 1 ? "Free Energy" : "",
            xgridvisible=false, ygridvisible=true)

        lines!(ax, 1:n, fe; color=COLORS[mt], linewidth=2)

        # Mark last 100 region
        if n > 100
            vlines!(ax, [n - 100]; color=:gray, linestyle=:dash, linewidth=1)
        end
    end

    outpath2 = joinpath(@__DIR__, "..", "viz", "free_energy_$(mt).png")
    save(outpath2, fig2; px_per_unit=2)
    println("Saved: $outpath2")
end

# --- Zoom: last 100 iters, dynamic & hierarchical on same scale per horizon ---
fig3 = Figure(size=(1200, 800), fontsize=14)

for (col, h) in enumerate(HORIZONS)
    for (row, mt) in enumerate(["dynamic", "hierarchical"])
        fe = load_fe(h, mt)
        fe === nothing && continue
        n = length(fe)
        n <= 100 && continue

        start = n - 100
        ax = Axis(fig3[row, col];
            title="$(LABELS[mt]) H=$h (last 100)",
            xlabel=row == 2 ? "Iteration" : "",
            ylabel=col == 1 ? "Free Energy" : "",
            xgridvisible=false, ygridvisible=true)

        lines!(ax, start:n, fe[start:end]; color=COLORS[mt], linewidth=2)

        # Annotate Δ per iter
        delta = fe[end] - fe[end-1]
        text!(ax, start + 5, fe[start+5];
            text="Δ/iter = $(round(Int, delta))",
            fontsize=11, color=:gray30)
    end
end

Label(fig3[0, :], "Last 100 Iterations — Still Converging?"; fontsize=16, font=:bold, tellwidth=false)

outpath3 = joinpath(@__DIR__, "..", "viz", "free_energy_last100.png")
save(outpath3, fig3; px_per_unit=2)
println("Saved: $outpath3")
