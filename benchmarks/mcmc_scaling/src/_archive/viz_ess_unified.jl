# Unified ESS vs N plot — merges scaling_v2_mcmc.csv (D=2,5,10 across N) with
# ess_vs_D_focused.csv (D=2,5,10,25,65 at N=10) and scaling_ess_small.csv (extra D=2 N's).
# One curve per D, 2 panels (Gibbs, NUTS).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

v2     = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_mcmc.csv"),     DataFrame)
focus  = CSV.read(joinpath(RESULTS_DIR, "ess_vs_D_focused.csv"),    DataFrame)
small  = CSV.read(joinpath(RESULTS_DIR, "scaling_ess_small.csv"),   DataFrame)

# Harmonize columns: keep :method, :N, :D, :min_ess, :median_ess, :t_per_iter_ms, :total_time_s
common = [:method, :N, :D, :min_ess, :median_ess, :total_time_s]

small_sub = filter(r -> r.sweep == "N@D2", small)
df = vcat(
    v2[:, common],
    focus[:, common],
    small_sub[:, common];
    cols = :union,
)
unique!(df, [:method, :N, :D])   # prefer first-seen (v2)
sort!(df, [:method, :D, :N])

D_VALUES = sort(unique(df.D))
# Manual palette — more distinguishable than viridis tail for D=65
palette = [:navy, :dodgerblue, :seagreen, :darkorange, :crimson]
D_COLORS = palette[1:length(D_VALUES)]

fig = Figure(size=(1700, 700), fontsize=15)
Label(fig[0, 1:2],
      "MCMC min-ESS vs N  —  one curve per D  (D=25 & D=65 only measured at N=10)",
      fontsize=17, font=:bold)

for (col, m) in enumerate(["Gibbs", "NUTS"])
    ax = Axis(fig[1, col];
              title = m,
              xlabel = "N (observations)",
              ylabel = col == 1 ? "min ESS" : "",
              xscale = log10, yscale = log10)
    for (k, D) in enumerate(D_VALUES)
        sub = sort(filter(r -> r.method == m && r.D == D, df), :N)
        sub = filter(r -> !ismissing(r.min_ess) && !isnan(r.min_ess) && r.min_ess > 0, sub)
        isempty(sub) && continue
        c = D_COLORS[k]
        lines!(ax, sub.N, sub.min_ess; color=c, linewidth=2.5)
        scatter!(ax, sub.N, sub.min_ess; color=c, markersize=14, label="D=$D")
        # Annotate numeric values on each point
        for r in eachrow(sub)
            text!(ax, r.N, r.min_ess; text=" $(round(Int, r.min_ess))",
                  fontsize=10, color=c, align=(:left, :bottom), offset=(3, 3))
        end
    end
    hlines!(ax, [100]; color=:gray, linestyle=:dash, linewidth=1)
    text!(ax, 10, 110; text="ESS=100 usability floor",
          color=:gray, fontsize=11, align=(:left, :bottom))
    axislegend(ax; position=:lt, labelsize=11)
    # Extend y-axis for NUTS so D=65 point below floor stays visible
    if m == "NUTS"
        ylims!(ax, 30, 3000)
    end
end

save(joinpath(RESULTS_DIR, "ess_unified.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "ess_unified.pdf"), fig)
println("Saved ess_unified.{png,pdf}")

show(stdout, df; allrows=true, allcols=true); println()
