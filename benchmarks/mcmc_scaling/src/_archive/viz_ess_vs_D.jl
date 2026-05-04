# ESS vs D at fixed N=10 — shows NUTS degradation at paper's D=65.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

df = CSV.read(joinpath(RESULTS_DIR, "ess_vs_D_focused.csv"), DataFrame)
sort!(df, :D)

fig = Figure(size=(1500, 600), fontsize=15)
Label(fig[0, 1:2],
      "MCMC scaling with D  (N=10 fixed)  —  NUTS mixing degrades below usability at paper's D=65",
      fontsize=16, font=:bold)

# ---- Left: min ESS vs D ----
ax1 = Axis(fig[1, 1];
           xlabel = "D (feature dim)", ylabel = "min ESS (out of 1000 for NUTS / 2000 for Gibbs)",
           title = "Minimum effective samples vs D",
           xscale = log10, yscale = log10)
for (m, color, marker) in [("Gibbs", :crimson, :circle), ("NUTS", :seagreen, :diamond)]
    sub = filter(r -> r.method == m, df)
    lines!(ax1, sub.D, sub.min_ess; color=color, linewidth=3, label=m)
    scatter!(ax1, sub.D, sub.min_ess; color=color, marker=marker, markersize=16)
    for r in eachrow(sub)
        text!(ax1, r.D, r.min_ess;
              text=" $(round(Int, r.min_ess))", fontsize=12, color=color,
              align=(:left, :bottom), offset=(4, 4))
    end
end
hlines!(ax1, [100]; color=:gray, linestyle=:dash, linewidth=1)
text!(ax1, 2, 110; text="ESS=100 usability floor",
      color=:gray, fontsize=11, align=(:left, :bottom))
axislegend(ax1; position=:lb)

# ---- Right: wall-clock time vs D ----
ax2 = Axis(fig[1, 2];
           xlabel = "D (feature dim)", ylabel = "wall-clock time (s)",
           title = "Time per run vs D",
           xscale = log10, yscale = log10)
for (m, color, marker) in [("Gibbs", :crimson, :circle), ("NUTS", :seagreen, :diamond)]
    sub = filter(r -> r.method == m, df)
    lines!(ax2, sub.D, sub.total_time_s; color=color, linewidth=3, label=m)
    scatter!(ax2, sub.D, sub.total_time_s; color=color, marker=marker, markersize=16)
    for r in eachrow(sub)
        text!(ax2, r.D, r.total_time_s;
              text=" $(round(r.total_time_s, digits=1))s", fontsize=11, color=color,
              align=(:left, :bottom), offset=(4, 4))
    end
end
axislegend(ax2; position=:lt)

save(joinpath(RESULTS_DIR, "ess_vs_D.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "ess_vs_D.pdf"), fig)
println("Saved ess_vs_D.{png,pdf}")

show(stdout, df; allrows=true, allcols=true)
println()
