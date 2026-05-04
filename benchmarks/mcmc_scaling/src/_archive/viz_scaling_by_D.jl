# For each D, plot how posterior quality improves with N.
# 2 rows (rel-L2 error, 95% coverage) × 3 cols (Gibbs, VMP-10, VMP-converged).
# Drop NUTS since we showed it matches Gibbs exactly.

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

mcmc = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_mcmc.csv"), DataFrame)
vmp  = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_vmp.csv"),  DataFrame)
vmp_conv = CSV.read(joinpath(RESULTS_DIR, "pareto_quality_vmp_converged.csv"), DataFrame)

for df in (mcmc, vmp, vmp_conv)
    if !("status" in names(df)); df.status = fill("OK", nrow(df)); end
end

cols = [:method, :N, :D, :rel_err_total, :total_time_s, :cov95, :status]
all = vcat(mcmc[:, cols], vmp[:, cols], vmp_conv[:, cols])
filter!(r -> r.status == "OK", all)

# Methods to show (in order)
METHODS = ["Gibbs", "VMP-dynamic", "VMP-converged"]
method_label = Dict(
    "Gibbs"         => "Gibbs",
    "VMP-dynamic"   => "VMP-10 iters",
    "VMP-converged" => "VMP-converged",
)

# One colour per D value
D_VALUES = sort(unique(all.D))
D_COLORS = cgrad(:viridis, length(D_VALUES); categorical=true)

fig = Figure(size=(1800, 1100), fontsize=15)
Label(fig[0, 1:length(METHODS)],
      "Does posterior improve with N?  —  one curve per D value",
      fontsize=17, font=:bold)

# Row 1: rel-L2 error vs N (log-log)
for (col, m) in enumerate(METHODS)
    ax = Axis(fig[1, col];
              title = "$(method_label[m])",
              xlabel = "N (observations)",
              ylabel = col == 1 ? "rel-L2 error" : "",
              xscale = log10, yscale = log10)
    for (k, D) in enumerate(D_VALUES)
        df = sort(filter(r -> r.method == m && r.D == D, all), :N)
        isempty(df) && continue
        c = D_COLORS[k]
        lines!(ax, df.N, df.rel_err_total; color=c, linewidth=2.5)
        scatter!(ax, df.N, df.rel_err_total; color=c, markersize=14, label="D=$D")
    end
    hlines!(ax, [1.0]; color=:gray, linestyle=:dash, linewidth=0.8)
    axislegend(ax; position=:lt, labelsize=11)
end

# Row 2: 95% coverage vs N
for (col, m) in enumerate(METHODS)
    ax = Axis(fig[2, col];
              title = "$(method_label[m]) — coverage",
              xlabel = "N (observations)",
              ylabel = col == 1 ? "95% CI coverage" : "",
              xscale = log10)
    for (k, D) in enumerate(D_VALUES)
        df = sort(filter(r -> r.method == m && r.D == D, all), :N)
        isempty(df) && continue
        c = D_COLORS[k]
        lines!(ax, df.N, df.cov95; color=c, linewidth=2.5)
        scatter!(ax, df.N, df.cov95; color=c, markersize=14, label="D=$D")
    end
    hlines!(ax, [0.95]; color=:gray, linestyle=:dash, linewidth=0.8)
    ylims!(ax, 0, 1.05)
    axislegend(ax; position=:lt, labelsize=11)
end

save(joinpath(RESULTS_DIR, "scaling_by_D.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "scaling_by_D.pdf"), fig)
println("Saved scaling_by_D.{png,pdf}")
