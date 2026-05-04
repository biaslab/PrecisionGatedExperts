# ESS vs N for MCMC methods, one curve per D.
# Uses scaling_v2_mcmc.csv + (if available) scaling_ess_small.csv for richer N sweep at D=2.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

v2 = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_mcmc.csv"), DataFrame)

# Also pull extra D=2 points from the small sweep (finer N grid)
small_path = joinpath(RESULTS_DIR, "scaling_ess_small.csv")
extra = if isfile(small_path)
    df = CSV.read(small_path, DataFrame)
    # Keep only the N@D2 sweep rows (those went up to N=500)
    df = filter(r -> r.sweep == "N@D2", df)
    # Align schema: drop :sweep
    select(df, [:method, :N, :D, :min_ess, :median_ess, :t_per_iter_ms, :max_rhat, :total_time_s])
else
    DataFrame()
end

# scaling_v2_mcmc has no :sweep column — use as-is
mcmc = vcat(v2[:, [:method, :N, :D, :min_ess, :median_ess, :t_per_iter_ms, :max_rhat, :total_time_s]],
            extra; cols=:union)

# De-duplicate: if the same (method, N, D) appears in both, prefer scaling_v2 (more samples)
unique!(mcmc, [:method, :N, :D])
sort!(mcmc, [:method, :D, :N])

D_VALUES = sort(unique(mcmc.D))
D_COLORS = cgrad(:viridis, length(D_VALUES); categorical=true)

METHODS = ["Gibbs", "NUTS"]

fig = Figure(size=(1500, 700), fontsize=15)
Label(fig[0, 1:2],
      "MCMC min-ESS vs N  —  one curve per D";
      fontsize=17, font=:bold)

for (col, m) in enumerate(METHODS)
    ax = Axis(fig[1, col];
              title = m,
              xlabel = "N (observations)",
              ylabel = col == 1 ? "min ESS" : "",
              xscale = log10, yscale = log10)
    for (k, D) in enumerate(D_VALUES)
        df = sort(filter(r -> r.method == m && r.D == D, mcmc), :N)
        isempty(df) && continue
        # Drop NaN rows (some configs may have failed)
        df = filter(r -> !ismissing(r.min_ess) && !isnan(r.min_ess) && r.min_ess > 0, df)
        isempty(df) && continue
        c = D_COLORS[k]
        lines!(ax, df.N, df.min_ess; color=c, linewidth=2.5)
        scatter!(ax, df.N, df.min_ess; color=c, markersize=14, label="D=$D")
    end
    hlines!(ax, [100]; color=:gray, linestyle=:dash, linewidth=0.8)
    text!(ax, 10, 110; text="ESS=100 target", color=:gray, fontsize=10)
    axislegend(ax; position=:lb, labelsize=11)
end

save(joinpath(RESULTS_DIR, "ess_vs_N.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "ess_vs_N.pdf"), fig)
println("Saved ess_vs_N.{png,pdf}")

println("\nTable:")
show(stdout, mcmc; allrows=true, allcols=true)
println()
