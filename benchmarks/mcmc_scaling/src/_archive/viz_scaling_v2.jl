# Merge scaling_v2_mcmc.csv + scaling_v2_vmp.csv, render 6-panel figure:
#   Row 1: min ESS vs N (per D)   — Gibbs, NUTS
#   Row 2: total wall-time vs N   — Gibbs, NUTS, VMP
#   Row 3: Pareto: wall-time vs min ESS (log-log)

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using CairoMakie
using CSV, DataFrames
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

mcmc = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_mcmc.csv"), DataFrame)
vmp  = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_vmp.csv"),  DataFrame)
fe_trace = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_vmp_fe.csv"), DataFrame)

# Unify columns for plotting
mcmc.free_energy = fill(NaN, nrow(mcmc))
mcmc.status = fill("OK", nrow(mcmc))
vmp.min_ess = fill(NaN, nrow(vmp))
vmp.median_ess = fill(NaN, nrow(vmp))
vmp.t_per_iter_ms = fill(NaN, nrow(vmp))
vmp.max_rhat = fill(NaN, nrow(vmp))

# Common schema
cols = [:method, :N, :D, :min_ess, :median_ess, :t_per_iter_ms, :max_rhat, :total_time_s, :free_energy, :status]
all = vcat(mcmc[:, cols], vmp[:, cols])

D_VALUES = sort(unique(all.D))
D_COLORS = Dict(D_VALUES .=> [:tab10][1] |> _ ->
    [(:dodgerblue, :darkorange, :seagreen)[min(i, 3)] for i in 1:length(D_VALUES)]
)
method_style = Dict(
    "Gibbs"       => (:crimson,   :circle,   :solid),
    "NUTS"        => (:seagreen,  :diamond,  :solid),
    "VMP-dynamic" => (:darkorange, :star5,   :dash),
)

fig = Figure(size=(1500, 1400), fontsize=12)

Label(fig[0, 1:3],
      "Pareto: Gibbs (un-collapsed) | Collapsed NUTS (Enzyme, 1000 samples) | VMP dynamic\n" *
      "N ∈ $(sort(unique(all.N)))   ×   D ∈ $D_VALUES",
      fontsize=14, font=:bold)

# ============================================================
# Row 0 (first row): VMP free-energy convergence per (N, D)
# ============================================================
N_VALUES = sort(unique(fe_trace.N))
N_COLORS = cgrad(:viridis, length(N_VALUES); categorical=true)

for (col, D) in enumerate(D_VALUES)
    ax = Axis(fig[1, col]; xlabel="VMP iteration", ylabel="free energy",
              title="D=$D — VMP FE convergence")
    df_D = filter(r -> r.D == D, fe_trace)
    for (k, N) in enumerate(N_VALUES)
        df_N = sort(filter(r -> r.N == N, df_D), :iter)
        isempty(df_N) && continue
        lines!(ax, df_N.iter, df_N.free_energy;
               color=N_COLORS[k], linewidth=2, label="N=$N")
        scatter!(ax, df_N.iter, df_N.free_energy;
                 color=N_COLORS[k], markersize=6)
    end
    axislegend(ax; position=:rt, labelsize=9)
end

# ============================================================
# Row 2: min ESS vs N, per D — Gibbs vs NUTS
# ============================================================
for (col, D) in enumerate(D_VALUES)
    ax = Axis(fig[2, col]; xlabel="N", ylabel="min ESS",
              title="D=$D — min ESS vs N",
              xscale=log10, yscale=log10)
    df_D = filter(r -> r.D == D, all)
    for m in ("Gibbs", "NUTS")
        df = sort(filter(r -> r.method == m, df_D), :N)
        isempty(df) && continue
        col_m, mk, _ = method_style[m]
        lines!(ax, df.N, map(x -> max(x, 1.0), df.min_ess);
               color=col_m, linewidth=2, label=m)
        scatter!(ax, df.N, map(x -> max(x, 1.0), df.min_ess);
                 color=col_m, marker=mk, markersize=10)
    end
    hlines!(ax, [100]; color=:gray, linestyle=:dash)
    axislegend(ax; position=:lb)
end

# ============================================================
# Row 3: total wall-time vs N, per D — all three methods
# ============================================================
for (col, D) in enumerate(D_VALUES)
    ax = Axis(fig[3, col]; xlabel="N", ylabel="total time (s)",
              title="D=$D — wall-time vs N",
              xscale=log10, yscale=log10)
    df_D = filter(r -> r.D == D, all)
    for m in ("Gibbs", "NUTS", "VMP-dynamic")
        df = sort(filter(r -> r.method == m, df_D), :N)
        isempty(df) && continue
        col_m, mk, ls = method_style[m]
        lines!(ax, df.N, df.total_time_s;
               color=col_m, linewidth=2, linestyle=ls, label=m)
        scatter!(ax, df.N, df.total_time_s;
                 color=col_m, marker=mk, markersize=10)
    end
    axislegend(ax; position=:lt)
end

# ============================================================
# Row 4: Pareto — time vs min ESS.  MCMC gets ESS; VMP is shown as vertical band at its time.
# ============================================================
for (col, D) in enumerate(D_VALUES)
    ax = Axis(fig[4, col]; xlabel="total time (s)", ylabel="min ESS",
              title="D=$D — Pareto: time vs ESS",
              xscale=log10, yscale=log10)
    df_D = filter(r -> r.D == D, all)
    for m in ("Gibbs", "NUTS")
        df = sort(filter(r -> r.method == m, df_D), :N)
        isempty(df) && continue
        col_m, mk, _ = method_style[m]
        lines!(ax, df.total_time_s, map(x -> max(x, 1.0), df.min_ess);
               color=col_m, linewidth=2, label=m)
        scatter!(ax, df.total_time_s, map(x -> max(x, 1.0), df.min_ess);
                 color=col_m, marker=mk, markersize=10)
        # Annotate N on each point
        for r in eachrow(df)
            text!(ax, r.total_time_s, max(r.min_ess, 1.0);
                  text="N=$(r.N)", fontsize=8, color=col_m,
                  align=(:left, :bottom), offset=(3, 3))
        end
    end
    # VMP reference: vertical lines at VMP total_time_s
    df_vmp = sort(filter(r -> r.method == "VMP-dynamic" && r.D == D && r.status == "OK", all), :N)
    for r in eachrow(df_vmp)
        vlines!(ax, [r.total_time_s]; color=(:darkorange, 0.4),
                linewidth=1.2, linestyle=:dash)
        text!(ax, r.total_time_s, 1.5; text="VMP N=$(r.N)",
              fontsize=8, color=:darkorange, rotation=π/2, align=(:left, :bottom))
    end
    hlines!(ax, [100]; color=:gray, linestyle=:dot)
    axislegend(ax; position=:lb)
end

save(joinpath(RESULTS_DIR, "scaling_v2.png"), fig; px_per_unit=3)
save(joinpath(RESULTS_DIR, "scaling_v2.pdf"), fig)
println("Saved scaling_v2.{png,pdf}")

println("\nFinal table (sorted):")
show(stdout, sort(all, [:D, :N, :method]); allrows=true, allcols=true)
println()
