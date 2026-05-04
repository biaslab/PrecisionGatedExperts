# 3×3 grid of VMP free-energy convergence plots — large, readable.

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

fe = CSV.read(joinpath(RESULTS_DIR, "scaling_v2_vmp_fe.csv"), DataFrame)

N_VALUES = sort(unique(fe.N))
D_VALUES = sort(unique(fe.D))
nR = length(D_VALUES)
nC = length(N_VALUES)

# Large figure — aim for full-page PDF size.
fig = Figure(size = (1800, 1400), fontsize = 15)

Label(fig[0, 1:nC],
      "VMP (dynamic) free-energy convergence  —  y-axis: FE / N (per observation)",
      fontsize = 18, font = :bold, tellheight = true)

axs = Matrix{Axis}(undef, nR, nC)

for (row, D) in enumerate(D_VALUES)
    for (col, N) in enumerate(N_VALUES)
        ax = Axis(fig[row, col];
                  title = "N = $N,   D = $D",
                  titlesize = 15,
                  xlabel = row == nR ? "VMP iteration" : "",
                  ylabel = col == 1 ? "FE / N" : "",
                  xlabelsize = 13, ylabelsize = 13,
                  xticklabelsize = 12, yticklabelsize = 12)

        df = sort(filter(r -> r.N == N && r.D == D, fe), :iter)
        if isempty(df)
            text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                  align = (:center, :center), color = :gray)
            axs[row, col] = ax
            continue
        end

        fe_norm = df.free_energy ./ N
        lines!(ax, df.iter, fe_norm;
               color = :darkorange, linewidth = 3)
        scatter!(ax, df.iter, fe_norm;
                 color = :darkorange, markersize = 11)

        # Convergence flag — % drift in last 3 iters (on normalized FE)
        final = fe_norm[end]
        if length(fe_norm) >= 3
            last3 = fe_norm[end-2:end]
            rel = (maximum(last3) - minimum(last3)) / max(abs(final), 1.0)
            stable = rel < 0.01
            label = stable ? "stable (Δ=$(round(100*rel, digits=2))%)" :
                              "drifting (Δ=$(round(100*rel, digits=1))%)"
            clr = stable ? :forestgreen : :firebrick
            text!(ax, 0.04, 0.94; text = label, space = :relative,
                  fontsize = 13, color = clr, font = :bold,
                  align = (:left, :top))
        end

        axs[row, col] = ax
    end
end

# Link y per row so shapes are comparable at same D across N
for row in 1:nR
    linkyaxes!(axs[row, :]...)
end

# Ensure the title row is thin and plot rows expand
rowsize!(fig.layout, 0, Fixed(40))
for row in 1:nR
    rowsize!(fig.layout, row, Relative(1 / nR))
end
for col in 1:nC
    colsize!(fig.layout, col, Relative(1 / nC))
end

save(joinpath(RESULTS_DIR, "vmp_fe_grid.png"), fig; px_per_unit = 2)
save(joinpath(RESULTS_DIR, "vmp_fe_grid.pdf"), fig)
println("Saved vmp_fe_grid.{png,pdf}")

println("\nSummary (FE divided by N):")
println(rpad("N/D", 10), rpad("first FE/N", 14), rpad("final FE/N", 14),
        rpad("Δ last3 %", 14), "verdict")
for D in D_VALUES, N in N_VALUES
    df = sort(filter(r -> r.N == N && r.D == D, fe), :iter)
    isempty(df) && continue
    nf = df.free_energy ./ N
    rel = length(nf) >= 3 ?
          (maximum(nf[end-2:end]) - minimum(nf[end-2:end])) /
              max(abs(nf[end]), 1.0) : NaN
    println(rpad("$N/$D", 10),
            rpad(string(round(nf[1], digits=2)), 14),
            rpad(string(round(nf[end], digits=2)), 14),
            rpad(string(round(100*rel, digits=2)), 14),
            rel < 0.01 ? "stable" : "drifting")
end
