# FE/N convergence trace for VMP at N=50, D=65 — sanity check that VMP settles.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

fe = CSV.read(joinpath(RESULTS_DIR, "vmp_fe_grid_full.csv"), DataFrame)
df = sort(filter(r -> r.N == 50 && r.D == 65, fe), :iter)
@assert !isempty(df) "no N=50, D=65 trace in vmp_fe_grid_full.csv"

fe_per_N = df.free_energy ./ 50

# Convergence diagnostic (drift over final 5 iters, relative)
tail  = fe_per_N[max(end-4, 1):end]
drift = (maximum(tail) - minimum(tail)) / max(abs(tail[end]), 1.0)
stable = drift < 0.01

fig = Figure(size=(1100, 650), fontsize=15)
Label(fig[0, 1],
      "VMP (dynamic) FE/N trace  —  N=50, D=65  ($(nrow(df)) iterations)",
      fontsize=17, font=:bold)

ax = Axis(fig[1, 1];
          xlabel = "VMP iteration",
          ylabel = "FE / N")

lines!(ax,   df.iter, fe_per_N; color=:darkorange, linewidth=3)
scatter!(ax, df.iter, fe_per_N; color=:darkorange, markersize=10)

# Annotate first/last
text!(ax, df.iter[1],   fe_per_N[1];
      text="  iter 1:  FE/N = $(round(fe_per_N[1], digits=2))",
      fontsize=13, color=:black, align=(:left, :bottom), offset=(6, 6))
text!(ax, df.iter[end], fe_per_N[end];
      text="iter $(df.iter[end]):  FE/N = $(round(fe_per_N[end], digits=2))  ",
      fontsize=13, color=:black, align=(:right, :top), offset=(-6, -6))

# Convergence verdict badge
verdict = stable ? "converged  (Δ last 5 = $(round(100*drift, digits=2))%)" :
                    "still moving  (Δ last 5 = $(round(100*drift, digits=2))%)"
vclr    = stable ? :forestgreen : :firebrick
text!(ax, 0.98, 0.94; text=verdict, space=:relative,
      fontsize=14, color=vclr, font=:bold, align=(:right, :top))

save(joinpath(RESULTS_DIR, "vmp_fe_n50_d65.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "vmp_fe_n50_d65.pdf"), fig)
println("Saved vmp_fe_n50_d65.{png,pdf}")
println("iters=$(nrow(df))  FE/N final=$(round(fe_per_N[end], digits=3))  drift(last5)=$(round(100*drift, digits=2))%  -> $(stable ? "converged" : "drifting")")
