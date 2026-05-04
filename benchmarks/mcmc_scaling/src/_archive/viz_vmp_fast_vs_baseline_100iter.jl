# VMP-fast (Fisher-metric log retraction on z) vs VMP-baseline (log-Euclidean
# product-manifold retraction) — FE convergence over iters at N=250, D=10.
#
# Two stacked panels:
#   (a) raw FE traces, baseline + fast
#   (b) difference  (FE_baseline − FE_fast)  emphasising the gap in a log-lin view

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const CSV_PATH    = joinpath(RESULTS_DIR, "vmp_fast_both_n250_d10_100iter.csv")

df = CSV.read(CSV_PATH, DataFrame)
sort!(df, :iter)
N = 250   # for FE/N normalisation (file was produced at N=250)

fe_base = df.fe_baseline ./ N
fe_fast = df.fe_fast     ./ N

# Per-iter gap (positive = fast is better, since lower FE is better)
gap = df.fe_baseline .- df.fe_fast

fig = Figure(size = (1400, 900), fontsize = 15)
Label(fig[0, 1],
      "VMP-fast (Fisher log retraction on z) vs VMP-baseline (product-manifold log-Euclidean)\n" *
      "N=250, D=10, 100 iterations.  Lower FE = better q(·) fit.",
      fontsize=16, font=:bold)

# ---- Panel 1: raw FE/N traces ----
ax1 = Axis(fig[1, 1];
           xlabel = "VMP iteration",
           ylabel = "FE / N",
           title  = "Free-energy trace per N")

lines!(ax1,   df.iter, fe_base; color=:steelblue,   linewidth=2.8, label="baseline")
scatter!(ax1, df.iter, fe_base; color=:steelblue,   markersize=8)

lines!(ax1,   df.iter, fe_fast; color=:darkorange,  linewidth=2.8, label="fast")
scatter!(ax1, df.iter, fe_fast; color=:darkorange,  markersize=8, marker=:diamond)

# Annotate endpoints
text!(ax1, df.iter[end], fe_base[end];
      text = "  baseline final: $(round(fe_base[end], digits=3))",
      fontsize=12, color=:steelblue, align=(:right, :top), offset=(-6, -4))
text!(ax1, df.iter[end], fe_fast[end];
      text = "  fast final: $(round(fe_fast[end], digits=3))",
      fontsize=12, color=:darkorange, align=(:right, :bottom), offset=(-6, 4))

axislegend(ax1; position = :rt)

# ---- Panel 2: gap  (baseline − fast) ----
ax2 = Axis(fig[2, 1];
           xlabel = "VMP iteration",
           ylabel = "FE_baseline − FE_fast (raw, positive → fast better)",
           title  = "Gap between baseline and fast (raw FE, not /N)")

lines!(ax2,   df.iter, gap; color=:firebrick, linewidth=2.8)
scatter!(ax2, df.iter, gap; color=:firebrick, markersize=8)
hlines!(ax2, [0]; color=:gray, linestyle=:dash, linewidth=1)
text!(ax2, df.iter[end], gap[end];
      text = "  iter $(df.iter[end]):  Δ = $(round(gap[end], digits=2))",
      fontsize=12, color=:firebrick, align=(:right, :center), offset=(-6, 0))

# Convergence verdict for each trace (drift over last 5 iters, relative)
function tail_drift(v)
    tail = v[max(end-4, 1):end]
    return (maximum(tail) - minimum(tail)) / max(abs(tail[end]), 1e-12)
end
drift_b = tail_drift(fe_base)
drift_f = tail_drift(fe_fast)
converged(x) = x < 0.005 ? "converged" : "still moving"
Label(fig[3, 1],
      "baseline:  Δ last 5 iters = $(round(100*drift_b, digits=2))%  → $(converged(drift_b))" *
      "                     " *
      "fast:  Δ last 5 iters = $(round(100*drift_f, digits=2))%  → $(converged(drift_f))",
      fontsize=13, font=:bold,
      color = drift_f < drift_b ? :darkorange : :steelblue)

save(joinpath(RESULTS_DIR, "vmp_fast_vs_baseline_n250_d10_100.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "vmp_fast_vs_baseline_n250_d10_100.pdf"), fig)
println("Saved vmp_fast_vs_baseline_n250_d10_100.{png,pdf}")

println("\nFinal FE:  baseline = $(df.fe_baseline[end]),  fast = $(df.fe_fast[end]),  gap = $(df.fe_baseline[end] - df.fe_fast[end])")
println("Final FE/N:  baseline = $(round(fe_base[end], digits=3)),  fast = $(round(fe_fast[end], digits=3))")
println("Last-5 drift (relative):  baseline = $(round(100*drift_b, digits=2))%,  fast = $(round(100*drift_f, digits=2))%")
