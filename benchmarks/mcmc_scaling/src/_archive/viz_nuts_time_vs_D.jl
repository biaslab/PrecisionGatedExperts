# NUTS wall-clock time vs D at N=10 — clean single-panel plot with power-law fit.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

df = CSV.read(joinpath(RESULTS_DIR, "ess_vs_D_focused.csv"), DataFrame)
nuts = sort(filter(r -> r.method == "NUTS", df), :D)

# Power-law fit: log(t) = a + b * log(D) → linear regression
logD = log10.(nuts.D)
logt = log10.(nuts.total_time_s)
b = sum((logD .- mean(logD)) .* (logt .- mean(logt))) / sum((logD .- mean(logD)).^2)
a = mean(logt) - b * mean(logD)

fig = Figure(size=(1100, 750), fontsize=16)
Label(fig[0, 1],
      "NUTS (collapsed, Enzyme) wall-clock time vs D  (N=10, 200 warmup + 1000 samples)",
      fontsize=17, font=:bold)

# Explicit decade ticks so log scale is unmistakable
Dticks   = [1, 2, 5, 10, 25, 65, 150]
tticks   = [1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
ax = Axis(fig[1, 1];
          xlabel = "D (feature dim)  — log scale",
          ylabel = "wall-clock time (s)  — log scale",
          xscale = log10, yscale = log10,
          xticks = (Dticks, string.(Dticks)),
          yticks = (tticks, string.(tticks)),
          xminorticksvisible = true, yminorticksvisible = true,
          xminorgridvisible  = true, yminorgridvisible  = true,
          xminorticks = IntervalsBetween(9), yminorticks = IntervalsBetween(9),
          xgridcolor = (:black, 0.15), ygridcolor = (:black, 0.15),
          xminorgridcolor = (:black, 0.05), yminorgridcolor = (:black, 0.05))

# Power-law fit line extended to span a full decade of D on log axes
Dfit = 10.0 .^ range(log10(minimum(nuts.D)) - 0.15,
                     log10(maximum(nuts.D)) + 0.35; length=200)
tfit = 10.0 .^ (a .+ b .* log10.(Dfit))
lines!(ax, Dfit, tfit; color=(:gray40, 0.9), linestyle=:dash, linewidth=2.5,
       label="power-law fit:  t ≈ $(round(10^a, digits=2)) · D^$(round(b, digits=2))")

# Reference slopes for intuition — t ∝ D^1 and t ∝ D^2 through mid-range anchor
Dref = nuts.D[3]; tref = nuts.total_time_s[3]
t_lin = tref .* (Dfit ./ Dref) .^ 1.0
t_quad = tref .* (Dfit ./ Dref) .^ 2.0
lines!(ax, Dfit, t_lin;  color=(:royalblue, 0.5), linestyle=:dot, linewidth=1.8,
       label="slope 1  (t ∝ D)")
lines!(ax, Dfit, t_quad; color=(:firebrick, 0.5), linestyle=:dot, linewidth=1.8,
       label="slope 2  (t ∝ D²)")

# Data points
lines!(ax, nuts.D, nuts.total_time_s; color=:seagreen, linewidth=3.5,
       label="NUTS measured")
scatter!(ax, nuts.D, nuts.total_time_s; color=:seagreen, marker=:diamond,
         markersize=22, strokecolor=:black, strokewidth=1)

# Annotate values next to each marker.
# D=25 label goes ABOVE-LEFT, D=65 label goes BELOW-RIGHT so they don't collide.
for (i, r) in enumerate(eachrow(nuts))
    label = "D=$(r.D):  $(round(r.total_time_s, digits=1)) s"
    if r.D == 25
        text!(ax, r.D, r.total_time_s;
              text = "$label  ", fontsize=13, color=:black,
              align = (:right, :bottom), offset = (-10, 10))
    elseif r.D == 65
        text!(ax, r.D, r.total_time_s;
              text = "  $label", fontsize=13, color=:black,
              align = (:left, :top), offset = (10, -10))
    else
        text!(ax, r.D, r.total_time_s;
              text = "  $label", fontsize=13, color=:black,
              align = (:left, :bottom), offset = (6, 6))
    end
end

xlims!(ax, minimum(Dfit), maximum(Dfit))
ylims!(ax, 0.6 * minimum(nuts.total_time_s), 2.0 * maximum(nuts.total_time_s))

axislegend(ax; position=:lt, labelsize=12, framevisible=true,
           backgroundcolor=(:white, 0.85))

save(joinpath(RESULTS_DIR, "nuts_time_vs_D.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "nuts_time_vs_D.pdf"), fig)
println("Saved nuts_time_vs_D.{png,pdf}")
println("Power-law fit: t = $(round(10^a, digits=3)) × D^$(round(b, digits=3))")
show(stdout, nuts[:, [:D, :min_ess, :max_rhat, :total_time_s]]; allrows=true)
println()
