# w-only WIS vs wall-clock time — the Pareto view.
# One scatter per D panel; curves within a panel connect N=10→50→250 for each method.
# Lower-left = better (less time, smaller WIS).

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using CSV, DataFrames
using Statistics

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const CSV_PATH    = joinpath(RESULTS_DIR, "pareto_wis.csv")

df = CSV.read(CSV_PATH, DataFrame)
filter!(r -> r.status == "OK", df)
sort!(df, [:method, :D, :N])

D_VALUES = sort(unique(df.D))
METHODS  = ["Gibbs", "VMP-dynamic"]

method_color = Dict("Gibbs" => :crimson, "VMP-dynamic" => :darkorange)
method_mark  = Dict("Gibbs" => :circle,  "VMP-dynamic" => :diamond)

fig = Figure(size = (2000, 500), fontsize = 14)
Label(fig[0, 1:length(D_VALUES)+1],
      "w-only WIS vs wall-clock time  —  one panel per D  —  lines connect N=10 → 50 → 250 within each method.\n" *
      "Lower-left = better (fast and accurate).",
      fontsize=15, font=:bold)

for (col, D) in enumerate(D_VALUES)
    ax = Axis(fig[1, col];
              title = "D = $D",
              xlabel = "wall-clock time (s)  — log",
              ylabel = col == 1 ? "w-only WIS  — log" : "",
              xscale = log10, yscale = log10)

    for m in METHODS
        sub = sort(filter(r -> r.method == m && r.D == D, df), :N)
        isempty(sub) && continue
        lines!(ax, sub.total_time_s, sub.wis_w;
               color = method_color[m], linewidth = 2.5, label = m)
        scatter!(ax, sub.total_time_s, sub.wis_w;
                 color = method_color[m], marker = method_mark[m],
                 markersize = 14, strokecolor = :black, strokewidth = 1)
        # Annotate each point with its N value
        for r in eachrow(sub)
            text!(ax, r.total_time_s, r.wis_w;
                  text = "  N=$(r.N)",
                  fontsize = 10, color = method_color[m],
                  align = (:left, :center), offset = (6, 0))
        end
    end
    # Legend only on the first panel so it doesn't clutter
    col == 1 && axislegend(ax; position = :rt, labelsize = 11)
end

save(joinpath(RESULTS_DIR, "wis_vs_time.png"), fig; px_per_unit=2)
save(joinpath(RESULTS_DIR, "wis_vs_time.pdf"), fig)
println("Saved wis_vs_time.{png,pdf}")

# Print the 30-row table sorted by D, N, method for a quick read
println("\nWIS vs compute (per (method, N, D)):")
show(stdout, sort(df[:, [:method, :N, :D, :wis_w, :total_time_s]], [:D, :N, :method]);
     allrows=true)
println()
