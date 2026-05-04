# Renders marginals_d2 with Gibbs + NUTS + VMP (dynamic and noisy).
# Run from MAIN project env — uses CairoMakie, KernelDensity, and Distributions
# from the top-level Project.toml (same as run_vmp_d2).
#
# Required inputs (produced by earlier scripts):
#   results/gibbs_samples_d2.csv       — from save_chains_d2.jl
#   results/nuts_samples_d2.csv        — from save_chains_d2.jl
#   results/vmp_posteriors_d2.jld2     — from run_vmp_d2.jl
#   results/truth_d2.csv               — from save_chains_d2.jl

using Pkg
Pkg.activate("/Users/mykola/repos/probabilistic_ensemble_forecasting")

using CairoMakie
using CSV, DataFrames
using JLD2
using Statistics
using LinearAlgebra
using KernelDensity
using Distributions

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

const N_EXPERTS = 7
const D = 2

# ============================================================
# Load inputs
# ============================================================
gibbs_df = CSV.read(joinpath(RESULTS_DIR, "gibbs_samples_d2.csv"), DataFrame)
nuts_df  = CSV.read(joinpath(RESULTS_DIR, "nuts_samples_d2.csv"),  DataFrame)
truth_df = CSV.read(joinpath(RESULTS_DIR, "truth_d2.csv"),         DataFrame)
vmp_blob = JLD2.load(joinpath(RESULTS_DIR, "vmp_posteriors_d2.jld2"), "vmp")

vmp_dyn   = get(vmp_blob, :dynamic, nothing)
vmp_noisy = nothing   # intentionally dropped — noisy is a different model
println("VMP dynamic available: ", vmp_dyn !== nothing)

function truth_for(expert::Int, key::String)
    row = filter(r -> r.expert == expert && r.param == key, truth_df)
    return only(row.value)
end

# ============================================================
# Helpers to get a VMP marginal pdf for a given (expert, param)
# ============================================================
struct VMPMarginal
    pdf::Function        # x -> density
    mean::Float64
    support_min::Float64
    support_max::Float64
end

function vmp_marginal(entry::Dict, param::Symbol, dim::Int)
    info = entry[param]
    if info[:family] == "MvNormal"
        μ = info[:mean][dim]
        σ = sqrt(info[:cov][dim, dim])
        d = Distributions.Normal(μ, σ)
        return VMPMarginal(x -> pdf(d, x), μ, μ - 4σ, μ + 4σ)
    elseif info[:family] == "Gamma"
        α = info[:shape]; r = info[:rate]
        d = Distributions.Gamma(α, 1/r)  # scale = 1/rate
        μ = α / r
        σ = sqrt(α) / r
        return VMPMarginal(x -> pdf(d, x), μ,
                           max(1e-12, μ - 4σ), μ + 4σ)
    else
        error("Unsupported family $(info[:family])")
    end
end

# ============================================================
# Figure: 4 rows × N_EXPERTS cols, same layout as marginals_d2
# ============================================================
param_specs = [
    (key=:w, dim=1, label_fmt = i -> "w[1, $i]  (intercept)", truth_key="w[1]"),
    (key=:w, dim=2, label_fmt = i -> "w[2, $i]  (slope)",     truth_key="w[2]"),
    (key=:τ, dim=1, label_fmt = i -> "τ[$i]  (precision)",    truth_key="τ"),
    (key=:β, dim=1, label_fmt = i -> "β[$i]  (γ-rate)",       truth_key="β"),
]

fig = Figure(size=(1800, 1050), fontsize=11)

Label(fig[0, 1:N_EXPERTS],
      "Posterior marginals at d=2  —  Gibbs (blue) | NUTS collapsed (green) | VMP dynamic (orange)\n" *
      "Red dashed = ground truth  (τ_true ∈ [0.5, 5])",
      fontsize=14, font=:bold)

function kde_curve!(ax, samples, color; strokelinewidth = 1.2, alpha_fill = 0.35)
    k = kde(samples)
    band!(ax, k.x, zeros(length(k.x)), k.density; color=(color, alpha_fill))
    lines!(ax, k.x, k.density; color=color, linewidth=strokelinewidth)
end

for i in 1:N_EXPERTS
    for (row, spec) in enumerate(param_specs)
        ax = Axis(fig[row, i]; xlabel=spec.label_fmt(i), xlabelsize=10)

        # --- Gather column names in the CSV ---
        if spec.key == :w
            col = Symbol("w_e$(i)_d$(spec.dim)")
        elseif spec.key == :τ
            col = Symbol("tau_e$i")
        else
            col = Symbol("beta_e$i")
        end

        gibbs_samples = collect(skipmissing(gibbs_df[!, col]))
        nuts_samples  = collect(skipmissing(nuts_df[!,  col]))

        # --- Gibbs density (blue) ---
        kde_curve!(ax, gibbs_samples, :steelblue)

        # --- NUTS histogram (green) overlaid at same scale ---
        hist!(ax, nuts_samples; bins=30, normalization=:pdf,
              color=(:seagreen, 0.35), strokewidth=0.4, strokecolor=:seagreen)

        # --- VMP dynamic (orange line) ---
        if vmp_dyn !== nothing
            vm = vmp_marginal(vmp_dyn[i], spec.key, spec.dim)
            xs_v = range(vm.support_min, vm.support_max; length=200)
            lines!(ax, xs_v, vm.pdf.(xs_v); color=:darkorange, linewidth=1.8)
        end

        # --- VMP noisy (purple dashed) ---
        if vmp_noisy !== nothing
            vn = vmp_marginal(vmp_noisy[i], spec.key, spec.dim)
            xs_n = range(vn.support_min, vn.support_max; length=200)
            lines!(ax, xs_n, vn.pdf.(xs_n); color=:purple, linewidth=1.5,
                   linestyle=:dash)
        end

        # --- Ground truth ---
        tru = truth_for(i, spec.truth_key)
        vlines!(ax, [tru]; color=:red, linestyle=:dash, linewidth=1.5)

        hideydecorations!(ax; grid=false)

        if i == 1
            row_label = Dict(1=>"w[1,·] intercept", 2=>"w[2,·] slope",
                             3=>"τ precision", 4=>"β γ-rate")[row]
            Label(fig[row, 0], row_label;
                  rotation=π/2, fontsize=11, font=:bold)
        end
        if row == 1
            Label(fig[row - 1, i, Top()], "Expert $i";
                  fontsize=10, font=:bold, padding=(0, 0, 2, 2))
        end
    end
end

# Legend
leg_elems = [
    PolyElement(color=(:steelblue, 0.35), strokecolor=:steelblue),
    PolyElement(color=(:seagreen, 0.35), strokecolor=:seagreen),
    LineElement(color=:darkorange, linewidth=2),
    LineElement(color=:red, linestyle=:dash, linewidth=1.5),
]
leg_labels = [
    "Gibbs reference",
    "NUTS collapsed (Enzyme)",
    "VMP dynamic",
    "Ground truth",
]
Legend(fig[5, 1:N_EXPERTS], leg_elems, leg_labels;
       orientation=:horizontal, tellwidth=false, tellheight=true, framevisible=false)
rowsize!(fig.layout, 5, Relative(0.05))

out_png = joinpath(RESULTS_DIR, "marginals_d2_allmethods.png")
out_pdf = joinpath(RESULTS_DIR, "marginals_d2_allmethods.pdf")
save(out_png, fig; px_per_unit=3)
save(out_pdf, fig)
println("Saved $(out_png)")
println("Saved $(out_pdf)")
