#!/usr/bin/env julia
#
# Confidence-interval comparison of the GP-SSM baseline (val-tuned large-l Matern-3/2)
# against the Dynamic ensemble from the paper, using the paper's exact CI recipe:
#
#   CI_half = Z_95 * std(per-test-point term) / sqrt(N_PAPER)
#
# where the per-test-point terms are (a) squared error (mean_pred - y)^2 for MSE and
# (b) Gaussian negative log predictive density for NLL, both on the standardized OT
# target. This matches scripts/build_comparision_table.jl (Z_95, N_PAPER=2881) and
# src/model_zoo/shared_pipeline.jl (mse_std / nll_std over per-test-point terms).
#
# The GP is evaluated on the SAME test windows as the Dynamic model (identical
# seq_len=96, split (0.6,0.2,0.2), and OT target), so point estimates are directly
# comparable and the CI uses the identical formula. Only the winning large-l grid
# config is re-run (its (l, var_noise) are read from the ablation sweep CSV).
#
# Run: julia --project=. scripts/gp_ssm_ci_vs_dynamic.jl

using RxInfer, LinearAlgebra, Statistics, Printf
using ProbabilisticEnsembling

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const SWEEP_CSV = joinpath(@__DIR__, "..", "results", "gp_ssm", "gp_ssm_ett_ablations_all_horizons.csv")
const SEQ_LEN = 96
const HORIZONS = [96, 192, 336, 720]
const TIME_UNIT_HOURS = 24.0
const Z_95 = 1.959963984540054        # matches build_comparision_table.jl
const N_PAPER = 2881                    # matches TEST_SIZE_BY_DATASET (paper convention)

# Dynamic ensemble, standardized OT, read from paper/results_vae_std/dynamic/*.jld2
# (mean, std) per horizon -> CI computed with the same formula below.
const DYNAMIC = Dict(
    "ETTh1" => Dict(
        96  => (mse = 0.12836, mse_std = 0.17875, nll = 0.41203, nll_std = 0.51779),
        192 => (mse = 0.11460, mse_std = 0.17280, nll = 0.37010, nll_std = 0.50990),
        336 => (mse = 0.09640, mse_std = 0.13510, nll = 0.31410, nll_std = 0.40210),
        720 => (mse = 0.11200, mse_std = 0.15030, nll = 0.37630, nll_std = 0.41140),
    ),
    "ETTh2" => Dict(
        96  => (mse = 0.34600, mse_std = 0.42500, nll = 0.93420, nll_std = 0.91500),
        192 => (mse = 0.33620, mse_std = 0.44100, nll = 0.92370, nll_std = 0.97800),
        336 => (mse = 0.35290, mse_std = 0.45200, nll = 0.96120, nll_std = 0.99400),
        720 => (mse = 0.32130, mse_std = 0.43800, nll = 0.86990, nll_std = 0.87900),
    ),
)

@model function gp_regression(y, P, A, Q, H, var_noise)
    f_prev ~ MvNormal(μ = zeros(length(H)), Σ = P)
    for i in eachindex(y)
        f[i] ~ MvNormal(μ = A[i] * f_prev, Σ = Q[i])
        y[i] ~ Normal(μ = dot(H, f[i]), var = var_noise)
        f_prev = f[i]
    end
end

function matern32(l, σ²)
    λ = sqrt(3) / l
    F = [0.0 1.0; -λ^2 -2λ]
    H = [1.0, 0.0]
    P∞ = [σ² 0.0; 0.0 λ^2*σ²]
    return (F = F, H = H, P∞ = P∞)
end

ci_half(std_val) = Z_95 * std_val / sqrt(N_PAPER)

# Read the val-selected (l, var_noise) for grid(M32) from the sweep CSV.
function grid_params(dataset, horizon)
    for line in eachline(SWEEP_CSV)
        parts = split(line, ",")
        length(parts) < 10 && continue
        if parts[1] == dataset && parts[2] == "grid(M32)" &&
           parts[3] == string(horizon) && parts[4] == "scaled"
            p = parts[10]
            l = parse(Float64, match(r"l=([0-9.]+)", p).captures[1])
            vn = parse(Float64, match(r"vn=([0-9.]+)", p).captures[1])
            return (l = l, var_noise = vn)
        end
    end
    return nothing
end

function gp_metrics(dataset, horizon, l, var_noise)
    Xmat, feat_cols = load_ett(joinpath(DATA_DIR, dataset * ".csv"))
    ot_idx = ProbabilisticEnsembling.find_column_index(feat_cols, "OT")
    X3, Y2 = make_sequences(Xmat; seq_len = SEQ_LEN, horizon = horizon)
    Xtr, _, _, _, Xte, Yte = train_val_test_split(X3, Y2; ratios = (0.6, 0.2, 0.2))
    scaler = fit_scaler(Xtr)
    μ = Float64(scaler.μ[ot_idx])
    σ = Float64(scaler.σ[ot_idx])

    y_windows = (Float64.(Xte[ot_idx, :, :]) .- μ) ./ σ
    y_true = (vec(Float64.(Yte[ot_idx, :])) .- μ) ./ σ
    n = length(y_true)

    ssm = matern32(l, 1.0)
    Δt = [fill(1.0 / TIME_UNIT_HOURS, SEQ_LEN); horizon / TIME_UNIT_HOURS]
    A = [exp(ssm.F * δ) for δ in Δt]
    Q = [Matrix(Symmetric(ssm.P∞ - a * ssm.P∞ * a')) for a in A]

    se = Vector{Float64}(undef, n)   # per-point squared error
    nll = Vector{Float64}(undef, n)  # per-point negative log predictive density
    for j = 1:n
        y_data = Vector{Union{Float64,Missing}}(undef, SEQ_LEN + 1)
        y_data[1:SEQ_LEN] .= @view y_windows[:, j]
        y_data[end] = missing
        res = infer(
            model = gp_regression(P = ssm.P∞, A = A, Q = Q, H = ssm.H, var_noise = var_noise),
            data = (y = y_data,),
        )
        f_pred = res.posteriors[:f][end]
        m = dot(ssm.H, mean(f_pred))
        v = dot(ssm.H, cov(f_pred) * ssm.H) + var_noise
        se[j] = (m - y_true[j])^2
        nll[j] = 0.5 * (log(2π * v) + (y_true[j] - m)^2 / v)
        j % 1000 == 0 && GC.gc()
    end
    return (
        n = n,
        mse = mean(se),
        mse_std = std(se),
        nll = mean(nll),
        nll_std = std(nll),
    )
end

function main()
    datasets = length(ARGS) >= 1 ? split(ARGS[1], ",") : ["ETTh1"]
    for dataset in datasets
        println("\n===== $(dataset): Dynamic (paper) vs GP-SSM (Matern-3/2, val-tuned l=128) =====")
        @printf(
            "%-4s | %-26s | %-26s | %-26s | %-26s\n",
            "H", "Dyn MSE", "GP MSE", "Dyn NLL", "GP NLL"
        )
        println(repeat("-", 122))
        for h in HORIZONS
            gp = grid_params(dataset, h)
            gp === nothing && (println("h$h: no GP grid params in CSV yet — skipping"); continue)
            g = gp_metrics(dataset, h, gp.l, gp.var_noise)
            d = DYNAMIC[dataset][h]
            @printf(
                "%-4d | %7.3f ± %-5.3f (n=%4d) | %7.3f ± %-5.3f (n=%4d) | %6.3f ± %-5.3f          | %6.3f ± %-5.3f\n",
                h,
                d.mse, ci_half(d.mse_std), N_PAPER,
                g.mse, ci_half(g.mse_std), g.n,
                d.nll, ci_half(d.nll_std),
                g.nll, ci_half(g.nll_std),
            )
        end
    end
    println("\nCI = $(round(Z_95,digits=3))·std/√$(N_PAPER); std over per-test-point terms; standardized OT.")
end

main()
