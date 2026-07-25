#!/usr/bin/env julia
#
# Ablations for the GP-SSM forecasting baseline (see gp_ssm_ett_forecast.jl):
#   fixed : notebook hyperparameters (l=1, σ²=1, var_noise=0.04) — reference
#   grid  : (l, var_noise) selected by MSE of h-step forecasts on the validation split
#   fe    : (l, σ², var_noise) fitted by free-energy (−log evidence) minimization
#           on the last N_FIT points of the training region
#   qp    : quasi-periodic kernel — Matern-3/2 + two damped harmonic resonators
#           with a 24h period — all hyperparameters fitted by free energy
#
# Protocol is identical to gp_ssm_ett_forecast.jl: condition on the previous
# SEQ_LEN hourly OT observations, predict the value HORIZON steps ahead.
#
# Run: julia --project=. scripts/gp_ssm_ett_ablations.jl
# Env: GP_SSM_STRIDE=k  evaluate every k-th test origin
#      GP_SSM_FAST=1    tiny smoke-test configuration

using RxInfer
using LinearAlgebra
using Statistics
using Printf
using CSV, DataFrames
using CairoMakie
using ProgressMeter
using Optim
using ProbabilisticEnsembling

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const OUT_DIR = joinpath(@__DIR__, "..", "results", "gp_ssm")

parse_list(env, default) =
    haskey(ENV, env) ? strip.(split(ENV[env], ",")) : default

const DATASETS = String.(parse_list("GP_SSM_DATASETS", ["ETTh1", "ETTh2"]))
const SEQ_LEN = 96
# ETT benchmark horizons; override with GP_SSM_HORIZONS=96,192 to run/resume a subset
const HORIZONS = parse.(Int, parse_list("GP_SSM_HORIZONS", ["96", "192", "336", "720"]))
const TIME_UNIT_HOURS = 24.0 # hours per model time unit (lengthscales are in days)
const PERIOD = 24.0 / TIME_UNIT_HOURS # daily cycle for the quasi-periodic kernel
const JITTER = 1e-10
const PLOT_ORIGINS = 300

const FAST = get(ENV, "GP_SSM_FAST", "0") == "1"
const TEST_STRIDE = parse(Int, get(ENV, "GP_SSM_STRIDE", FAST ? "200" : "1"))
const VAL_ORIGINS = FAST ? 20 : 150   # validation origins for the grid search
const N_FIT = FAST ? 240 : 1000      # train points for free-energy fitting
const FE_ITERS = FAST ? 15 : 200     # Nelder-Mead iterations

const GRID_L = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
const GRID_NOISE = (0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0)

@model function gp_regression(y, P, A, Q, H, var_noise)
    f_prev ~ MvNormal(μ = zeros(length(H)), Σ = P) # initial state
    for i in eachindex(y)
        f[i] ~ MvNormal(μ = A[i] * f_prev, Σ = Q[i])
        y[i] ~ Normal(μ = dot(H, f[i]), var = var_noise)
        f_prev = f[i]
    end
end

# ---------------------------------------------------------------------------
# Kernels in SDE / state-space form
# ---------------------------------------------------------------------------

function matern_ssm(ν, l, σ²)
    if ν == 3 // 2
        λ = sqrt(3) / l
        F = [0.0 1.0; -λ^2 -2λ]
        H = [1.0, 0.0]
        P∞ = [σ² 0.0; 0.0 λ^2*σ²]
    elseif ν == 5 // 2
        λ = sqrt(5) / l
        F = [0.0 1.0 0.0; 0.0 0.0 1.0; -λ^3 -3λ^2 -3λ]
        L = [0.0, 0.0, 1.0]
        H = [1.0, 0.0, 0.0]
        Qc = 16 / 3 * σ² * λ^5
        Id = diageye(3)
        vec_P = inv(kron(Id, F) + kron(F, Id)) * vec(-L * Qc * L')
        P∞ = Matrix(Symmetric(reshape(vec_P, 3, 3)))
    else
        error("Unsupported Matern order ν = $(ν)")
    end
    return (F = F, H = H, P∞ = P∞)
end

# Damped harmonic resonator: stationary covariance σ² exp(-τ/l_damp) cos(ωτ)
function resonator_ssm(ω, l_damp, σ²)
    F = [-1/l_damp -ω; ω -1/l_damp]
    H = [1.0, 0.0]
    P∞ = [σ² 0.0; 0.0 σ²]
    return (F = F, H = H, P∞ = P∞)
end

# Sum of kernels = block-diagonal state space with concatenated emission
function combine_ssm(comps...)
    F = Matrix(cat((c.F for c in comps)...; dims = (1, 2)))
    H = vcat((c.H for c in comps)...)
    P∞ = Matrix(cat((c.P∞ for c in comps)...; dims = (1, 2)))
    return (F = F, H = H, P∞ = P∞)
end

function transition_matrices(F, P∞, Δt)
    d = size(F, 1)
    A = [exp(F * δ) for δ in Δt]
    Q = [Matrix(Symmetric(P∞ - a * P∞ * a')) + JITTER * diageye(d) for a in A]
    return A, Q
end

# ---------------------------------------------------------------------------
# Forecasting and evidence
# ---------------------------------------------------------------------------

function forecast_gp_ssm(
    y_windows::AbstractMatrix{Float64},
    ssm,
    var_noise,
    horizon;
    progress = true,
)
    Δt = [fill(1.0 / TIME_UNIT_HOURS, SEQ_LEN); horizon / TIME_UNIT_HOURS]
    A, Q = transition_matrices(ssm.F, ssm.P∞, Δt)

    n = size(y_windows, 2)
    means = Vector{Float64}(undef, n)
    vars = Vector{Float64}(undef, n)
    prog = ProgressMeter.Progress(n; desc = "  forecasting", enabled = progress)
    for j = 1:n
        y_data = Vector{Union{Float64,Missing}}(undef, SEQ_LEN + 1)
        y_data[1:SEQ_LEN] .= @view y_windows[:, j]
        y_data[end] = missing
        result = infer(
            model = gp_regression(
                P = ssm.P∞,
                A = A,
                Q = Q,
                H = ssm.H,
                var_noise = var_noise,
            ),
            data = (y = y_data,),
        )
        # Predictive from the state posterior; RxInfer 5.2's `predictions` field
        # reports only var_noise and drops HᵀΣH.
        f_pred = result.posteriors[:f][end]
        means[j] = dot(ssm.H, mean(f_pred))
        vars[j] = dot(ssm.H, cov(f_pred) * ssm.H) + var_noise
        j % 1000 == 0 && GC.gc() # keep memory bounded over the full sweep
        ProgressMeter.next!(prog)
    end
    return means, vars
end

# Bethe free energy of the fully observed chain = -log evidence (exact for trees)
function neg_log_evidence(y_seg::Vector{Float64}, ssm, var_noise)
    Δt = fill(1.0 / TIME_UNIT_HOURS, length(y_seg))
    A, Q = transition_matrices(ssm.F, ssm.P∞, Δt)
    result = infer(
        model = gp_regression(P = ssm.P∞, A = A, Q = Q, H = ssm.H, var_noise = var_noise),
        data = (y = y_seg,),
        free_energy = true,
        options = (limit_stack_depth = 100,), # deep chain overflows the stack otherwise
    )
    return last(result.free_energy)
end

# ---------------------------------------------------------------------------
# Hyperparameter fitting
# ---------------------------------------------------------------------------

function grid_search(ν, y_windows_val, y_val, horizon)
    best = nothing
    for l in GRID_L, vn in GRID_NOISE
        ssm = matern_ssm(ν, l, 1.0)
        m, _ = forecast_gp_ssm(y_windows_val, ssm, vn, horizon; progress = false)
        score = mse(m, y_val)
        if best === nothing || score < best.score
            best = (l = l, var_noise = vn, score = score)
        end
    end
    return best
end

function fit_free_energy(build, x0, y_seg)
    obj = x -> begin
        any(abs.(x) .> 12) && return Inf # keep exp(x) in a sane range
        try
            ssm, vn = build(x)
            fe = neg_log_evidence(y_seg, ssm, vn)
            return isfinite(fe) ? fe : Inf
        catch
            return Inf
        end
    end
    res = Optim.optimize(obj, x0, NelderMead(), Optim.Options(iterations = FE_ITERS))
    isfinite(Optim.minimum(res)) ||
        @warn "free-energy fit failed: objective was Inf everywhere, returning x0"
    return Optim.minimizer(res), Optim.minimum(res)
end

build_matern(ν) = x -> (matern_ssm(ν, exp(x[1]), exp(x[2])), exp(x[3]))

# x = log.([l_m, σ²_m, l_damp, σ²_1, σ²_2, var_noise]); period is fixed at 24h
build_qp =
    x -> (
        combine_ssm(
            matern_ssm(3 // 2, exp(x[1]), exp(x[2])),
            resonator_ssm(2π / PERIOD, exp(x[3]), exp(x[4])),
            resonator_ssm(4π / PERIOD, exp(x[3]), exp(x[5])),
        ),
        exp(x[6]),
    )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

gaussian_nlpd(m, v, y) = 0.5 * (log(2π * v) + (y - m)^2 / v)

function prepare(dataset::String, horizon::Int)
    Xmat, feat_cols = load_ett(joinpath(DATA_DIR, dataset * ".csv"))
    ot_idx = ProbabilisticEnsembling.find_column_index(feat_cols, "OT")

    X3, Y2 = make_sequences(Xmat; seq_len = SEQ_LEN, horizon = horizon)
    Xtr, _, Xva, Yva, Xte, Yte = train_val_test_split(X3, Y2; ratios = (0.6, 0.2, 0.2))

    scaler = fit_scaler(Xtr)
    μ_ot = Float64(scaler.μ[ot_idx])
    σ_ot = Float64(scaler.σ[ot_idx])

    val_origins =
        unique(round.(Int, range(1, size(Xva, 3), length = min(VAL_ORIGINS, size(Xva, 3)))))
    test_origins = 1:TEST_STRIDE:size(Xte, 3)

    scale_win(X, idx) = (Float64.(X[ot_idx, :, idx]) .- μ_ot) ./ σ_ot
    scale_tgt(Y, idx) = (vec(Float64.(Y[ot_idx, idx])) .- μ_ot) ./ σ_ot

    # Contiguous scaled series from the training region for free-energy fitting.
    # Anchored to a fixed 60% split of the raw timeline so it is horizon-independent
    # (the free-energy objective does not depend on the forecast horizon).
    T = size(Xmat, 2)
    n_tr_time = round(Int, 0.6 * T)
    ot_series = (Float64.(Xmat[ot_idx, :]) .- μ_ot) ./ σ_ot
    y_fit = ot_series[max(1, n_tr_time - N_FIT + 1):n_tr_time]

    return (
        w_val = scale_win(Xva, val_origins),
        y_val = scale_tgt(Yva, val_origins),
        w_test = scale_win(Xte, test_origins),
        y_test = scale_tgt(Yte, test_origins),
        y_fit = y_fit,
        μ = μ_ot,
        σ = σ_ot,
    )
end

function metric_rows(dataset, config, params_str, horizon, m_s, v_s, y_s, μ, σ)
    y_o = y_s .* σ .+ μ
    m_o = m_s .* σ .+ μ
    v_o = v_s .* σ^2
    row(scale, ŷ, y, nlpd) = (
        dataset = dataset,
        config = config,
        horizon = horizon,
        scale = scale,
        mse = mse(ŷ, y),
        mae = mae(ŷ, y),
        rmse = rmse(ŷ, y),
        r2 = r2(ŷ, y),
        nlpd = nlpd,
        params = params_str,
    )
    return [
        row("scaled", m_s, y_s, mean(gaussian_nlpd.(m_s, v_s, y_s))),
        row("original", m_o, y_o, mean(gaussian_nlpd.(m_o, v_o, y_o))),
    ]
end

function fixed_configs()
    configs = NamedTuple[]
    for (kname, ν) in (("M32", 3 // 2), ("M52", 5 // 2))
        push!(
            configs,
            (
                name = "fixed($kname)",
                ssm = matern_ssm(ν, 1.0, 1.0),
                var_noise = 0.04,
                params = "l=1 σ²=1 vn=0.04",
            ),
        )
    end
    return configs
end

# Free-energy fits optimize the 1-step train evidence, which is horizon-independent,
# so these are computed once per dataset and reused across all horizons.
function fe_configs(dataset, y_fit)
    configs = NamedTuple[]
    for (kname, ν) in (("M32", 3 // 2), ("M52", 5 // 2))
        t = @elapsed x, fe =
            fit_free_energy(build_matern(ν), log.([1.0, 1.0, 0.05]), y_fit)
        l, σ², vn = exp.(x)
        @printf(
            "%s fe(%s): l=%.3f σ²=%.3f vn=%.4f (fe=%.2f) [%.1fs]\n",
            dataset, kname, l, σ², vn, fe, t
        )
        ssm, var_noise = build_matern(ν)(x)
        push!(
            configs,
            (
                name = "fe($kname)",
                ssm = ssm,
                var_noise = var_noise,
                params = @sprintf("l=%.3f σ²=%.3f vn=%.4f", l, σ², vn),
            ),
        )
    end

    t = @elapsed x, fe =
        fit_free_energy(build_qp, log.([2.0, 0.5, 7.0, 0.3, 0.2, 0.05]), y_fit)
    lm, σm, ld, σ1, σ2, vn = exp.(x)
    @printf(
        "%s fe(QP): l_m=%.3f σ²_m=%.3f l_damp=%.2f σ²_1=%.3f σ²_2=%.3f vn=%.4f (fe=%.2f) [%.1fs]\n",
        dataset, lm, σm, ld, σ1, σ2, vn, fe, t
    )
    ssm, var_noise = build_qp(x)
    push!(
        configs,
        (
            name = "fe(QP+M32)",
            ssm = ssm,
            var_noise = var_noise,
            params = @sprintf(
                "l_m=%.3f σ²_m=%.3f l_damp=%.2f σ²_1=%.3f σ²_2=%.3f vn=%.4f",
                lm, σm, ld, σ1, σ2, vn
            ),
        ),
    )
    return configs
end

# Grid search selects (l, var_noise) by h-step validation MSE, so it re-runs per horizon.
function grid_configs(dataset, prep, horizon)
    configs = NamedTuple[]
    for (kname, ν) in (("M32", 3 // 2), ("M52", 5 // 2))
        t = @elapsed best = grid_search(ν, prep.w_val, prep.y_val, horizon)
        @printf(
            "%s h%d grid(%s): l=%.2f vn=%.3f (val mse=%.4f) [%.1fs]\n",
            dataset, horizon, kname, best.l, best.var_noise, best.score, t
        )
        push!(
            configs,
            (
                name = "grid($kname)",
                ssm = matern_ssm(ν, best.l, 1.0),
                var_noise = best.var_noise,
                params = @sprintf("l=%.2f σ²=1 vn=%.3f", best.l, best.var_noise),
            ),
        )
    end
    return configs
end

function plot_ablation(dataset, horizon, y_true, forecasts, path)
    n = min(PLOT_ORIGINS, length(y_true))
    x = 1:n
    fig = Figure(size = (1100, 460))
    ax = Axis(
        fig[1, 1],
        title = "$(dataset): GP-SSM ablations, $(horizon)-step-ahead forecasts (scaled OT)",
        xlabel = "test origin",
        ylabel = "OT (scaled)",
    )
    lines!(ax, x, y_true[1:n], color = :black, linewidth = 2, label = "truth")
    colors = Makie.wong_colors()
    for (i, (name, fc)) in enumerate(forecasts)
        lines!(
            ax,
            x,
            fc.mean[1:n],
            color = colors[mod1(i, length(colors))],
            linewidth = 1.5,
            label = name,
        )
    end
    axislegend(ax, position = :rt, nbanks = 2)
    save(path, fig)
end

function main()
    mkpath(OUT_DIR)
    # Incremental CSV: append each (dataset, horizon) block as it completes so a
    # killed run keeps finished results. Resume with GP_SSM_DATASETS / GP_SSM_HORIZONS.
    csv_path = joinpath(OUT_DIR, "gp_ssm_ett_ablations_all_horizons.csv")
    wrote_header = isfile(csv_path)

    all_rows = NamedTuple[]
    for dataset in DATASETS
        println("== $(dataset) ==")
        # Fixed and free-energy configs are horizon-independent — build once per dataset.
        prep0 = prepare(dataset, first(HORIZONS))
        base_fixed = fixed_configs()
        base_fe = fe_configs(dataset, prep0.y_fit)

        for horizon in HORIZONS
            prep = horizon == first(HORIZONS) ? prep0 : prepare(dataset, horizon)
            configs = vcat(base_fixed, grid_configs(dataset, prep, horizon), base_fe)

            block_rows = NamedTuple[]
            forecasts = Vector{Pair{String,NamedTuple}}()
            for c in configs
                println("$(dataset) h$(horizon) $(c.name): test (n=$(length(prep.y_test)))")
                m_s, v_s = forecast_gp_ssm(prep.w_test, c.ssm, c.var_noise, horizon)
                append!(
                    block_rows,
                    metric_rows(
                        dataset,
                        c.name,
                        c.params,
                        horizon,
                        m_s,
                        v_s,
                        prep.y_test,
                        prep.μ,
                        prep.σ,
                    ),
                )
                push!(forecasts, c.name => (mean = m_s, var = v_s))
            end
            CSV.write(csv_path, DataFrame(block_rows); append = wrote_header)
            wrote_header = true
            append!(all_rows, block_rows)
            plot_ablation(
                dataset,
                horizon,
                prep.y_test,
                forecasts,
                joinpath(OUT_DIR, "$(dataset)_h$(horizon)_ablations.png"),
            )
            GC.gc()
        end
    end

    println()
    println("GP-SSM ablations, horizons=$(HORIZONS), OT, test stride=$(TEST_STRIDE)")
    println("scaled = directly comparable to current ensemble metrics")
    println()
    for r in all_rows
        r.scale == "scaled" || continue
        @printf(
            "%s h%-3d %-11s mse=%.4f mae=%.4f rmse=%.4f r2=%8.4f nlpd=%8.4f\n",
            r.dataset, r.horizon, r.config, r.mse, r.mae, r.rmse, r.r2, r.nlpd
        )
    end
    println("\nSaved: $(csv_path) and per-(dataset,horizon) PNGs")
end

main()
