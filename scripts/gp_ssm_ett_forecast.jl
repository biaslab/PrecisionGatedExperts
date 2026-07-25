#!/usr/bin/env julia
#
# GP regression via a state-space model (Matern kernels as linear SDEs) used as a
# forecasting baseline on ETTh1/ETTh2. Follows the RxInfer "GP Regression by SSM"
# example: condition on the previous SEQ_LEN observations of OT and predict the
# single value HORIZON steps ahead, matching the evaluation protocol of
# scripts/eval_ot_only_models.jl (6:2:2 split, metrics on scaled and original OT).
#
# Run: julia --project=. scripts/gp_ssm_ett_forecast.jl

using RxInfer
using LinearAlgebra
using Statistics
using Printf
using CSV, DataFrames
using CairoMakie
using ProgressMeter
using ProbabilisticEnsembling

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const OUT_DIR = joinpath(@__DIR__, "..", "results", "gp_ssm")

const DATASETS = ["ETTh1", "ETTh2"]
const SEQ_LEN = 96
const HORIZON = 96
const θ = [1.0, 1.0]          # [lengthscale l, signal variance σ²], fixed as in the notebook
const VAR_NOISE = 0.04        # observation noise variance, fixed as in the notebook
const TIME_UNIT_HOURS = 24.0  # hours per model time unit, so l = 1 spans one day
const ORIGIN_STRIDE = parse(Int, get(ENV, "GP_SSM_STRIDE", "1")) # every k-th test origin
const PLOT_ORIGINS = 300      # test origins shown in the figure

@model function gp_regression(y, P, A, Q, H, var_noise)
    f_prev ~ MvNormal(μ = zeros(length(H)), Σ = P) # initial state
    for i in eachindex(y)
        f[i] ~ MvNormal(μ = A[i] * f_prev, Σ = Q[i])
        y[i] ~ Normal(μ = dot(H, f[i]), var = var_noise)
        f_prev = f[i]
    end
end

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

function transition_matrices(F, P∞, Δt)
    A = [exp(F * δ) for δ in Δt]
    Q = [Matrix(Symmetric(P∞ - a * P∞ * a')) for a in A]
    return A, Q
end

gaussian_nlpd(m, v, y) = 0.5 * (log(2π * v) + (y - m)^2 / v)

function forecast_gp_ssm(y_windows::AbstractMatrix{Float64}, ssm)
    # Δt in model time units: SEQ_LEN observed hourly steps (the first transition
    # leaves the stationary prior, so its Δt is arbitrary), then one HORIZON-hour
    # jump to the prediction — exact for a linear SDE, no intermediate states needed.
    Δt = [fill(1.0 / TIME_UNIT_HOURS, SEQ_LEN); HORIZON / TIME_UNIT_HOURS]
    A, Q = transition_matrices(ssm.F, ssm.P∞, Δt)

    n = size(y_windows, 2)
    means = Vector{Float64}(undef, n)
    vars = Vector{Float64}(undef, n)
    @showprogress desc = "  forecasting" for j = 1:n
        y_data = Vector{Union{Float64,Missing}}(undef, SEQ_LEN + 1)
        y_data[1:SEQ_LEN] .= @view y_windows[:, j]
        y_data[end] = missing
        result = infer(
            model = gp_regression(
                P = ssm.P∞,
                A = A,
                Q = Q,
                H = ssm.H,
                var_noise = VAR_NOISE,
            ),
            data = (y = y_data,),
        )
        # Predictive p(y) from the state posterior at the prediction node; the
        # `predictions` field in RxInfer 5.2 reports only var_noise, dropping HᵀΣH.
        f_pred = result.posteriors[:f][end]
        means[j] = dot(ssm.H, mean(f_pred))
        vars[j] = dot(ssm.H, cov(f_pred) * ssm.H) + VAR_NOISE
    end
    return means, vars
end

function metric_row(dataset, kernel, scale, ŷ, y, nlpd)
    return (
        dataset = dataset,
        kernel = kernel,
        horizon = HORIZON,
        scale = scale,
        mse = mse(ŷ, y),
        mae = mae(ŷ, y),
        rmse = rmse(ŷ, y),
        r2 = r2(ŷ, y),
        mape = mape(ŷ, y),
        smape = smape(ŷ, y),
        nlpd = nlpd,
    )
end

function evaluate_dataset(dataset::String)
    Xmat, feat_cols = load_ett(joinpath(DATA_DIR, dataset * ".csv"))
    ot_idx = ProbabilisticEnsembling.find_column_index(feat_cols, "OT")

    X3, Y2 = make_sequences(Xmat; seq_len = SEQ_LEN, horizon = HORIZON)
    Xtr, _, _, _, Xte, Yte = train_val_test_split(X3, Y2; ratios = (0.6, 0.2, 0.2))

    scaler = fit_scaler(Xtr)
    Xte_s = scale_inputs(scaler, Xte)
    Yte_s = scale_targets(scaler, Yte)

    origins = 1:ORIGIN_STRIDE:size(Xte_s, 3)
    y_windows = Float64.(Xte_s[ot_idx, :, origins])
    y_true_s = vec(Float64.(Yte_s[ot_idx, origins]))

    μ_ot = Float64(scaler.μ[ot_idx])
    σ_ot = Float64(scaler.σ[ot_idx])
    y_true = y_true_s .* σ_ot .+ μ_ot

    rows = NamedTuple[]
    forecasts = Dict{String,NamedTuple}()
    for (kernel, ν) in (("Matern32", 3 // 2), ("Matern52", 5 // 2))
        println("dataset=$(dataset) kernel=$(kernel) n_origins=$(length(origins))")
        ssm = matern_ssm(ν, θ[1], θ[2])
        m_s, v_s = forecast_gp_ssm(y_windows, ssm)

        nlpd_s = mean(gaussian_nlpd.(m_s, v_s, y_true_s))
        m_o = m_s .* σ_ot .+ μ_ot
        v_o = v_s .* σ_ot^2
        nlpd_o = mean(gaussian_nlpd.(m_o, v_o, y_true))

        push!(rows, metric_row(dataset, kernel, "scaled", m_s, y_true_s, nlpd_s))
        push!(rows, metric_row(dataset, kernel, "original", m_o, y_true, nlpd_o))
        forecasts[kernel] = (mean = m_s, std = sqrt.(v_s))
    end
    return rows, (y_true = y_true_s, forecasts = forecasts)
end

function plot_forecasts(dataset, plotdata, path)
    n = min(PLOT_ORIGINS, length(plotdata.y_true))
    x = 1:n
    fig = Figure(size = (1000, 420))
    ax = Axis(
        fig[1, 1],
        title = "$(dataset): GP-SSM $(HORIZON)-step-ahead forecasts (scaled OT)",
        xlabel = "test origin",
        ylabel = "OT (scaled)",
    )
    colors = Dict("Matern32" => Makie.wong_colors()[1], "Matern52" => Makie.wong_colors()[2])
    for (kernel, fc) in sort(collect(plotdata.forecasts); by = first)
        m, s = fc.mean[1:n], fc.std[1:n]
        band!(ax, x, m .- s, m .+ s, color = (colors[kernel], 0.2))
        lines!(ax, x, m, color = colors[kernel], linewidth = 1.5, label = kernel)
    end
    lines!(ax, x, plotdata.y_true[1:n], color = :black, linewidth = 1.5, label = "truth")
    axislegend(ax, position = :rt)
    save(path, fig)
end

function main()
    mkpath(OUT_DIR)
    all_rows = NamedTuple[]
    for dataset in DATASETS
        rows, plotdata = evaluate_dataset(dataset)
        append!(all_rows, rows)
        plot_forecasts(dataset, plotdata, joinpath(OUT_DIR, "$(dataset)_h$(HORIZON).png"))
    end

    df = DataFrame(all_rows)
    CSV.write(joinpath(OUT_DIR, "gp_ssm_ett_metrics.csv"), df)

    println()
    println("GP-SSM (Matern) $(HORIZON)-step-ahead OT forecasting, stride=$(ORIGIN_STRIDE)")
    println("scaled = directly comparable to current ensemble metrics")
    println()
    for r in all_rows
        @printf(
            "%s %s [%s] mse=%.6f mae=%.6f rmse=%.6f r2=%.4f nlpd=%.4f\n",
            r.dataset,
            r.kernel,
            r.scale,
            r.mse,
            r.mae,
            r.rmse,
            r.r2,
            r.nlpd,
        )
    end
    println("\nSaved: $(joinpath(OUT_DIR, "gp_ssm_ett_metrics.csv")) and per-dataset PNGs")
end

main()
