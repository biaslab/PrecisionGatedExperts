#!/usr/bin/env julia
#
# Minimal script to generate the GP-SSM row of the comparison table:
# Matern-3/2 GP as a state-space model, large lengthscale l=128 (the val-tuned
# winner), single value h steps ahead on standardized OT. Reports MSE and NLL with
# the paper's 95% CI (1.9599*std/sqrt(2881)) over per-test-point terms.
#
# Run: julia --project=. scripts/gp_ssm_table.jl        # ETTh1 + ETTh2

using RxInfer, LinearAlgebra, Statistics, Printf
using ProbabilisticEnsembling

const L = 128.0                              # lengthscale (days); val-tuned winner
const NOISE = Dict(                          # val-selected observation noise variance
    ("ETTh1", 96) => 0.6, ("ETTh1", 192) => 0.6, ("ETTh1", 336) => 1.0, ("ETTh1", 720) => 0.6,
    ("ETTh2", 96) => 0.3, ("ETTh2", 192) => 0.6, ("ETTh2", 336) => 2.0, ("ETTh2", 720) => 0.3,
)
const Z, N = 1.959963984540054, 2881         # paper CI: z * std / sqrt(N)
ci(s) = Z * s / sqrt(N)

@model function gp(y, P, A, Q, var_noise)
    f_prev ~ MvNormal(μ = [0.0, 0.0], Σ = P)
    for i in eachindex(y)
        f[i] ~ MvNormal(μ = A[i] * f_prev, Σ = Q[i])
        y[i] ~ Normal(μ = dot([1.0, 0.0], f[i]), var = var_noise)
        f_prev = f[i]
    end
end

function gp_row(dataset, h)
    X, cols = load_ett(joinpath(@__DIR__, "..", "data", dataset * ".csv"))
    ot = ProbabilisticEnsembling.find_column_index(cols, "OT")
    X3, Y2 = make_sequences(X; seq_len = 96, horizon = h)
    Xtr, _, _, _, Xte, Yte = train_val_test_split(X3, Y2; ratios = (0.6, 0.2, 0.2))
    s = fit_scaler(Xtr)
    μ, σ = Float64(s.μ[ot]), Float64(s.σ[ot])
    W = (Float64.(Xte[ot, :, :]) .- μ) ./ σ         # (96, n) input windows
    yt = (vec(Float64.(Yte[ot, :])) .- μ) ./ σ      # n targets
    vn = NOISE[(dataset, h)]

    λ = sqrt(3) / L
    F = [0.0 1.0; -λ^2 -2λ]
    P∞ = [1.0 0.0; 0.0 λ^2]
    Δt = [fill(1 / 24, 96); h / 24]                 # final step jumps h hours ahead
    A = [exp(F * d) for d in Δt]
    Q = [Matrix(Symmetric(P∞ - a * P∞ * a')) for a in A]

    n = length(yt)
    se, nll = zeros(n), zeros(n)
    for j = 1:n
        data = Vector{Union{Float64,Missing}}([W[:, j]; missing])
        f = infer(model = gp(P = P∞, A = A, Q = Q, var_noise = vn), data = (y = data,)).posteriors[:f][end]
        m, v = mean(f)[1], cov(f)[1, 1] + vn
        se[j] = (m - yt[j])^2
        nll[j] = 0.5 * (log(2π * v) + (yt[j] - m)^2 / v)
        j % 1000 == 0 && GC.gc()
    end
    return mean(se), std(se), mean(nll), std(nll)
end

for dataset in ["ETTh1", "ETTh2"]
    println("\n$dataset  (GP-SSM Matern-3/2, l=$L)")
    for h in [96, 192, 336, 720]
        mse, mse_s, nll, nll_s = gp_row(dataset, h)
        @printf("  h%-3d  MSE %.3f ± %.3f   NLL %.3f ± %.3f\n", h, mse, ci(mse_s), nll, ci(nll_s))
    end
end
