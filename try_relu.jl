#!/usr/bin/env julia

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Statistics
using Plots

# Fixed, non-learned parameters for the ReLU capability demo.
#
# The architecture is a fixed precision-gated mixture of two experts:
#   zero branch:     y ≈ 0
#   identity branch: y ≈ x
#
# The gate score is h = S * x.  The branch precisions are fixed numbers:
#
#   kappa_identity = exp(h)
#   kappa_zero     = exp(-h)
#
# Therefore the posterior mean is exactly
#
#   y = (exp(h) * x + exp(-h) * 0) / (exp(h) + exp(-h))
#     = sigmoid(2Sx) * x
#
# This is a smooth, hardening approximation to max(0, x).
const RELU_GATE_WEIGHT = [320.0, 0.0]
const RELU_IDENTITY_WEIGHT = [1.0, 0.0]
const RELU_ZERO_WEIGHT = [0.0, 0.0]
const TAU_SOFTDOT = 1e6

relu_gate_approx(x; scale = RELU_GATE_WEIGHT[1]) = x / (1.0 + exp(-2.0 * scale * x))

@model function deep_model_relu_gate(n_obs, features, y, identity_precision, zero_precision)
    local z_identity, z_zero

    for j = 1:n_obs
        z_identity[j] ~ softdot(features[j], RELU_IDENTITY_WEIGHT, TAU_SOFTDOT)
        z_zero[j] ~ softdot(features[j], RELU_ZERO_WEIGHT, TAU_SOFTDOT)

        y[j] ~ NormalMeanPrecision(z_identity[j], identity_precision[j])
        y[j] ~ NormalMeanPrecision(z_zero[j], zero_precision[j])
    end
end

@constraints function relu_gate_constraints()
    q(z_identity, z_zero) = q(z_identity)q(z_zero)
end

@initialization function relu_gate_init()
    q(z_identity) = NormalMeanVariance(0.0, 1.0)
    q(z_zero) = NormalMeanVariance(0.0, 1.0)
end

function relu_dataset()
    xs = collect(range(-2.0, 2.0; length = 17))
    features = [Float64[x, 1.0] for x in xs]
    targets = max.(0.0, xs)
    return xs, features, targets
end

function run_relu_demo()
    xs, features, targets = relu_dataset()

    result = infer(
        model = deep_model_relu_gate(n_obs = length(features)),
        data = (
            features = features,
            y = fill(missing, length(features)),
            identity_precision = exp.([dot(RELU_GATE_WEIGHT, f) for f in features]),
            zero_precision = exp.(-[dot(RELU_GATE_WEIGHT, f) for f in features]),
        ),
        constraints = relu_gate_constraints(),
        initialization = relu_gate_init(),
        iterations = 10,
        free_energy = false,
        showprogress = false,
    )

    y_post = result.predictions[:y][end]
    y_mean = mean.(y_post)
    y_var = var.(y_post)

    h = [dot(RELU_GATE_WEIGHT, f) for f in features]
    identity_precision = exp.(h)
    zero_precision = exp.(-h)

    mse = mean((y_mean .- targets) .^ 2)
    max_abs_error = maximum(abs.(y_mean .- targets))

    println("ReLU capability check (fixed numeric architecture):")
    println("  target function: max(0, x)")
    println("  smooth form:     x * sigmoid(2 * $(RELU_GATE_WEIGHT[1]) * x)")
    for j = 1:length(xs)
        println(
            "x=$(rpad(round(xs[j], digits = 3), 6)) ",
            "target=$(rpad(round(targets[j], digits = 4), 6)) ",
            "pred=$(rpad(round(y_mean[j], digits = 4), 8)) ",
            "var=$(rpad(round(y_var[j], digits = 4), 8)) ",
            "h=$(rpad(round(h[j], digits = 4), 8)) ",
            "kappa=($(round(identity_precision[j], digits = 4)), $(round(zero_precision[j], digits = 4)))",
        )
    end
    println("MSE: ", round(mse, digits = 6))
    println("Max abs error: ", round(max_abs_error, digits = 6))
end

function plot_relu_difference(; outpath = "test_dataset/viz/fixed_relu_gate_vs_relu.png")
    xs = collect(range(-10.0, 10.0; length = 5000))
    true_relu = max.(0.0, xs)
    approx = relu_gate_approx.(xs)
    residual = approx .- true_relu

    p1 = plot(
        xs,
        true_relu;
        label = "ReLU max(0, x)",
        linewidth = 3,
        color = :black,
        xlabel = "x",
        ylabel = "output",
        title = "Fixed Precision Gate vs ReLU",
        legend = :topleft,
    )
    plot!(p1, xs, approx; label = "x * sigmoid(16x)", linewidth = 3, color = :blue)

    p2 = plot(
        xs,
        residual;
        label = "approx - ReLU",
        linewidth = 3,
        color = :red,
        xlabel = "x",
        ylabel = "error",
        title = "Difference",
        legend = :bottomright,
    )
    hline!(p2, [0.0]; label = "", linestyle = :dash, color = :black, alpha = 0.5)

    plot(p1, p2; layout = (2, 1), size = (900, 700))
    savefig(outpath)
    println("Saved plot: $outpath")
end

run_relu_demo()
plot_relu_difference()
