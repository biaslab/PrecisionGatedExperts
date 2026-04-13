using CSV
using DataFrames
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using ProbabilisticEnsembling
using RxInfer
using Statistics
using LinearAlgebra
using StableRNGs
using Random
using Plots
using Printf

# Replaces the true ReLU node
#
#     gamma[k, j] ~ ReLU(za[k, j])
#
# with an Exp-node factor graph surrogate:
#
#     gate_pos  = +S * za
#     gate_zero = -S * za
#     rho_pos   = exp(gate_pos)
#     rho_zero  = exp(gate_zero)
#     gamma is pulled toward za with precision rho_pos
#     gamma is pulled toward 0  with precision rho_zero
#
# The mean behavior is approximately:
#
#     gamma ~= za * sigmoid(2S * za)
#
# which is a smooth approximation to max(0, za).  S is fixed, not learned.
const RELU_SURROGATE_SCALE = parse(Float64, get(ENV, "RELU_SURROGATE_SCALE", "16.0"))
const RELU_SURROGATE_TAU = parse(Float64, get(ENV, "RELU_SURROGATE_TAU", "200.0"))
const GAMMA_FLOOR = 1e-8

const N_NEURONS = parse(Int, get(ENV, "N_NEURONS", "16"))
const N_ITERATIONS = parse(Int, get(ENV, "N_ITERATIONS", "10"))
const TRAIN_FRACTION = parse(Float64, get(ENV, "TRAIN_FRACTION", "0.3"))
const OBS_PRECISION = parse(Float64, get(ENV, "OBS_PRECISION", "1e6"))
const DATASET = get(ENV, "XOR_DATASET", "test_dataset/xor_simple_dataset.csv")
const OUTPUT_PREFIX = get(ENV, "OUTPUT_PREFIX", "test_dataset/viz/xor_relu_softdot_approx_exp")
const FREE_ENERGY = parse(Bool, get(ENV, "FREE_ENERGY", "false"))

function stable_sigmoid(x)
    if x >= 0
        return 1.0 / (1.0 + exp(-x))
    end
    ex = exp(x)
    return ex / (1.0 + ex)
end

function relu_surrogate_mean(a)
    return max(GAMMA_FLOOR, a * stable_sigmoid(2.0 * RELU_SURROGATE_SCALE * a))
end

@model function xor_relu_softdot_approx_exp(n_neurons, features, y, priors, obs_precision)
    local w_mean, w_a, z_mean, za, gate_pos, gate_zero
    local rho_pos, rho_zero, gamma, tau, out

    tau ~ priors[:tau]

    for k = 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k] ~ priors[:w_a][k]
    end

    for j = 1:length(y)
        for k = 1:n_neurons
            z_mean[k, j] ~ softdot(features[j], w_mean[k], tau)
            za[k, j] ~ softdot(features[j], w_a[k], tau)

            gate_pos[k, j] ~ softdot(za[k, j], RELU_SURROGATE_SCALE, RELU_SURROGATE_TAU)
            gate_zero[k, j] ~ softdot(za[k, j], -RELU_SURROGATE_SCALE, RELU_SURROGATE_TAU)

            rho_pos[k, j] ~ Exp(gate_pos[k, j])
            rho_zero[k, j] ~ Exp(gate_zero[k, j])

            gamma[k, j] ~ NormalMeanPrecision(za[k, j], rho_pos[k, j])
            gamma[k, j] ~ NormalMeanPrecision(0.0, rho_zero[k, j])
            out[j] ~ NormalMeanPrecision(z_mean[k, j], gamma[k, j])
        end
        y[j] ~ NormalMeanPrecision(out[j], obs_precision)
    end
end

@constraints function xor_relu_softdot_approx_exp_constraints()
    q(w_mean, w_a, z_mean, za, gate_pos, gate_zero, rho_pos, rho_zero, gamma, tau, out) =
        q(w_mean, z_mean)q(w_a, za)q(gate_pos, gate_zero, rho_pos, rho_zero)q(gamma)q(tau)q(out)

    q(za)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(gate_pos)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(gate_zero)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(rho_pos)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(rho_zero)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(gamma)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
end

@initialization function xor_relu_softdot_approx_exp_init(priors)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(0.0, 1.0)
    q(za) = NormalMeanVariance(0.0, 1.0)
    q(gate_pos) = NormalMeanVariance(0.0, 1.0)
    q(gate_zero) = NormalMeanVariance(0.0, 1.0)
    q(rho_pos) = GammaShapeScale(2.0, 1.0)
    q(rho_zero) = GammaShapeScale(2.0, 1.0)
    q(gamma) = GammaShapeScale(2.0, 1.0)
    q(tau) = priors[:tau]
    q(out) = NormalMeanVariance(0.5, 1.0)
end

function make_priors(; n_neurons = 4, n_features::Int = 3, seed::Int = 42)
    rng = StableRNG(seed)

    w_mean = [
        MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-4, n_features)))
        for _ in 1:n_neurons
    ]
    w_a = [
        MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-4, n_features)))
        for _ in 1:n_neurons
    ]

    return Dict{Symbol,Any}(
        :w_mean => w_mean,
        :w_a => w_a,
        :tau => GammaShapeRate(1e6, 1.0),
    )
end

build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

function split_dataset(df; train_fraction = 0.3, seed = 2027)
    rng = StableRNG(seed)
    idx = randperm(rng, nrow(df))
    n = round(Int, train_fraction * nrow(df))
    return df[idx[1:n], :], df[idx[(n + 1):end], :]
end

function predict_point(x1, x2, w_means, w_as)
    f = [1.0, x1, x2]
    mus = [dot(w_means[k], f) for k in eachindex(w_means)]
    precisions = [relu_surrogate_mean(dot(w_as[k], f)) for k in eachindex(w_as)]
    total = sum(precisions)
    return sum(precisions .* mus) / total
end

function save_heatmap(result, df_test, features_test, n_neurons)
    w_means = [mean(result.posteriors[:w_mean][end][k]) for k in 1:n_neurons]
    w_as = [mean(result.posteriors[:w_a][end][k]) for k in 1:n_neurons]

    x_coords = [f[2] for f in features_test]
    y_coords = [f[3] for f in features_test]
    grid_x = range(minimum(x_coords) - 0.5, maximum(x_coords) + 0.5; length = 200)
    grid_y = range(minimum(y_coords) - 0.5, maximum(y_coords) + 0.5; length = 200)

    z_grid = [predict_point(x, y, w_means, w_as) for y in grid_y, x in grid_x]
    z_actual = [Float64(x * y < 0) for y in grid_y, x in grid_x]

    p1 = contourf(
        grid_x,
        grid_y,
        z_grid;
        c = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Exp-node ReLU surrogate",
        linewidth = 0,
    )
    scatter!(p1, df_test.x1, df_test.x2; marker_z = df_test.OT, ms = 1.5, label = "", colorbar = false)

    p2 = contourf(
        grid_x,
        grid_y,
        z_actual;
        c = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Actual XOR",
        linewidth = 0,
    )

    mkpath(dirname(OUTPUT_PREFIX))
    outpath = "$(OUTPUT_PREFIX)_heatmap.png"
    plot(
        p1,
        p2;
        layout = (1, 2),
        size = (1000, 420),
        plot_title = "XOR with Exp-node ReLU surrogate, S=$(RELU_SURROGATE_SCALE)",
    )
    savefig(outpath)
    return outpath
end

function main()
    df = CSV.read(DATASET, DataFrame)
    df_train, df_test = split_dataset(df; train_fraction = TRAIN_FRACTION)
    features_train = build_features(df_train)
    features_test = build_features(df_test)

    println("=" ^ 78)
    println("XOR with Exp-node softdot approximation replacing ReLU")
    println("  surrogate gamma ~= za * sigmoid(2S * za)")
    println("  S=$RELU_SURROGATE_SCALE, tau_surrogate=$RELU_SURROGATE_TAU")
    println("  neurons=$N_NEURONS, iterations=$N_ITERATIONS, obs_precision=$OBS_PRECISION")
    println("  free_energy=$FREE_ENERGY")
    println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
    println("=" ^ 78)

    priors = make_priors(n_neurons = N_NEURONS)
    result = infer(
        model = xor_relu_softdot_approx_exp(
            n_neurons = N_NEURONS,
            priors = priors,
            obs_precision = OBS_PRECISION,
        ),
        data = (y = df_train.OT, features = features_train),
        constraints = xor_relu_softdot_approx_exp_constraints(),
        initialization = xor_relu_softdot_approx_exp_init(priors),
        iterations = N_ITERATIONS,
        free_energy = FREE_ENERGY,
        showprogress = true,
        options = (limit_stack_depth = 100,),
    )

    println("\nLearned weights:")
    for k in 1:N_NEURONS
        wm = round.(mean(result.posteriors[:w_mean][end][k]); digits = 4)
        wa = round.(mean(result.posteriors[:w_a][end][k]); digits = 4)
        println("  neuron $k: w_mean=$wm  w_a=$wa")
    end

    w_means = [mean(result.posteriors[:w_mean][end][k]) for k in 1:N_NEURONS]
    w_as = [mean(result.posteriors[:w_a][end][k]) for k in 1:N_NEURONS]
    y_pred = [predict_point(f[2], f[3], w_means, w_as) for f in features_test]
    mse = mean((y_pred .- df_test.OT) .^ 2)
    mse_constant = mean((0.52 .- df_test.OT) .^ 2)

    @printf("\nTest MSE (Exp-node ReLU surrogate) = %.6f\n", mse)
    @printf("Baseline MSE (constant 0.52)      = %.6f\n", mse_constant)

    heatmap_path = save_heatmap(result, df_test, features_test, N_NEURONS)
    println("Heatmap saved to $heatmap_path")

    return result
end

main()
