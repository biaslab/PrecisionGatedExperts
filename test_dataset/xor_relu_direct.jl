using CSV
using DataFrames
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using ClosedFormExpectations: EnzymeBackend
using ProbabilisticEnsembling
using RxInfer
using Statistics
using LinearAlgebra
using StableRNGs
using Random
using Plots

@model function xor_relu_direct(n_neurons, features, y, priors)
    local w_mean, w_a, z_mean, za, γ, τ, out

    τ ~ priors[:τ]

    for k = 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k] ~ priors[:w_a][k]
    end

    for j = 1:length(y)
        for k = 1:n_neurons
            z_mean[k, j] ~ softdot(features[j], w_mean[k], τ)
            za[k, j] ~ softdot(features[j], w_a[k], τ) where {meta = LowRankMeta()}
            γ[k, j] ~ ReLU(za[k, j])
            out[j] ~ NormalMeanPrecision(z_mean[k, j], γ[k, j])
        end
        y[j] ~ NormalMeanPrecision(out[j], 1e6)
    end
end

@constraints function xor_relu_direct_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, out) = q(w_mean, z_mean)q(w_a)q(za, γ)q(τ)q(out)
    q(za)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
end

@initialization function xor_relu_direct_init(priors)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(0.0, 1.0)
    q(za) = GammaShapeScale(2.0, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
    q(τ) = priors[:τ]

end

function make_priors(; n_neurons=4, n_features::Int = 3, seed::Int = 42)
    rng = StableRNG(seed)

    w_mean = [MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-4, n_features))) for _ in 1:n_neurons]
    w_a = [MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-3, n_features))) for _ in 1:n_neurons]

    return Dict{Symbol,Any}(
        :w_mean => w_mean,
        :w_a => w_a,
        :τ => GammaShapeRate(1e6, 1.0),
    )
end

build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

function split_dataset(df; train_fraction = 0.3, seed = 2026)
    rng = StableRNG(seed)
    idx = randperm(rng, nrow(df))
    n = round(Int, train_fraction * nrow(df))
    df[idx[1:n], :], df[idx[n+1:end], :]
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
n_neurons = 16
df = CSV.read("test_dataset/xor_simple_dataset.csv", DataFrame)
df_train, df_test = split_dataset(df; seed = 2027)

println("=" ^ 70)
println("XOR ReLU Direct: precision-weighted, $(n_neurons) neurons")
println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
println("=" ^ 70)

priors = make_priors(n_neurons = n_neurons)
features_train = build_features(df_train)

result = infer(
    model = xor_relu_direct(n_neurons = n_neurons, priors = priors),
    data = (y = df_train.OT, features = features_train),
    constraints = xor_relu_direct_constraints(),
    initialization = xor_relu_direct_init(priors),
    iterations = 100,
    free_energy = true,
    showprogress = true,
    options = (limit_stack_depth = 100,),
)

fe = result.free_energy
plot(1:length(fe), fe)

println("\nLearned weights:")
for k in 1:n_neurons
    wm = round.(mean(result.posteriors[:w_mean][end][k]); digits = 4)
    wa = round.(mean(result.posteriors[:w_a][end][k]); digits = 4)
    println("  neuron $k: w_mean=$wm  w_a=$wa")
end

# Evaluate
features_test = build_features(df_test)
w_means_post = [mean(result.posteriors[:w_mean][end][k]) for k in 1:n_neurons]
w_a_post = [mean(result.posteriors[:w_a][end][k]) for k in 1:n_neurons]

function predict_point(x1, x2, w_means, w_as)
    f = [1.0, x1, x2]
    μs = [dot(w_means[k], f) for k in 1:n_neurons]
    γs = [max(0.0, dot(w_as[k], f)) for k in 1:n_neurons]
    total = sum(γs)
    return sum(γs .* μs) / total
end

y_pred = map(features_test) do f
    predict_point(f[2], f[3], w_means_post, w_a_post)
end

y_test = df_test.OT
mse = mean((y_pred .- y_test) .^ 2)
mse_constant = mean((0.52 .- y_test) .^ 2)
println("\nTest MSE (precision-weighted) = $(round(mse; digits=4))")
println("(Baseline: predicting 0.52 gives MSE ≈ $(mse_constant))")

x_test_coords = [features_test[i][2] for i in 1:length(features_test)]
y_test_coords = [features_test[i][3] for i in 1:length(features_test)]

grid_x = range(minimum(x_test_coords) - 0.5, maximum(x_test_coords) + 0.5; length = 200)
grid_y = range(minimum(y_test_coords) - 0.5, maximum(y_test_coords) + 0.5; length = 200)
z_grid = [predict_point(x, y, w_means_post, w_a_post) for y in grid_y, x in grid_x]

z_actual = [Float64(x * y < 0) for y in grid_y, x in grid_x]

begin
    p1 = contourf(
        grid_x, grid_y, z_grid,
        c = :RdBu,
        levels = 20,
        xlabel = "x1",
        ylabel = "x2",
        title = "Predicted",
        linewidth = 0,
    )

    p2 = contourf(
        grid_x, grid_y, z_actual,
        c = :RdBu,
        levels = 20,
        xlabel = "x1",
        ylabel = "x2",
        title = "Actual XOR",
        linewidth = 0,
    )

    plot(p1, p2, layout = (1, 2), size = (1000, 400), plot_title = "XOR ReLU")
end
savefig("test_dataset/viz/heatmap_relu.png")

# ---------------------------------------------------------------------------
# Animation: contourf across variational iterations
# ---------------------------------------------------------------------------
n_iters = length(result.posteriors[:w_mean])

anim = @animate for iter in 1:n_iters
    wm_iter = [mean(result.posteriors[:w_mean][iter][k]) for k in 1:n_neurons]
    wa_iter = [mean(result.posteriors[:w_a][iter][k]) for k in 1:n_neurons]

    z_iter = [predict_point(x, y, wm_iter, wa_iter) for y in grid_y, x in grid_x]

    p1 = contourf(
        grid_x, grid_y, z_iter,
        c = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predicted  (iter $iter / $n_iters)",
        linewidth = 0,
    )

    p2 = contourf(
        grid_x, grid_y, z_actual,
        c = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Actual XOR",
        linewidth = 0,
    )

    plot(p1, p2, layout = (1, 2), size = (1000, 400),
         plot_title = "XOR ReLU — variational iteration $iter")
end

gif(anim, "test_dataset/viz/relu_iterations.gif", fps = 5)
println("Animation saved to test_dataset/viz/relu_iterations.gif")