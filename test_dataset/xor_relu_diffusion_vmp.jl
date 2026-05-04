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
using Printf
using ProbabilisticEnsembling: StrangeMissingMeta

const DATASET = get(ENV, "XOR_DATASET", "test_dataset/xor_simple_dataset.csv")
const OUTPUT_PREFIX = get(ENV, "OUTPUT_PREFIX", "test_dataset/viz/xor_relu_diffusion_vmp")
const N_NEURONS = parse(Int, get(ENV, "N_NEURONS", "8"))
const L_DEPTH = parse(Int, get(ENV, "L_DEPTH", "3"))
const N_ITERATIONS = parse(Int, get(ENV, "N_ITERATIONS", "5"))
const PREDICTION_ITERATIONS = parse(Int, get(ENV, "PREDICTION_ITERATIONS", "5"))
const TRAIN_FRACTION = parse(Float64, get(ENV, "TRAIN_FRACTION", "0.5"))
const GRID_SIZE = parse(Int, get(ENV, "GRID_SIZE", "32"))

# Diffusion noise schedule. sigma^2 at l=L (clean end) = SIGMA_MIN^2,
# sigma^2 at l=1 (noise end) = SIGMA_MAX^2.
const SIGMA_MIN = parse(Float64, get(ENV, "SIGMA_MIN", "0.05"))
const SIGMA_MAX = parse(Float64, get(ENV, "SIGMA_MAX", "0.50"))
const SCHEDULE = lowercase(get(ENV, "SCHEDULE", "linear"))  # linear | geometric | constant
const CHAIN_VAR_OVERRIDE = parse(Float64, get(ENV, "CHAIN_VAR", "-1.0"))
const CHAIN_VAR_FLOOR = 1e-4

const W_MEAN_PRECISION = parse(Float64, get(ENV, "W_MEAN_PRECISION", "0.1"))
const W_A_PRECISION = parse(Float64, get(ENV, "W_A_PRECISION", "1.0"))
const WEIGHT_PRIOR_MEAN_SCALE = parse(Float64, get(ENV, "WEIGHT_PRIOR_MEAN_SCALE", "1.0"))
const TAU_SHAPE = parse(Float64, get(ENV, "TAU_SHAPE", "1e8"))
const TAU_RATE = parse(Float64, get(ENV, "TAU_RATE", "100.0"))
const OBS_TAU_SHAPE = parse(Float64, get(ENV, "OBS_TAU_SHAPE", "1e8"))
const OBS_TAU_RATE = parse(Float64, get(ENV, "OBS_TAU_RATE", "1e6"))

const PRIOR_SEED = parse(Int, get(ENV, "PRIOR_SEED", "2027"))
const SPLIT_SEED = parse(Int, get(ENV, "SPLIT_SEED", "2027"))
const NOISE_SEED = parse(Int, get(ENV, "NOISE_SEED", "2028"))
const FREE_ENERGY = lowercase(get(ENV, "FREE_ENERGY", "true")) in ("1", "true", "yes")

const MAKE_GIF = lowercase(get(ENV, "MAKE_GIF", "false")) in ("1", "true", "yes")
const GIF_STRIDE = parse(Int, get(ENV, "GIF_STRIDE", "1"))
const GIF_PRED_ITER = parse(Int, get(ENV, "GIF_PRED_ITER", "3"))
const GIF_FPS = parse(Int, get(ENV, "GIF_FPS", "2"))

# ---------------------------------------------------------------------------
# Diffusion model: strategy (B) — forward-diffuse targets y[l, j] = y_true[j] + N(0, sigma2[l]).
# Weights w_mean[k], w_a[k] are shared across levels (weight tying).
# Chain: out[l, j] ~ N(out[l-1, j], chain_var).
# Noise-end anchor: out[1, j] ~ N(0, sigma2[1]); experts refine from l=2..L.
# Clean-end prediction is out[L, j].
# ---------------------------------------------------------------------------

@model function xor_relu_diffusion_vmp(n_neurons, n_levels, features, y, priors,
                                        chain_var, prior_var)
    local w_mean, w_a, z_mean, za, γ, τ, obs_tau, out

    τ       ~ priors[:τ]
    obs_tau ~ priors[:obs_tau]

    for k = 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k]    ~ priors[:w_a][k]
    end

    for j = 1:size(y, 2)
        out[1, j] ~ NormalMeanVariance(0.0, prior_var)

        for l = 1:n_levels
            if l > 1
                out[l, j] ~ NormalMeanVariance(out[l-1, j], chain_var)
            end

            for k = 1:n_neurons
                z_mean[k, l, j] ~ softdot(features[j], w_mean[k], τ)
                za[k, l, j]    ~ softdot(features[j], w_a[k], τ) where {meta = LowRankMeta()}
                γ[k, l, j]     ~ ReLU(za[k, l, j])
                out[l, j]      ~ NormalMeanPrecision(z_mean[k, l, j], γ[k, l, j])
            end

            y[l, j] ~ NormalMeanPrecision(out[l, j], obs_tau)
        end
    end
end

@model function xor_relu_diffusion_vmp_prediction(n_neurons, n_levels, n_obs, features, priors,
                                                    chain_var, prior_var)
    local w_mean, w_a, z_mean, za, γ, τ, obs_tau, out, y

    τ       ~ priors[:τ]
    obs_tau ~ priors[:obs_tau]

    for k = 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k]    ~ priors[:w_a][k]
    end

    for j = 1:n_obs
        out[1, j] ~ NormalMeanVariance(0.0, prior_var)

        for l = 1:n_levels
            if l > 1
                out[l, j] ~ NormalMeanVariance(out[l-1, j], chain_var)
            end

            for k = 1:n_neurons
                z_mean[k, l, j] ~ softdot(features[j], w_mean[k], τ)
                za[k, l, j]    ~ softdot(features[j], w_a[k], τ) where {meta = LowRankMeta()}
                γ[k, l, j]     ~ ReLU(za[k, l, j])
                out[l, j]      ~ NormalMeanPrecision(z_mean[k, l, j], γ[k, l, j])
            end

            y[l, j] ~ NormalMeanPrecision(out[l, j], obs_tau) where {meta = StrangeMissingMeta()}
            y[l, j] ~ Uninformative()
        end
    end
end

@constraints function xor_relu_diffusion_vmp_constraints()
    q(w_mean, w_a, z_mean, za, γ, τ, obs_tau, out) =
        q(w_mean, z_mean, out)q(w_a)q(za, γ)q(τ)q(obs_tau)

    q(z_mean)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(out)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(za)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
end

@constraints function xor_relu_diffusion_vmp_prediction_constraints(priors)
    q(w_mean, w_a, z_mean, za, γ, τ, obs_tau, out, y) =
        q(w_mean)q(w_a)q(τ)q(obs_tau)q(z_mean)q(za, γ)q(out, y)

    q(z_mean)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(out)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(za)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
    q(γ)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))

    q(τ)::RxInfer.FixedMarginalFormConstraint(priors[:τ])
    q(obs_tau)::RxInfer.FixedMarginalFormConstraint(priors[:obs_tau])

    for (k, prior) in enumerate(priors[:w_mean])
        q(w_mean[k])::RxInfer.FixedMarginalFormConstraint(prior)
    end
    for (k, prior) in enumerate(priors[:w_a])
        q(w_a[k])::RxInfer.FixedMarginalFormConstraint(prior)
    end
end

@initialization function xor_relu_diffusion_vmp_init(priors)
    q(w_mean)  = deepcopy(priors[:w_mean])
    q(w_a)     = deepcopy(priors[:w_a])
    q(τ)       = priors[:τ]
    q(obs_tau) = priors[:obs_tau]
    q(z_mean)  = NormalMeanVariance(0.0, 1.0)
    q(za)      = GammaShapeScale(2.0, 1.0)
    q(γ)       = GammaShapeScale(2.0, 1.0)
    q(out)     = NormalMeanVariance(0.5, 1.0)

    μ(out)    = NormalMeanVariance(0.5, 10.0)
    μ(z_mean) = NormalMeanVariance(0.0, 10.0)
end

@initialization function xor_relu_diffusion_vmp_prediction_init(priors)
    q(w_mean)  = deepcopy(priors[:w_mean])
    q(w_a)     = deepcopy(priors[:w_a])
    q(τ)       = priors[:τ]
    q(obs_tau) = priors[:obs_tau]
    q(z_mean)  = NormalMeanVariance(0.0, 1.0)
    q(za)      = GammaShapeScale(2.0, 1.0)
    q(γ)       = GammaShapeScale(2.0, 1.0)
    q(out)     = NormalMeanVariance(0.5, 1.0)
    q(y)       = NormalMeanVariance(0.5, 1.0)

    μ(out)    = NormalMeanVariance(0.5, 10.0)
    μ(z_mean) = NormalMeanVariance(0.0, 10.0)
    μ(y)      = NormalMeanVariance(0.5, 10.0)
end

# ---------------------------------------------------------------------------
# Variance schedule & target generation
# ---------------------------------------------------------------------------

function variance_schedule(n_levels, sigma_min, sigma_max; kind = "linear")
    if n_levels == 1
        return [sigma_min^2]
    end
    if kind == "constant"
        return fill((sigma_min^2 + sigma_max^2) / 2, n_levels)
    elseif kind == "geometric"
        # sigma^2[L] = sigma_min^2, sigma^2[1] = sigma_max^2
        r = (sigma_max^2 / sigma_min^2)^(1.0 / (n_levels - 1))
        return [sigma_min^2 * r^(n_levels - l) for l in 1:n_levels]
    else  # linear
        step = (sigma_max^2 - sigma_min^2) / (n_levels - 1)
        return [sigma_max^2 - step * (l - 1) for l in 1:n_levels]
    end
end

function forward_diffuse_targets(y_true, sigma2_schedule; seed = NOISE_SEED)
    rng = StableRNG(seed)
    n_levels = length(sigma2_schedule)
    n = length(y_true)
    y = Matrix{Float64}(undef, n_levels, n)
    for l = 1:n_levels
        σ = sqrt(sigma2_schedule[l])
        for j = 1:n
            y[l, j] = y_true[j] + σ * randn(rng)
        end
    end
    return y
end

function make_weight_prior(rng, n_features, precision)
    precision_matrix = Diagonal(fill(precision, n_features))
    actual_mean = WEIGHT_PRIOR_MEAN_SCALE .* randn(rng, n_features)
    return MvNormalWeightedMeanPrecision(precision_matrix * actual_mean, precision_matrix)
end

function make_priors(; n_neurons = N_NEURONS, n_features = 3, seed = PRIOR_SEED)
    rng = StableRNG(seed)
    return Dict{Symbol,Any}(
        :w_mean  => [make_weight_prior(rng, n_features, W_MEAN_PRECISION) for _ in 1:n_neurons],
        :w_a     => [make_weight_prior(rng, n_features, W_A_PRECISION) for _ in 1:n_neurons],
        :τ       => GammaShapeRate(TAU_SHAPE, TAU_RATE),
        :obs_tau => GammaShapeRate(OBS_TAU_SHAPE, OBS_TAU_RATE),
    )
end

build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

function split_train_test(df)
    rng = StableRNG(SPLIT_SEED)
    idx = randperm(rng, nrow(df))
    n_train = round(Int, TRAIN_FRACTION * nrow(df))
    return df[idx[1:n_train], :], df[idx[(n_train + 1):end], :]
end

function posterior_priors(result; iteration = length(result.posteriors[:w_mean]))
    return Dict{Symbol,Any}(
        :w_mean  => deepcopy(result.posteriors[:w_mean][iteration]),
        :w_a     => deepcopy(result.posteriors[:w_a][iteration]),
        :τ       => result.posteriors[:τ][iteration],
        :obs_tau => result.posteriors[:obs_tau][iteration],
    )
end

function predict_with_rxinfer(priors, features, chain_var, prior_var;
                               iterations = PREDICTION_ITERATIONS, showprogress = true)
    return infer(
        model = xor_relu_diffusion_vmp_prediction(
            n_neurons = N_NEURONS,
            n_levels = L_DEPTH,
            n_obs = length(features),
            priors = priors,
            chain_var = chain_var,
            prior_var = prior_var,
        ),
        data = (features = features,),
        constraints = xor_relu_diffusion_vmp_prediction_constraints(priors),
        initialization = xor_relu_diffusion_vmp_prediction_init(priors),
        iterations = iterations,
        free_energy = false,
        showprogress = showprogress,
        options = (limit_stack_depth = 100,),
    )
end

function clean_out_marginals(prediction_result)
    # Final-iteration posterior over out[L, j] for all j
    final_out = prediction_result.posteriors[:out][end]
    return [final_out[L_DEPTH, j] for j in 1:size(final_out, 2)]
end

predictive_mean(marginals) = Float64.(mean.(marginals))
predictive_variance(marginals) = Float64.(var.(marginals))

function mse_from_marginals(marginals, y_true)
    y_pred = predictive_mean(marginals)
    return mean((y_pred .- y_true) .^ 2)
end

function training_diagnostics(result, y_true_train)
    rows = DataFrame(
        iteration = Int[],
        free_energy = Float64[],
        tau = Float64[],
        obs_tau = Float64[],
        w_mean_max_norm = Float64[],
        w_a_max_norm = Float64[],
        train_mse_cleanend = Float64[],
    )
    fe = FREE_ENERGY ? result.free_energy : Real[]
    n_it = length(result.posteriors[:w_mean])
    for iteration in 1:n_it
        w_means = [mean(result.posteriors[:w_mean][iteration][k]) for k in 1:N_NEURONS]
        w_as = [mean(result.posteriors[:w_a][iteration][k]) for k in 1:N_NEURONS]
        fe_val = iteration <= length(fe) ? Float64(fe[iteration]) : NaN

        out_mat = result.posteriors[:out][iteration]
        y_pred = [Float64(mean(out_mat[L_DEPTH, j])) for j in 1:length(y_true_train)]
        mse = mean((y_pred .- y_true_train) .^ 2)

        push!(rows, (
            iteration,
            fe_val,
            Float64(mean(result.posteriors[:τ][iteration])),
            Float64(mean(result.posteriors[:obs_tau][iteration])),
            maximum(norm.(w_means)),
            maximum(norm.(w_as)),
            mse,
        ))
    end
    return rows
end

function per_level_out_stats(result, n_train)
    n_it = length(result.posteriors[:out])
    rows = DataFrame(iteration = Int[], level = Int[], mean_abs = Float64[], mean_std = Float64[])
    for iteration in 1:n_it
        mat = result.posteriors[:out][iteration]
        for l in 1:L_DEPTH
            ms = [mean(mat[l, j]) for j in 1:n_train]
            vs = [var(mat[l, j]) for j in 1:n_train]
            push!(rows, (iteration, l, mean(abs.(ms)), mean(sqrt.(max.(vs, 0.0)))))
        end
    end
    return rows
end

function save_free_energy_plot(diagnostics)
    outpath = "$(OUTPUT_PREFIX)_free_energy.png"
    p = plot(
        diagnostics.iteration,
        diagnostics.free_energy;
        marker = :circle,
        xlabel = "iteration",
        ylabel = "free energy",
        title = "XOR diffusion VMP (L=$L_DEPTH, schedule=$SCHEDULE)",
        label = "free energy",
        size = (760, 440),
    )
    savefig(p, outpath)
    return outpath
end

function save_stability_plot(diagnostics)
    outpath = "$(OUTPUT_PREFIX)_stability.png"
    p1 = plot(diagnostics.iteration, diagnostics.train_mse_cleanend;
        marker = :circle, xlabel = "iteration", ylabel = "train MSE (clean end)",
        title = "Train MSE", label = "train MSE", size = (380, 300), lw = 2)
    p2 = plot(diagnostics.iteration, diagnostics.obs_tau;
        marker = :circle, xlabel = "iteration", ylabel = "E[obs_tau]",
        title = "Learned obs_tau", label = "obs_tau", size = (380, 300), lw = 2)
    p3 = plot(diagnostics.iteration, diagnostics.w_mean_max_norm;
        marker = :circle, xlabel = "iteration", ylabel = "max |w|",
        title = "Weight norms", label = "|w_mean|", size = (380, 300), lw = 2)
    plot!(p3, diagnostics.iteration, diagnostics.w_a_max_norm;
        marker = :square, label = "|w_a|", lw = 2)
    plot(p1, p2, p3; layout = (1, 3), size = (1200, 330),
        plot_title = "diffusion VMP stability (L=$L_DEPTH, $SCHEDULE)")
    savefig(outpath)
    return outpath
end

function save_prediction_csv(marginals, df_test)
    outpath = "$(OUTPUT_PREFIX)_test_predictions.csv"
    y_mean = predictive_mean(marginals)
    y_var = predictive_variance(marginals)
    predictions = DataFrame(
        x1 = df_test.x1,
        x2 = df_test.x2,
        y_true = df_test.OT,
        y_mean = y_mean,
        y_variance = y_var,
        y_std = sqrt.(max.(y_var, 0.0)),
    )
    CSV.write(outpath, predictions)
    return outpath
end

function prediction_grid(features_test; grid_size = GRID_SIZE)
    x_coords = [f[2] for f in features_test]
    y_coords = [f[3] for f in features_test]
    grid_x = range(minimum(x_coords) - 0.5, maximum(x_coords) + 0.5; length = grid_size)
    grid_y = range(minimum(y_coords) - 0.5, maximum(y_coords) + 0.5; length = grid_size)
    features_grid = vec([[1.0, x, y] for y in grid_y, x in grid_x])
    actual_grid = [Float64(x * y < 0) for y in grid_y, x in grid_x]
    return grid_x, grid_y, features_grid, actual_grid
end

function build_iteration_gif(training_result, grid_x, grid_y, features_grid, df_test,
                              chain_var, prior_var; stride = GIF_STRIDE,
                              pred_iter = GIF_PRED_ITER, fps = GIF_FPS)
    n_it = length(training_result.posteriors[:w_mean])
    iter_indices = collect(1:stride:n_it)
    if iter_indices[end] != n_it
        push!(iter_indices, n_it)
    end
    println("\nBuilding heatmap GIF over $(length(iter_indices)) VMP iterations (stride=$stride, pred_iter=$pred_iter)")

    anim = Animation()
    for (f, it) in enumerate(iter_indices)
        println("  frame $f/$(length(iter_indices)): iter=$it")
        pr = Dict{Symbol,Any}(
            :w_mean  => deepcopy(training_result.posteriors[:w_mean][it]),
            :w_a     => deepcopy(training_result.posteriors[:w_a][it]),
            :τ       => training_result.posteriors[:τ][it],
            :obs_tau => training_result.posteriors[:obs_tau][it],
        )
        pred = predict_with_rxinfer(pr, features_grid, chain_var, prior_var;
                                     iterations = pred_iter, showprogress = false)
        mg = clean_out_marginals(pred)
        m_grid = reshape(predictive_mean(mg), length(grid_y), length(grid_x))
        sd_grid = sqrt.(max.(reshape(predictive_variance(mg), length(grid_y), length(grid_x)), 0.0))

        p1 = contourf(grid_x, grid_y, m_grid;
            c = :RdBu, levels = 20, clims = (-1, 1),
            xlabel = "x1", ylabel = "x2",
            title = "predictive mean (iter $it)", linewidth = 0)
        scatter!(p1, df_test.x1, df_test.x2; marker_z = df_test.OT, ms = 1.2,
            label = "", colorbar = false)

        p2 = contourf(grid_x, grid_y, sd_grid;
            c = :viridis, levels = 20,
            xlabel = "x1", ylabel = "x2",
            title = "predictive std (iter $it)", linewidth = 0)

        plot(p1, p2; layout = (1, 2), size = (900, 400),
            plot_title = "diffusion VMP (L=$L_DEPTH, $SCHEDULE)  —  VMP iter $it/$n_it")
        frame(anim)
    end

    outpath = "$(OUTPUT_PREFIX)_iter_heatmap.gif"
    gif(anim, outpath; fps = fps)
    return outpath
end

function save_prediction_heatmaps(grid_x, grid_y, grid_marginals, actual_grid, df_test)
    y_mean_grid = reshape(predictive_mean(grid_marginals), length(grid_y), length(grid_x))
    y_var_grid = reshape(predictive_variance(grid_marginals), length(grid_y), length(grid_x))
    y_std_grid = sqrt.(max.(y_var_grid, 0.0))

    p1 = contourf(grid_x, grid_y, y_mean_grid;
        c = :RdBu, levels = 20, clims = (-1, 1),
        xlabel = "x1", ylabel = "x2", title = "Predictive mean", linewidth = 0)
    scatter!(p1, df_test.x1, df_test.x2; marker_z = df_test.OT, ms = 1.5, label = "", colorbar = false)

    p2 = contourf(grid_x, grid_y, y_std_grid;
        c = :viridis, levels = 20, xlabel = "x1", ylabel = "x2", title = "Predictive std", linewidth = 0)

    p3 = contourf(grid_x, grid_y, actual_grid;
        c = :RdBu, levels = 20, clims = (0, 1),
        xlabel = "x1", ylabel = "x2", title = "Actual XOR", linewidth = 0)

    outpath = "$(OUTPUT_PREFIX)_posterior_predictive_heatmaps.png"
    plot(p1, p2, p3; layout = (1, 3), size = (1350, 420),
        plot_title = "diffusion VMP (L=$L_DEPTH, $SCHEDULE)")
    savefig(outpath)
    return outpath
end

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

df = CSV.read(DATASET, DataFrame)
df_train, df_test = split_train_test(df)
features_train = build_features(df_train)
features_test = build_features(df_test)
priors = make_priors()

sigma2_sched = variance_schedule(L_DEPTH, SIGMA_MIN, SIGMA_MAX; kind = SCHEDULE)
prior_var = sigma2_sched[1]  # noise-end variance
chain_var_val = CHAIN_VAR_OVERRIDE > 0 ? CHAIN_VAR_OVERRIDE :
                max(CHAIN_VAR_FLOOR,
                    L_DEPTH > 1 ? (SIGMA_MAX^2 - SIGMA_MIN^2) / (L_DEPTH - 1) : SIGMA_MAX^2)

y_train_matrix = forward_diffuse_targets(df_train.OT, sigma2_sched)

println("=" ^ 78)
println("XOR diffusion VMP (weight-tied, learned obs_tau)")
println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
println("  L_DEPTH=$L_DEPTH, schedule=$SCHEDULE (for target-noise generation only)")
println("  sigma^2 per level = $(round.(sigma2_sched, digits=4))")
println("  chain_var = $(round(chain_var_val, digits=4)), prior_var = $(round(prior_var, digits=4))")
println("  obs_tau prior = GammaShapeRate($OBS_TAU_SHAPE, $OBS_TAU_RATE), mean = $(OBS_TAU_SHAPE / OBS_TAU_RATE)")
println("  training iter=$N_ITERATIONS, prediction iter=$PREDICTION_ITERATIONS, neurons=$N_NEURONS")
println("=" ^ 78)

training_result = infer(
    model = xor_relu_diffusion_vmp(
        n_neurons = N_NEURONS,
        n_levels = L_DEPTH,
        priors = priors,
        chain_var = chain_var_val,
        prior_var = prior_var,
    ),
    data = (y = y_train_matrix, features = features_train),
    constraints = xor_relu_diffusion_vmp_constraints(),
    initialization = xor_relu_diffusion_vmp_init(priors),
    iterations = N_ITERATIONS,
    free_energy = FREE_ENERGY,
    showprogress = true,
    options = (limit_stack_depth = 100,),
)

prediction_priors = posterior_priors(training_result)

println("\nRunning posterior predictive inference on test data")
test_prediction = predict_with_rxinfer(prediction_priors, features_test, chain_var_val, prior_var)
test_marginals = clean_out_marginals(test_prediction)

println("\nRunning posterior predictive inference on grid")
grid_x, grid_y, features_grid, actual_grid = prediction_grid(features_test)
grid_prediction = predict_with_rxinfer(prediction_priors, features_grid, chain_var_val, prior_var)
grid_marginals = clean_out_marginals(grid_prediction)

baseline_mse = mean((0.52 .- df_test.OT) .^ 2)
final_mse = mse_from_marginals(test_marginals, df_test.OT)
y_var = predictive_variance(test_marginals)

mkpath(dirname(OUTPUT_PREFIX))
diagnostics = training_diagnostics(training_result, Vector{Float64}(df_train.OT))
diagnostics_path = "$(OUTPUT_PREFIX)_training_diagnostics.csv"
CSV.write(diagnostics_path, diagnostics)
level_stats = per_level_out_stats(training_result, nrow(df_train))
CSV.write("$(OUTPUT_PREFIX)_level_stats.csv", level_stats)
prediction_path = save_prediction_csv(test_marginals, df_test)
free_energy_plot = save_free_energy_plot(diagnostics)
stability_plot = save_stability_plot(diagnostics)
heatmap_path = save_prediction_heatmaps(grid_x, grid_y, grid_marginals, actual_grid, df_test)

@printf("Baseline MSE constant 0.52 = %.6f\n", baseline_mse)
@printf("Posterior predictive test MSE = %.6f\n", final_mse)
@printf("Mean posterior predictive std = %.6f\n", mean(sqrt.(max.(y_var, 0.0))))
@printf("Final τ = %.6f, Final obs_tau = %.6f\n", diagnostics.tau[end], diagnostics.obs_tau[end])
@printf("Final w_mean max norm = %.6f\n", diagnostics.w_mean_max_norm[end])
@printf("Final w_a max norm    = %.6f\n", diagnostics.w_a_max_norm[end])
println("Training diagnostics saved to $diagnostics_path")
println("Per-level out stats saved to $(OUTPUT_PREFIX)_level_stats.csv")
println("Test predictions saved to $prediction_path")
println("Free-energy plot saved to $free_energy_plot")
println("Stability plot saved to $stability_plot")
println("Posterior predictive heatmaps saved to $heatmap_path")

if MAKE_GIF
    gif_path = build_iteration_gif(training_result, grid_x, grid_y, features_grid, df_test,
                                    chain_var_val, prior_var)
    println("Per-iteration heatmap GIF saved to $gif_path")
end

println("\nPer-iteration diagnostics:")
for row in eachrow(diagnostics)
    @printf("  iter %2d: FE=%.3e  τ=%.2e  obs_tau=%.3e  |w_m|=%.3f  |w_a|=%.3f  train_mse(L)=%.5f\n",
        row.iteration, row.free_energy, row.tau, row.obs_tau,
        row.w_mean_max_norm, row.w_a_max_norm, row.train_mse_cleanend)
end

println("\nPer-level out statistics (final iteration):")
last_it = maximum(level_stats.iteration)
for row in eachrow(level_stats[level_stats.iteration .== last_it, :])
    @printf("  l=%d  mean|out|=%.3f  mean_std(out)=%.3f\n",
        row.level, row.mean_abs, row.mean_std)
end
