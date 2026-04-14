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

const DATASET = get(ENV, "XOR_DATASET", "test_dataset/xor_simple_dataset.csv")
const OUTPUT_PREFIX = get(ENV, "OUTPUT_PREFIX", "test_dataset/viz/xor_residual_plus_learn_obs_clean")
const N_NEURONS = parse(Int, get(ENV, "N_NEURONS", "8"))
const N_ITERATIONS = parse(Int, get(ENV, "N_ITERATIONS", "10"))
const TRAIN_FRACTION = parse(Float64, get(ENV, "TRAIN_FRACTION", "0.3"))
const VALIDATION_FRACTION = parse(Float64, get(ENV, "VALIDATION_FRACTION", "0.0"))
const OBS_TAU_SHAPE = parse(Float64, get(ENV, "OBS_TAU_SHAPE", "100.0"))
const OBS_TAU_RATE = parse(Float64, get(ENV, "OBS_TAU_RATE", "1.0"))
const TAU_SKIP = parse(Float64, get(ENV, "TAU_SKIP", "20.0"))
const RESIDUAL_SCALE = parse(Float64, get(ENV, "RESIDUAL_SCALE", "0.25"))
const PRIOR_SEED = parse(Int, get(ENV, "PRIOR_SEED", "42"))
const SPLIT_SEED = parse(Int, get(ENV, "SPLIT_SEED", "2027"))
const VALIDATION_SEED = parse(Int, get(ENV, "VALIDATION_SEED", "2028"))

@model function xor_residual_plus_learn_obs(n_neurons, features, residual_features, y, priors, tau_skip)
    local w_base, w_delta, w_gate, base, delta, mu, za, gamma, tau, obs_tau, out

    tau ~ priors[:tau]
    obs_tau ~ priors[:obs_tau]
    w_base ~ priors[:w_base]

    for k = 1:n_neurons
        w_delta[k] ~ priors[:w_delta][k]
        w_gate[k] ~ priors[:w_gate][k]
    end

    for j = 1:length(y)
        base[j] ~ softdot(features[j], w_base, tau)
        out[j] ~ NormalMeanPrecision(base[j], tau_skip)

        for k = 1:n_neurons
            delta[k, j] ~ softdot(residual_features[j], w_delta[k], tau)
            mu[k, j] := base[j] + delta[k, j]

            za[k, j] ~ softdot(features[j], w_gate[k], tau) where {meta = LowRankMeta()}
            gamma[k, j] ~ ReLU(za[k, j])
            out[j] ~ NormalMeanPrecision(mu[k, j], gamma[k, j])
        end

        y[j] ~ NormalMeanPrecision(out[j], obs_tau)
    end
end

@constraints function xor_residual_plus_learn_obs_constraints()
    q(w_base, w_delta, w_gate, base, delta, mu, za, gamma, tau, obs_tau, out) =
        q(w_base, base, w_delta, delta, mu, out)q(w_gate)q(za, gamma)q(tau)q(obs_tau)

    q(base)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(delta)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(mu)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(za)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())),
    )
    q(gamma)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())),
    )
end

@initialization function xor_residual_plus_learn_obs_init(priors)
    q(w_base) = priors[:w_base]
    q(w_delta) = deepcopy(priors[:w_delta])
    q(w_gate) = deepcopy(priors[:w_gate])
    q(base) = NormalMeanVariance(0.5, 1.0)
    q(delta) = NormalMeanVariance(0.0, 1.0)
    q(mu) = NormalMeanVariance(0.5, 1.0)
    q(za) = GammaShapeScale(2.0, 1.0)
    q(gamma) = GammaShapeScale(2.0, 1.0)
    q(tau) = priors[:tau]
    q(obs_tau) = priors[:obs_tau]
    q(out) = NormalMeanVariance(0.5, 1.0)

    # These message initializations start the loopy residual path.
    μ(base) = NormalMeanVariance(0.5, 10.0)
    μ(delta) = NormalMeanVariance(0.0, 10.0)
    μ(mu) = NormalMeanVariance(0.5, 10.0)
end

function make_weight_prior(rng, n_features, precision)
    return MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(precision, n_features)))
end

function make_priors(; n_neurons = N_NEURONS, n_features = 3, seed = PRIOR_SEED)
    rng = StableRNG(seed)

    w_base = make_weight_prior(rng, n_features, 1e-3)

    # Consume the unused direct-model mean priors after w_base so seed 42
    # reproduces the comparison run in xor_residual_relu_experiment.jl exactly.
    _unused_w_mean = [make_weight_prior(rng, n_features, 1e-4) for _ in 1:n_neurons]

    return Dict{Symbol,Any}(
        :w_base => w_base,
        :w_delta => [make_weight_prior(rng, n_features, 1e-3) for _ in 1:n_neurons],
        :w_gate => [make_weight_prior(rng, n_features, 1e-3) for _ in 1:n_neurons],
        :obs_tau => GammaShapeRate(OBS_TAU_SHAPE, OBS_TAU_RATE),
        :tau => GammaShapeRate(1e6, 1.0),
    )
end

build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]
build_residual_features(features) = [RESIDUAL_SCALE .* f for f in features]

function split_train_test(df)
    rng = StableRNG(SPLIT_SEED)
    idx = randperm(rng, nrow(df))
    n_train = round(Int, TRAIN_FRACTION * nrow(df))
    return df[idx[1:n_train], :], df[idx[(n_train + 1):end], :]
end

function split_fit_validation(df_train)
    if VALIDATION_FRACTION <= 0
        return df_train, nothing
    end

    rng = StableRNG(VALIDATION_SEED)
    idx = randperm(rng, nrow(df_train))
    n_val = max(1, round(Int, VALIDATION_FRACTION * nrow(df_train)))
    n_val = min(n_val, nrow(df_train) - 1)

    return df_train[idx[(n_val + 1):end], :], df_train[idx[1:n_val], :]
end

relu_gate(x) = max(0.0, x)

function predict_point(f, w_base, w_deltas, w_gates)
    base = dot(w_base, f)
    fres = RESIDUAL_SCALE .* f
    deltas = [dot(w, fres) for w in w_deltas]
    gammas = [relu_gate(dot(w, f)) for w in w_gates]
    total = TAU_SKIP + sum(gammas)
    return (TAU_SKIP * base + sum(gammas .* (base .+ deltas))) / total
end

function predictor_at(result, iteration)
    w_base = mean(result.posteriors[:w_base][iteration])
    w_deltas = [mean(result.posteriors[:w_delta][iteration][k]) for k in 1:N_NEURONS]
    w_gates = [mean(result.posteriors[:w_gate][iteration][k]) for k in 1:N_NEURONS]
    return f -> predict_point(f, w_base, w_deltas, w_gates)
end

function mse_for(predict_fn, features, y)
    y_pred = [predict_fn(f) for f in features]
    return mean((y_pred .- y) .^ 2)
end

function mse_trace(result, df_test, features_test, df_val, features_val)
    rows = DataFrame(iteration = Int[], test_mse = Float64[], validation_mse = Union{Missing,Float64}[])

    for iteration in 1:length(result.posteriors[:w_gate])
        predict_fn = predictor_at(result, iteration)
        test_mse = mse_for(predict_fn, features_test, df_test.OT)
        validation_mse = isnothing(df_val) ? missing : mse_for(predict_fn, features_val, df_val.OT)
        push!(rows, (iteration, test_mse, validation_mse))
    end

    return rows
end

function selected_iteration(trace)
    if any(!ismissing, trace.validation_mse)
        values = collect(skipmissing(trace.validation_mse))
        return trace.iteration[argmin(values)], "validation"
    end

    return trace.iteration[end], "final"
end

function save_heatmap(label, predict_fn, features_test, df_test)
    x_coords = [f[2] for f in features_test]
    y_coords = [f[3] for f in features_test]
    grid_x = range(minimum(x_coords) - 0.5, maximum(x_coords) + 0.5; length = 200)
    grid_y = range(minimum(y_coords) - 0.5, maximum(y_coords) + 0.5; length = 200)

    z_grid = [predict_fn([1.0, x, y]) for y in grid_y, x in grid_x]
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
        title = label,
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

    outpath = "$(OUTPUT_PREFIX)_$(label)_heatmap.png"
    plot(p1, p2; layout = (1, 2), size = (1000, 420), plot_title = "residual_plus_learn_obs")
    savefig(outpath)
    return outpath
end

function save_trace_plot(trace)
    outpath = "$(OUTPUT_PREFIX)_mse_by_iteration.png"
    p = plot(
        trace.iteration,
        trace.test_mse;
        marker = :circle,
        label = "test",
        xlabel = "iteration",
        ylabel = "MSE",
        title = "residual_plus_learn_obs MSE by iteration",
        size = (720, 420),
    )

    if any(!ismissing, trace.validation_mse)
        plot!(p, trace.iteration, collect(skipmissing(trace.validation_mse)); marker = :circle, label = "validation")
    end

    savefig(p, outpath)
    return outpath
end

function run_inference(df_fit, features_fit)
    priors = make_priors(n_neurons = N_NEURONS, seed = PRIOR_SEED)
    residual_features_fit = build_residual_features(features_fit)

    return infer(
        model = xor_residual_plus_learn_obs(
            n_neurons = N_NEURONS,
            priors = priors,
            tau_skip = TAU_SKIP,
        ),
        data = (
            y = df_fit.OT,
            features = features_fit,
            residual_features = residual_features_fit,
        ),
        constraints = xor_residual_plus_learn_obs_constraints(),
        initialization = xor_residual_plus_learn_obs_init(priors),
        iterations = N_ITERATIONS,
        free_energy = false,
        showprogress = true,
        options = (limit_stack_depth = 100,),
    )
end

function main()
    mkpath(dirname(OUTPUT_PREFIX))

    df = CSV.read(DATASET, DataFrame)
    df_train, df_test = split_train_test(df)
    df_fit, df_val = split_fit_validation(df_train)

    features_fit = build_features(df_fit)
    features_test = build_features(df_test)
    features_val = isnothing(df_val) ? nothing : build_features(df_val)

    println("=" ^ 78)
    println("XOR residual_plus_learn_obs")
    println("  dataset=$DATASET")
    println("  neurons=$N_NEURONS, iterations=$N_ITERATIONS")
    println("  train_fraction=$TRAIN_FRACTION, validation_fraction=$VALIDATION_FRACTION")
    println("  n_fit=$(nrow(df_fit)), n_validation=$(isnothing(df_val) ? 0 : nrow(df_val)), n_test=$(nrow(df_test))")
    println("  tau_skip=$TAU_SKIP, residual_scale=$RESIDUAL_SCALE")
    println("  obs_tau_prior=GammaShapeRate($OBS_TAU_SHAPE, $OBS_TAU_RATE)")
    println("=" ^ 78)

    result = run_inference(df_fit, features_fit)
    trace = mse_trace(result, df_test, features_test, df_val, features_val)

    final_iter = trace.iteration[end]
    final_test_mse = trace.test_mse[end]
    best_test_idx = argmin(trace.test_mse)
    best_test_iteration = trace.iteration[best_test_idx]
    best_test_mse = trace.test_mse[best_test_idx]
    chosen_iteration, selection_rule = selected_iteration(trace)
    chosen_test_mse = trace.test_mse[findfirst(==(chosen_iteration), trace.iteration)]
    learned_obs_tau = mean(result.posteriors[:obs_tau][end])

    final_heatmap = save_heatmap("final$(final_iter)", predictor_at(result, final_iter), features_test, df_test)
    selected_heatmap = save_heatmap("selected$(chosen_iteration)", predictor_at(result, chosen_iteration), features_test, df_test)
    trace_plot = save_trace_plot(trace)

    trace_path = "$(OUTPUT_PREFIX)_mse_by_iteration.csv"
    summary_path = "$(OUTPUT_PREFIX)_summary.csv"
    CSV.write(trace_path, trace)
    CSV.write(
        summary_path,
        DataFrame(
            final_iteration = [final_iter],
            final_test_mse = [final_test_mse],
            best_test_iteration = [best_test_iteration],
            best_test_mse = [best_test_mse],
            selected_iteration = [chosen_iteration],
            selected_by = [selection_rule],
            selected_test_mse = [chosen_test_mse],
            learned_obs_tau = [learned_obs_tau],
            trace_path = [trace_path],
            trace_plot = [trace_plot],
            final_heatmap = [final_heatmap],
            selected_heatmap = [selected_heatmap],
        ),
    )

    println("\nResults")
    @printf("  final test MSE = %.6f at iteration %d\n", final_test_mse, final_iter)
    @printf("  best diagnostic test MSE = %.6f at iteration %d\n", best_test_mse, best_test_iteration)
    @printf("  selected test MSE = %.6f at iteration %d selected_by=%s\n", chosen_test_mse, chosen_iteration, selection_rule)
    @printf("  learned obs_tau = %.6f\n", learned_obs_tau)
    println("  summary CSV: $summary_path")
    println("  trace CSV: $trace_path")
    println("  trace plot: $trace_plot")
    println("  final heatmap: $final_heatmap")
    println("  selected heatmap: $selected_heatmap")
end

main()
