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
using Printf
using Plots
using ProgressMeter

function print_usage()
    println("""
    Usage:
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl [projection_niterations]

    Options:
      --projection-iterations=N  Inner Manopt iterations used by ProjectedTo.
      --outer-iterations=N       Outer VMP iterations. Default: 100.
      --batch-size=N             Enable BONG-like online mini-batch mode.
      --batch-iterations=N       VMP iterations per mini-batch. Default: 1.
      --epochs=N                 Online passes over the train split. Default: 1.
      --sampling=epoch|random    Online batch schedule. Default: epoch.
      --random-subsample         Alias for --sampling=random.
      --updates=N                Number of random-subsample online updates.
      --early-stop-window=K      Stop if FE improves too little over K steps. Default: 0 disables.
      --early-stop-rtol=X        Relative FE improvement threshold. Default: 1e-4.
      --early-stop-atol=X        Absolute FE improvement threshold. Default: 0.0.
      --early-stop-min-steps=N   Do not stop before N FE observations. Default: 0.
      --early-stop-patience=N    Require N consecutive failed FE checks. Default: 1.
      --no-early-stop            Disable FE early stopping.
      --neurons=N                Number of ReLU-gated experts. Default: 16.
      --dataset=PATH             Dataset CSV. Default: test_dataset/xor_simple_dataset.csv.
      --train-fraction=X         Train fraction. Default: 0.3.
      --prior-seed=N             Prior seed. Default: 42.
      --split-seed=N             Train/test split seed. Default: 2027.
      --output-prefix=PATH       Output path prefix. Default: test_dataset/viz/relu_projection_niter_N.
      --no-shuffle               Preserve train split order in mini-batch mode.
      --no-progress              Disable the online-mode progress bar.
      --no-plots                 Skip heatmap and GIF generation.
      --no-animation             Save heatmap only.
      --help                     Show this message.

    Examples:
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl 1
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=5 --outer-iterations=100
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32 --sampling=random --updates=100
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --early-stop-window=5 --early-stop-rtol=1e-3
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32 --batch-iterations=5 --sampling=random --epochs=100 --early-stop-window=75 --early-stop-min-steps=750 --early-stop-patience=10
    """)
end

function parse_cli(args)
    batch_size_env = get(ENV, "BATCH_SIZE", "")
    updates_env = get(ENV, "UPDATES", "")
    config = Dict{Symbol,Any}(
        :projection_niterations => parse(Int, get(ENV, "PROJECTION_NITERATIONS", "100")),
        :outer_iterations => parse(Int, get(ENV, "OUTER_ITERATIONS", "100")),
        :batch_size => isempty(batch_size_env) ? nothing : parse(Int, batch_size_env),
        :batch_iterations => parse(Int, get(ENV, "BATCH_ITERATIONS", "1")),
        :epochs => parse(Int, get(ENV, "EPOCHS", "1")),
        :batch_sampling => get(ENV, "BATCH_SAMPLING", "epoch"),
        :updates => isempty(updates_env) ? nothing : parse(Int, updates_env),
        :early_stop_window => parse(Int, get(ENV, "EARLY_STOP_WINDOW", "0")),
        :early_stop_rtol => parse(Float64, get(ENV, "EARLY_STOP_RTOL", "1e-4")),
        :early_stop_atol => parse(Float64, get(ENV, "EARLY_STOP_ATOL", "0.0")),
        :early_stop_min_steps => parse(Int, get(ENV, "EARLY_STOP_MIN_STEPS", "0")),
        :early_stop_patience => parse(Int, get(ENV, "EARLY_STOP_PATIENCE", "1")),
        :n_neurons => parse(Int, get(ENV, "N_NEURONS", "16")),
        :dataset => get(ENV, "XOR_DATASET", "test_dataset/xor_simple_dataset.csv"),
        :train_fraction => parse(Float64, get(ENV, "TRAIN_FRACTION", "0.3")),
        :prior_seed => parse(Int, get(ENV, "PRIOR_SEED", "42")),
        :split_seed => parse(Int, get(ENV, "SPLIT_SEED", "2027")),
        :output_prefix => nothing,
        :shuffle_batches => true,
        :show_progress => true,
        :save_plots => true,
        :save_animation => true,
    )

    positional = String[]
    for arg in args
        if arg in ("--help", "-h")
            print_usage()
            exit(0)
        elseif startswith(arg, "--projection-iterations=")
            config[:projection_niterations] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--outer-iterations=")
            config[:outer_iterations] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--batch-size=")
            config[:batch_size] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--batch-iterations=")
            config[:batch_iterations] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--epochs=")
            config[:epochs] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--sampling=")
            config[:batch_sampling] = split(arg, "=", limit = 2)[2]
        elseif arg == "--random-subsample"
            config[:batch_sampling] = "random"
        elseif startswith(arg, "--updates=")
            config[:updates] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--early-stop-window=")
            config[:early_stop_window] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--early-stop-rtol=")
            config[:early_stop_rtol] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--early-stop-atol=")
            config[:early_stop_atol] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--early-stop-min-steps=")
            config[:early_stop_min_steps] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--early-stop-patience=")
            config[:early_stop_patience] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif arg == "--no-early-stop"
            config[:early_stop_window] = 0
        elseif startswith(arg, "--neurons=")
            config[:n_neurons] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--dataset=")
            config[:dataset] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--train-fraction=")
            config[:train_fraction] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--prior-seed=")
            config[:prior_seed] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--split-seed=")
            config[:split_seed] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output-prefix=")
            config[:output_prefix] = split(arg, "=", limit = 2)[2]
        elseif arg == "--no-shuffle"
            config[:shuffle_batches] = false
        elseif arg == "--no-progress"
            config[:show_progress] = false
        elseif arg == "--no-plots"
            config[:save_plots] = false
            config[:save_animation] = false
        elseif arg == "--no-animation"
            config[:save_animation] = false
        elseif startswith(arg, "--")
            error("Unknown argument: $arg")
        else
            push!(positional, arg)
        end
    end

    if !isempty(positional)
        config[:projection_niterations] = parse(Int, positional[1])
    end
    if length(positional) > 1
        error("Unexpected positional arguments: $(join(positional[2:end], ", "))")
    end

    if isnothing(config[:output_prefix])
        niter = config[:projection_niterations]
        if isnothing(config[:batch_size])
            config[:output_prefix] = "test_dataset/viz/relu_projection_niter_$(niter)"
        else
            batch_size = config[:batch_size]
            batch_iterations = config[:batch_iterations]
            sampling = config[:batch_sampling]
            config[:output_prefix] = "test_dataset/viz/relu_projection_niter_$(niter)_batch_$(batch_size)_biter_$(batch_iterations)_$(sampling)"
        end
    end

    return config
end

@model function xor_relu_direct_projection_niterations(n_neurons, features, y, priors)
    local w_mean, w_a, z_mean, za, gamma, tau, out

    tau ~ priors[:tau]

    for k = 1:n_neurons
        w_mean[k] ~ priors[:w_mean][k]
        w_a[k] ~ priors[:w_a][k]
    end

    for j = 1:length(y)
        for k = 1:n_neurons
            z_mean[k, j] ~ softdot(features[j], w_mean[k], tau)
            za[k, j] ~ softdot(features[j], w_a[k], tau) where {meta = LowRankMeta()}
            gamma[k, j] ~ ReLU(za[k, j])
            out[j] ~ NormalMeanPrecision(z_mean[k, j], gamma[k, j])
        end
        y[j] ~ NormalMeanPrecision(out[j], 1e6)
    end
end

@constraints function xor_relu_direct_projection_niterations_constraints(projection_niterations)
    q(w_mean, w_a, z_mean, za, gamma, tau, out) = q(w_mean, z_mean)q(w_a)q(za)q(gamma)q(tau)q(out)
    q(za)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(EnzymeBackend()),
            niterations = projection_niterations,
        ),
    )
    q(gamma)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(EnzymeBackend()),
            niterations = projection_niterations,
        ),
    )
end

@initialization function xor_relu_direct_projection_niterations_init(priors)
    q(w_mean) = deepcopy(priors[:w_mean])
    q(w_a) = deepcopy(priors[:w_a])
    q(z_mean) = NormalMeanVariance(0.0, 1.0)
    q(za) = GammaShapeScale(2.0, 1.0)
    q(gamma) = GammaShapeScale(2.0, 1.0)
    q(tau) = priors[:tau]
end

function make_priors(; n_neurons = 4, n_features::Int = 3, seed::Int = 42)
    rng = StableRNG(seed)

    w_mean = [
        MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-4, n_features)))
        for _ in 1:n_neurons
    ]
    w_a = [
        MvNormalWeightedMeanPrecision(randn(rng, n_features), Diagonal(fill(1e-3, n_features)))
        for _ in 1:n_neurons
    ]

    return Dict{Symbol,Any}(
        :w_mean => w_mean,
        :w_a => w_a,
        :tau => GammaShapeRate(1e6, 1.0),
    )
end

build_features(df) = [[1.0, df.x1[i], df.x2[i]] for i in 1:nrow(df)]

function split_dataset(df; train_fraction = 0.3, seed = 2026)
    rng = StableRNG(seed)
    idx = randperm(rng, nrow(df))
    n = round(Int, train_fraction * nrow(df))
    return df[idx[1:n], :], df[idx[n+1:end], :]
end

function predict_point(x1, x2, w_means, w_as)
    f = [1.0, x1, x2]
    mus = [dot(w_means[k], f) for k in eachindex(w_means)]
    precisions = [max(0.0, dot(w_as[k], f)) for k in eachindex(w_as)]
    total = sum(precisions)
    return total > eps(Float64) ? sum(precisions .* mus) / total : NaN
end

function predict_features(features, w_means, w_as)
    return map(features) do f
        predict_point(f[2], f[3], w_means, w_as)
    end
end

function gate_masses(features, w_as)
    return map(features) do f
        sum(max(0.0, dot(w_a, f)) for w_a in w_as)
    end
end

function finite_mse(y_pred, y_true)
    valid = findall(isfinite, y_pred)
    if length(valid) != length(y_pred)
        return NaN, length(y_pred) - length(valid)
    end
    return mean((y_pred .- y_true) .^ 2), 0
end

function per_iteration_metrics(result, features_test, y_test, n_neurons)
    n_iters = length(result.posteriors[:w_mean])
    free_energy = result.free_energy

    rows = DataFrame(
        iteration = Int[],
        free_energy = Float64[],
        test_mse = Float64[],
        invalid_predictions = Int[],
        mean_gate_mass = Float64[],
        min_gate_mass = Float64[],
    )

    for iter in 1:n_iters
        w_means = [mean(result.posteriors[:w_mean][iter][k]) for k in 1:n_neurons]
        w_as = [mean(result.posteriors[:w_a][iter][k]) for k in 1:n_neurons]

        y_pred = predict_features(features_test, w_means, w_as)
        mse, invalid = finite_mse(y_pred, y_test)
        masses = gate_masses(features_test, w_as)
        fe = iter <= length(free_energy) ? Float64(free_energy[iter]) : NaN

        push!(rows, (
            iteration = iter,
            free_energy = fe,
            test_mse = mse,
            invalid_predictions = invalid,
            mean_gate_mass = mean(masses),
            min_gate_mass = minimum(masses),
        ))
    end

    return rows
end

function best_metric_row(metrics)
    valid = findall(isfinite, metrics.test_mse)
    isempty(valid) && return nothing
    best_valid_index = argmin(metrics.test_mse[valid])
    return metrics[valid[best_valid_index], :]
end

mutable struct WindowedFreeEnergyStopper
    window::Int
    rtol::Float64
    atol::Float64
    min_steps::Int
    patience::Int
    values::Vector{Float64}
    failed_checks::Int
    stopped::Bool
    reason::String
end

function WindowedFreeEnergyStopper(
    window::Int,
    rtol::Real,
    atol::Real,
    min_steps::Int,
    patience::Int,
)
    return WindowedFreeEnergyStopper(
        window,
        Float64(rtol),
        Float64(atol),
        min_steps,
        patience,
        Float64[],
        0,
        false,
        "",
    )
end

function early_stopping_enabled(config)
    return config[:early_stop_window] > 0
end

function make_early_stopper(config)
    if !early_stopping_enabled(config)
        return nothing
    end

    window = config[:early_stop_window]
    rtol = config[:early_stop_rtol]
    atol = config[:early_stop_atol]
    min_steps = config[:early_stop_min_steps]
    patience = config[:early_stop_patience]

    if window <= 0
        return nothing
    end
    if rtol < 0
        error("--early-stop-rtol must be non-negative")
    end
    if atol < 0
        error("--early-stop-atol must be non-negative")
    end
    if min_steps < 0
        error("--early-stop-min-steps must be non-negative")
    end
    if patience <= 0
        error("--early-stop-patience must be positive")
    end

    return WindowedFreeEnergyStopper(window, rtol, atol, min_steps, patience)
end

function current_bethe_free_energy(model)
    current_fe = Ref(0.0)
    subscribe!(
        score(model, RxInfer.BetheFreeEnergy(Real), RxInfer.DefaultObjectiveDiagnosticChecks) |> take(1),
        value -> current_fe[] = Float64(value),
    )
    return current_fe[]
end

function update_early_stopper!(stopper::WindowedFreeEnergyStopper, current_fe, step)
    fe = Float64(current_fe)
    push!(stopper.values, fe)
    global_step = length(stopper.values)

    if global_step <= stopper.window || global_step < stopper.min_steps
        return false
    end

    reference_fe = stopper.values[end - stopper.window]
    if !isfinite(fe) || !isfinite(reference_fe)
        return false
    end

    improvement = reference_fe - fe
    scale = max(abs(reference_fe), eps(Float64))
    relative_improvement = improvement / scale
    threshold = max(stopper.atol, stopper.rtol * scale)
    relative_threshold = threshold / scale
    failed_check = improvement <= threshold

    if failed_check
        stopper.failed_checks += 1
    else
        stopper.failed_checks = 0
    end

    should_stop = stopper.failed_checks >= stopper.patience

    if should_stop
        stopper.stopped = true
        stopper.reason = @sprintf(
            "Early stopping at FE observation %d: FE %.6g vs %.6g %d observations ago, relative improvement %.6g <= %.6g for %d/%d consecutive checks",
            global_step,
            fe,
            reference_fe,
            stopper.window,
            relative_improvement,
            relative_threshold,
            stopper.failed_checks,
            stopper.patience,
        )
    end

    return should_stop
end

function (stopper::WindowedFreeEnergyStopper)(model, iteration::Int)
    return update_early_stopper!(stopper, current_bethe_free_energy(model), iteration)
end

function posterior_weight_history(result, n_neurons)
    n_states = length(result.posteriors[:w_mean])
    w_mean_history = Vector{Any}(undef, n_states)
    w_a_history = Vector{Any}(undef, n_states)

    for state in 1:n_states
        w_mean_history[state] = [mean(result.posteriors[:w_mean][state][k]) for k in 1:n_neurons]
        w_a_history[state] = [mean(result.posteriors[:w_a][state][k]) for k in 1:n_neurons]
    end

    return w_mean_history, w_a_history
end

function save_visuals_from_history(
    w_mean_history,
    w_a_history,
    config,
    features_test;
    frame_label = "VMP iter",
)
    output_prefix = config[:output_prefix]
    mkpath(dirname(output_prefix))

    x_test_coords = [f[2] for f in features_test]
    y_test_coords = [f[3] for f in features_test]

    grid_x = range(minimum(x_test_coords) - 0.5, maximum(x_test_coords) + 0.5; length = 200)
    grid_y = range(minimum(y_test_coords) - 0.5, maximum(y_test_coords) + 0.5; length = 200)
    z_actual = [Float64(x * y < 0) for y in grid_y, x in grid_x]

    w_means_post = w_mean_history[end]
    w_a_post = w_a_history[end]
    z_grid = [predict_point(x, y, w_means_post, w_a_post) for y in grid_y, x in grid_x]

    p1 = contourf(
        grid_x,
        grid_y,
        z_grid,
        c = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Predicted",
        linewidth = 0,
    )

    p2 = contourf(
        grid_x,
        grid_y,
        z_actual,
        c = :RdBu,
        levels = 20,
        clims = (0, 1),
        xlabel = "x1",
        ylabel = "x2",
        title = "Actual XOR",
        linewidth = 0,
    )

    plot(
        p1,
        p2,
        layout = (1, 2),
        size = (1000, 400),
        plot_title = "XOR ReLU, projection niterations=$(config[:projection_niterations])",
    )
    heatmap_path = "$(output_prefix)_heatmap.png"
    savefig(heatmap_path)
    println("Heatmap saved to $heatmap_path")

    if config[:save_animation]
        n_states = length(w_mean_history)

        anim = @animate for iter in 1:n_states
            wm_iter = w_mean_history[iter]
            wa_iter = w_a_history[iter]
            z_iter = [predict_point(x, y, wm_iter, wa_iter) for y in grid_y, x in grid_x]

            p_iter = contourf(
                grid_x,
                grid_y,
                z_iter,
                c = :RdBu,
                levels = 20,
                clims = (0, 1),
                xlabel = "x1",
                ylabel = "x2",
                title = "Predicted  ($frame_label $iter / $n_states)",
                linewidth = 0,
            )

            p_actual = contourf(
                grid_x,
                grid_y,
                z_actual,
                c = :RdBu,
                levels = 20,
                clims = (0, 1),
                xlabel = "x1",
                ylabel = "x2",
                title = "Actual XOR",
                linewidth = 0,
            )

            plot(
                p_iter,
                p_actual,
                layout = (1, 2),
                size = (1000, 400),
                plot_title = "XOR ReLU, projection niterations=$(config[:projection_niterations]), $frame_label $iter",
            )
        end

        gif_path = "$(output_prefix)_iterations.gif"
        gif(anim, gif_path, fps = 5)
        println("Animation saved to $gif_path")
    end
end

function save_visuals(result, config, features_test, y_test, n_neurons)
    w_mean_history, w_a_history = posterior_weight_history(result, n_neurons)
    return save_visuals_from_history(
        w_mean_history,
        w_a_history,
        config,
        features_test;
        frame_label = "VMP iter",
    )
end

function update_priors_from_result(result)
    return Dict{Symbol,Any}(
        :w_mean => deepcopy(result.posteriors[:w_mean][end]),
        :w_a => deepcopy(result.posteriors[:w_a][end]),
        :tau => deepcopy(result.posteriors[:tau][end]),
    )
end

function weights_from_priors(priors, n_neurons)
    w_means = [mean(priors[:w_mean][k]) for k in 1:n_neurons]
    w_as = [mean(priors[:w_a][k]) for k in 1:n_neurons]
    return w_means, w_as
end

function append_online_metric!(
    metrics,
    update,
    epoch,
    batch,
    batch_size,
    free_energy,
    features_test,
    y_test,
    w_means,
    w_as,
)
    y_pred = predict_features(features_test, w_means, w_as)
    mse, invalid = finite_mse(y_pred, y_test)
    masses = gate_masses(features_test, w_as)

    push!(metrics, (
        update = update,
        epoch = epoch,
        batch = batch,
        batch_size = batch_size,
        free_energy = free_energy,
        test_mse = mse,
        invalid_predictions = invalid,
        mean_gate_mass = mean(masses),
        min_gate_mass = minimum(masses),
    ))
end

function online_update_plan(config, n_train)
    batch_size = config[:batch_size]
    updates_per_epoch = cld(n_train, batch_size)

    if config[:batch_sampling] == "random"
        n_updates = isnothing(config[:updates]) ? updates_per_epoch * config[:epochs] : config[:updates]
    else
        n_updates = updates_per_epoch * config[:epochs]
    end

    return n_updates, updates_per_epoch
end

function infer_callbacks(early_stopper)
    return isnothing(early_stopper) ? nothing : (after_iteration = early_stopper,)
end

function report_online_update!(progress, row, completed_iterations, total_updates)
    if isnothing(progress)
        @printf(
            "update=%03d/%03d epoch=%d batch=%d n=%d inner=%d fe=%.4f test_mse=%.6f invalid=%d\n",
            row.update,
            total_updates,
            row.epoch,
            row.batch,
            row.batch_size,
            completed_iterations,
            row.free_energy,
            row.test_mse,
            row.invalid_predictions,
        )
        return nothing
    end

    showvalues = [
        (:update, "$(row.update)/$total_updates"),
        (:epoch, row.epoch),
        (:batch, row.batch),
        (:batch_size, row.batch_size),
        (:inner, completed_iterations),
        (:test_mse, row.test_mse),
        (:invalid, row.invalid_predictions),
    ]

    for _ in 1:completed_iterations
        ProgressMeter.next!(progress; showvalues = showvalues)
    end

    return nothing
end

function run_online_minibatch_inference(
    config,
    initial_priors,
    df_train,
    features_train,
    features_test,
    y_test,
)
    n_neurons = config[:n_neurons]
    projection_niterations = config[:projection_niterations]
    batch_size = config[:batch_size]
    batch_iterations = config[:batch_iterations]
    epochs = config[:epochs]
    batch_sampling = config[:batch_sampling]
    early_stopper = make_early_stopper(config)
    callbacks = infer_callbacks(early_stopper)

    if batch_size <= 0
        error("--batch-size must be positive")
    end
    if batch_iterations <= 0
        error("--batch-iterations must be positive")
    end
    if epochs <= 0
        error("--epochs must be positive")
    end
    if !(batch_sampling in ("epoch", "random"))
        error("--sampling must be either epoch or random")
    end

    if batch_sampling == "epoch" && epochs > 1
        println("Warning: epochs=$epochs revisits the same train split and therefore counts data more than once.")
    elseif batch_sampling == "random"
        println("Random-subsample mode draws a fresh random subset each update; examples may reappear across updates.")
    end
    if !isnothing(early_stopper) && batch_sampling == "random"
        println("Warning: random-subsample free energy is noisy because each update sees a different subset; use --early-stop-min-steps and --early-stop-patience or disable FE early stopping.")
    end
    if batch_sampling == "epoch" && !isnothing(config[:updates])
        println("Note: --updates is ignored with --sampling=epoch; use --sampling=random to control update count directly.")
    end

    metrics = DataFrame(
        update = Int[],
        epoch = Int[],
        batch = Int[],
        batch_size = Int[],
        free_energy = Float64[],
        test_mse = Float64[],
        invalid_predictions = Int[],
        mean_gate_mass = Float64[],
        min_gate_mass = Float64[],
    )

    current_priors = deepcopy(initial_priors)
    w_mean_history = Any[]
    w_a_history = Any[]
    update = 0
    n_train = nrow(df_train)
    n_updates, updates_per_epoch = online_update_plan(config, n_train)
    if n_updates <= 0
        error("planned online update count must be positive")
    end
    total_inner_steps = n_updates * batch_iterations
    progress = config[:show_progress] ? Progress(
        total_inner_steps;
        desc = "Online mini-batch VMP",
        dt = 0.5,
    ) : nothing

    if batch_sampling == "random"
        if batch_size > n_train
            error("--batch-size must be <= number of training rows for --sampling=random")
        end

        rng = StableRNG(config[:split_seed] + 10_000)
        for update in 1:n_updates
            epoch = cld(update, updates_per_epoch)
            batch_number = update - (epoch - 1) * updates_per_epoch
            batch_indices = randperm(rng, n_train)[1:batch_size]

            batch_features = features_train[batch_indices]
            batch_y = Vector(df_train.OT[batch_indices])

            result = infer(
                model = xor_relu_direct_projection_niterations(
                    n_neurons = n_neurons,
                    priors = current_priors,
                ),
                data = (y = batch_y, features = batch_features),
                constraints = xor_relu_direct_projection_niterations_constraints(projection_niterations),
                initialization = xor_relu_direct_projection_niterations_init(current_priors),
                iterations = batch_iterations,
                free_energy = true,
                showprogress = false,
                callbacks = callbacks,
                options = (limit_stack_depth = 100,),
            )

            current_priors = update_priors_from_result(result)
            w_means, w_as = weights_from_priors(current_priors, n_neurons)
            push!(w_mean_history, w_means)
            push!(w_a_history, w_as)

            completed_iterations = length(result.free_energy)
            fe = isempty(result.free_energy) ? NaN : Float64(result.free_energy[end])
            append_online_metric!(
                metrics,
                update,
                epoch,
                batch_number,
                length(batch_indices),
                fe,
                features_test,
                y_test,
                w_means,
                w_as,
            )

            row = metrics[end, :]
            report_online_update!(progress, row, completed_iterations, n_updates)
            if !isnothing(early_stopper) && early_stopper.stopped
                !isnothing(progress) && ProgressMeter.cancel(progress, "Early stopped")
                println(early_stopper.reason)
                return current_priors, metrics, w_mean_history, w_a_history
            end
        end

        return current_priors, metrics, w_mean_history, w_a_history
    end

    for epoch in 1:epochs
        order = collect(1:n_train)
        if config[:shuffle_batches]
            shuffle!(StableRNG(config[:split_seed] + epoch), order)
        end

        batch_number = 0
        for first_idx in 1:batch_size:n_train
            batch_number += 1
            update += 1
            last_idx = min(first_idx + batch_size - 1, n_train)
            batch_indices = order[first_idx:last_idx]

            batch_features = features_train[batch_indices]
            batch_y = Vector(df_train.OT[batch_indices])

            result = infer(
                model = xor_relu_direct_projection_niterations(
                    n_neurons = n_neurons,
                    priors = current_priors,
                ),
                data = (y = batch_y, features = batch_features),
                constraints = xor_relu_direct_projection_niterations_constraints(projection_niterations),
                initialization = xor_relu_direct_projection_niterations_init(current_priors),
                iterations = batch_iterations,
                free_energy = true,
                showprogress = false,
                callbacks = callbacks,
                options = (limit_stack_depth = 100,),
            )

            current_priors = update_priors_from_result(result)
            w_means, w_as = weights_from_priors(current_priors, n_neurons)
            push!(w_mean_history, w_means)
            push!(w_a_history, w_as)

            completed_iterations = length(result.free_energy)
            fe = isempty(result.free_energy) ? NaN : Float64(result.free_energy[end])
            append_online_metric!(
                metrics,
                update,
                epoch,
                batch_number,
                length(batch_indices),
                fe,
                features_test,
                y_test,
                w_means,
                w_as,
            )

            row = metrics[end, :]
            report_online_update!(progress, row, completed_iterations, n_updates)
            if !isnothing(early_stopper) && early_stopper.stopped
                !isnothing(progress) && ProgressMeter.cancel(progress, "Early stopped")
                println(early_stopper.reason)
                return current_priors, metrics, w_mean_history, w_a_history
            end
        end
    end

    return current_priors, metrics, w_mean_history, w_a_history
end

function main(args)
    config = parse_cli(args)

    n_neurons = config[:n_neurons]
    projection_niterations = config[:projection_niterations]
    outer_iterations = config[:outer_iterations]
    online_mode = !isnothing(config[:batch_size])

    df = CSV.read(config[:dataset], DataFrame)
    df_train, df_test = split_dataset(
        df;
        train_fraction = config[:train_fraction],
        seed = config[:split_seed],
    )

    println("=" ^ 78)
    if online_mode
        planned_updates, _ = online_update_plan(config, nrow(df_train))
        println("XOR ReLU Direct: BONG-like mini-batch VMP")
        println("  projection_niterations=$projection_niterations")
        println("  neurons=$n_neurons, batch_size=$(config[:batch_size]), batch_iterations=$(config[:batch_iterations])")
        println("  sampling=$(config[:batch_sampling]), epochs=$(config[:epochs]), updates=$(config[:updates])")
        println("  planned_online_updates=$planned_updates, progress_steps=$(planned_updates * config[:batch_iterations])")
    else
        println("XOR ReLU Direct: full-batch VMP")
        println("  projection_niterations=$projection_niterations")
        println("  neurons=$n_neurons, outer_iterations=$outer_iterations")
    end
    if early_stopping_enabled(config)
        println("  early_stop_window=$(config[:early_stop_window]), early_stop_rtol=$(config[:early_stop_rtol]), early_stop_atol=$(config[:early_stop_atol])")
        println("  early_stop_min_steps=$(config[:early_stop_min_steps]), early_stop_patience=$(config[:early_stop_patience])")
    end
    println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
    println("  dataset=$(config[:dataset])")
    println("=" ^ 78)

    priors = make_priors(n_neurons = n_neurons, seed = config[:prior_seed])
    features_train = build_features(df_train)
    features_test = build_features(df_test)
    y_test = df_test.OT

    if online_mode
        final_priors, metrics, w_mean_history, w_a_history = run_online_minibatch_inference(
            config,
            priors,
            df_train,
            features_train,
            features_test,
            y_test,
        )

        output_prefix = config[:output_prefix]
        mkpath(dirname(output_prefix))
        metrics_path = "$(output_prefix)_metrics.csv"
        CSV.write(metrics_path, metrics)

        final_row = metrics[end, :]
        best_row = best_metric_row(metrics)
        mse_constant = mean((0.52 .- y_test) .^ 2)

        println("\nLearned weights after final mini-batch update:")
        w_means_final, w_as_final = weights_from_priors(final_priors, n_neurons)
        for k in 1:n_neurons
            wm = round.(w_means_final[k]; digits = 4)
            wa = round.(w_as_final[k]; digits = 4)
            println("  neuron $k: w_mean=$wm  w_a=$wa")
        end

        println("\nBONG-like mini-batch VMP")
        println("  projection niterations = $projection_niterations")
        println("  batch size = $(config[:batch_size]), batch iterations = $(config[:batch_iterations])")
        println("  sampling = $(config[:batch_sampling])")
        @printf("Final test MSE = %.6f, invalid predictions = %d\n", final_row.test_mse, final_row.invalid_predictions)
        if !isnothing(best_row)
            @printf("Best test MSE  = %.6f at online update %d\n", best_row.test_mse, best_row.update)
        end
        @printf("Baseline MSE   = %.6f for constant 0.52\n", mse_constant)
        println("Metrics saved to $metrics_path")

        if config[:save_plots]
            save_visuals_from_history(
                w_mean_history,
                w_a_history,
                config,
                features_test;
                frame_label = "online update",
            )
        end

        return nothing
    end

    early_stopper = make_early_stopper(config)
    result = infer(
        model = xor_relu_direct_projection_niterations(
            n_neurons = n_neurons,
            priors = priors,
        ),
        data = (y = df_train.OT, features = features_train),
        constraints = xor_relu_direct_projection_niterations_constraints(projection_niterations),
        initialization = xor_relu_direct_projection_niterations_init(priors),
        iterations = outer_iterations,
        free_energy = true,
        showprogress = true,
        callbacks = infer_callbacks(early_stopper),
        options = (limit_stack_depth = 100,),
    )
    if !isnothing(early_stopper) && early_stopper.stopped
        println(early_stopper.reason)
    end

    println("\nLearned weights at final iteration:")
    for k in 1:n_neurons
        wm = round.(mean(result.posteriors[:w_mean][end][k]); digits = 4)
        wa = round.(mean(result.posteriors[:w_a][end][k]); digits = 4)
        println("  neuron $k: w_mean=$wm  w_a=$wa")
    end

    metrics = per_iteration_metrics(result, features_test, y_test, n_neurons)
    output_prefix = config[:output_prefix]
    mkpath(dirname(output_prefix))
    metrics_path = "$(output_prefix)_metrics.csv"
    CSV.write(metrics_path, metrics)

    final_row = metrics[end, :]
    best_row = best_metric_row(metrics)
    mse_constant = mean((0.52 .- y_test) .^ 2)

    println("\nProjection niterations = $projection_niterations")
    @printf("Final test MSE = %.6f, invalid predictions = %d\n", final_row.test_mse, final_row.invalid_predictions)
    if !isnothing(best_row)
        @printf("Best test MSE  = %.6f at VMP iteration %d\n", best_row.test_mse, best_row.iteration)
    end
    @printf("Baseline MSE   = %.6f for constant 0.52\n", mse_constant)
    println("Metrics saved to $metrics_path")

    if config[:save_plots]
        save_visuals(result, config, features_test, y_test, n_neurons)
    end
end

main(ARGS)
