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
import BayesBase: weightedmean, invcov

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
      --obs-precision=X          Fixed observation precision. Default: 1e6.
      --prior-forgetting-rho=X   Scale posterior natural parameters before the next online update. Default: 1.0.
      --w-mean-forgetting-rho=X  Override forgetting rho for w_mean only. Default: --prior-forgetting-rho.
      --w-a-forgetting-rho=X     Override forgetting rho for w_a only. Default: --prior-forgetting-rho.
      --posterior-damping-alpha=X
                                  Natural-parameter posterior step size for w_mean and w_a. Default: 1.0.
      --w-mean-damping-alpha=X   Override damping alpha for w_mean only. Default: --posterior-damping-alpha.
      --w-a-damping-alpha=X      Override damping alpha for w_a only. Default: --posterior-damping-alpha.
      --svi-likelihood-scaling   Scale online candidate obs precision by n_train / batch_size.
      --likelihood-scale=X       Extra multiplier for online candidate obs precision. Default: 1.0.
      --likelihood-tempering-beta=X
                                  Extra tempering multiplier for online candidate obs precision. Default: 1.0.
      --candidate-prior=current|initial
                                  Prior used inside each mini-batch candidate inference. Default: current.
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
      --prior-parameterization=weighted|mean|zero
                                  Weighted keeps old behavior. Mean treats random init as ordinary mean. Zero uses zero mean. Default: weighted.
      --w-mean-prior-precision=X Diagonal precision for w_mean priors. Default: 1e-4.
      --w-a-prior-precision=X    Diagonal precision for w_a priors. Default: 1e-3.
      --split-seed=N             Train/test split seed. Default: 2027.
      --output-prefix=PATH       Output path prefix. Default: test_dataset/viz/relu_projection_niter_N.
      --no-shuffle               Preserve train split order in mini-batch mode.
      --no-progress              Disable the online-mode progress bar.
      --report-every=N           With --no-progress, print every N updates. Use 0 for only summary. Default: 1.
      --no-plots                 Skip heatmap and GIF generation.
      --no-animation             Save heatmap only.
      --help                     Show this message.

    Examples:
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl 1
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=5 --outer-iterations=100
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32 --sampling=random --updates=100
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32 --obs-precision=1e4 --prior-forgetting-rho=0.9
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --early-stop-window=5 --early-stop-rtol=1e-3
      julia --project=. test_dataset/xor_relu_direct_projection_niterations.jl --projection-iterations=1 --batch-size=32 --batch-iterations=5 --sampling=random --epochs=100 --early-stop-window=75 --early-stop-min-steps=750 --early-stop-patience=10
    """)
end

function numeric_path_token(value)
    token = @sprintf("%.4g", Float64(value))
    token = replace(token, "." => "p")
    token = replace(token, "+" => "")
    token = replace(token, "-" => "m")
    return token
end

function parse_cli(args)
    batch_size_env = get(ENV, "BATCH_SIZE", "")
    updates_env = get(ENV, "UPDATES", "")
    w_mean_forgetting_env = get(ENV, "W_MEAN_FORGETTING_RHO", "")
    w_a_forgetting_env = get(ENV, "W_A_FORGETTING_RHO", "")
    posterior_damping_env = get(ENV, "POSTERIOR_DAMPING_ALPHA", "1.0")
    w_mean_damping_env = get(ENV, "W_MEAN_DAMPING_ALPHA", "")
    w_a_damping_env = get(ENV, "W_A_DAMPING_ALPHA", "")
    config = Dict{Symbol,Any}(
        :projection_niterations => parse(Int, get(ENV, "PROJECTION_NITERATIONS", "100")),
        :outer_iterations => parse(Int, get(ENV, "OUTER_ITERATIONS", "100")),
        :batch_size => isempty(batch_size_env) ? nothing : parse(Int, batch_size_env),
        :batch_iterations => parse(Int, get(ENV, "BATCH_ITERATIONS", "1")),
        :epochs => parse(Int, get(ENV, "EPOCHS", "1")),
        :batch_sampling => get(ENV, "BATCH_SAMPLING", "epoch"),
        :updates => isempty(updates_env) ? nothing : parse(Int, updates_env),
        :obs_precision => parse(Float64, get(ENV, "OBS_PRECISION", "1e6")),
        :prior_forgetting_rho => parse(Float64, get(ENV, "PRIOR_FORGETTING_RHO", "1.0")),
        :w_mean_forgetting_rho => isempty(w_mean_forgetting_env) ? nothing : parse(Float64, w_mean_forgetting_env),
        :w_a_forgetting_rho => isempty(w_a_forgetting_env) ? nothing : parse(Float64, w_a_forgetting_env),
        :posterior_damping_alpha => parse(Float64, posterior_damping_env),
        :w_mean_damping_alpha => isempty(w_mean_damping_env) ? nothing : parse(Float64, w_mean_damping_env),
        :w_a_damping_alpha => isempty(w_a_damping_env) ? nothing : parse(Float64, w_a_damping_env),
        :svi_likelihood_scaling => get(ENV, "SVI_LIKELIHOOD_SCALING", "false") in ("1", "true", "TRUE", "yes"),
        :likelihood_scale => parse(Float64, get(ENV, "LIKELIHOOD_SCALE", "1.0")),
        :likelihood_tempering_beta => parse(Float64, get(ENV, "LIKELIHOOD_TEMPERING_BETA", "1.0")),
        :candidate_prior => get(ENV, "CANDIDATE_PRIOR", "current"),
        :early_stop_window => parse(Int, get(ENV, "EARLY_STOP_WINDOW", "0")),
        :early_stop_rtol => parse(Float64, get(ENV, "EARLY_STOP_RTOL", "1e-4")),
        :early_stop_atol => parse(Float64, get(ENV, "EARLY_STOP_ATOL", "0.0")),
        :early_stop_min_steps => parse(Int, get(ENV, "EARLY_STOP_MIN_STEPS", "0")),
        :early_stop_patience => parse(Int, get(ENV, "EARLY_STOP_PATIENCE", "1")),
        :n_neurons => parse(Int, get(ENV, "N_NEURONS", "16")),
        :dataset => get(ENV, "XOR_DATASET", "test_dataset/xor_simple_dataset.csv"),
        :train_fraction => parse(Float64, get(ENV, "TRAIN_FRACTION", "0.3")),
        :prior_seed => parse(Int, get(ENV, "PRIOR_SEED", "42")),
        :prior_parameterization => get(ENV, "PRIOR_PARAMETERIZATION", "weighted"),
        :w_mean_prior_precision => parse(Float64, get(ENV, "W_MEAN_PRIOR_PRECISION", "1e-4")),
        :w_a_prior_precision => parse(Float64, get(ENV, "W_A_PRIOR_PRECISION", "1e-3")),
        :split_seed => parse(Int, get(ENV, "SPLIT_SEED", "2027")),
        :output_prefix => nothing,
        :shuffle_batches => true,
        :show_progress => true,
        :report_every => parse(Int, get(ENV, "REPORT_EVERY", "1")),
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
        elseif startswith(arg, "--obs-precision=")
            config[:obs_precision] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--prior-forgetting-rho=")
            config[:prior_forgetting_rho] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--w-mean-forgetting-rho=")
            config[:w_mean_forgetting_rho] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--w-a-forgetting-rho=")
            config[:w_a_forgetting_rho] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--posterior-damping-alpha=")
            config[:posterior_damping_alpha] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--w-mean-damping-alpha=")
            config[:w_mean_damping_alpha] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--w-a-damping-alpha=")
            config[:w_a_damping_alpha] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif arg == "--svi-likelihood-scaling"
            config[:svi_likelihood_scaling] = true
        elseif startswith(arg, "--likelihood-scale=")
            config[:likelihood_scale] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--likelihood-tempering-beta=")
            config[:likelihood_tempering_beta] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--candidate-prior=")
            config[:candidate_prior] = split(arg, "=", limit = 2)[2]
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
        elseif startswith(arg, "--prior-parameterization=")
            config[:prior_parameterization] = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--w-mean-prior-precision=")
            config[:w_mean_prior_precision] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--w-a-prior-precision=")
            config[:w_a_prior_precision] = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--split-seed=")
            config[:split_seed] = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--output-prefix=")
            config[:output_prefix] = split(arg, "=", limit = 2)[2]
        elseif arg == "--no-shuffle"
            config[:shuffle_batches] = false
        elseif arg == "--no-progress"
            config[:show_progress] = false
        elseif startswith(arg, "--report-every=")
            config[:report_every] = parse(Int, split(arg, "=", limit = 2)[2])
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

    if !(isfinite(config[:obs_precision]) && config[:obs_precision] > 0)
        error("--obs-precision must be finite and positive")
    end
    if !(isfinite(config[:prior_forgetting_rho]) && 0 < config[:prior_forgetting_rho] <= 1)
        error("--prior-forgetting-rho must be finite and in (0, 1]")
    end
    if isnothing(config[:w_mean_forgetting_rho])
        config[:w_mean_forgetting_rho] = config[:prior_forgetting_rho]
    end
    if isnothing(config[:w_a_forgetting_rho])
        config[:w_a_forgetting_rho] = config[:prior_forgetting_rho]
    end
    if !(isfinite(config[:w_mean_forgetting_rho]) && 0 < config[:w_mean_forgetting_rho] <= 1)
        error("--w-mean-forgetting-rho must be finite and in (0, 1]")
    end
    if !(isfinite(config[:w_a_forgetting_rho]) && 0 < config[:w_a_forgetting_rho] <= 1)
        error("--w-a-forgetting-rho must be finite and in (0, 1]")
    end
    if isnothing(config[:w_mean_damping_alpha])
        config[:w_mean_damping_alpha] = config[:posterior_damping_alpha]
    end
    if isnothing(config[:w_a_damping_alpha])
        config[:w_a_damping_alpha] = config[:posterior_damping_alpha]
    end
    if !(isfinite(config[:posterior_damping_alpha]) && 0 <= config[:posterior_damping_alpha] <= 1)
        error("--posterior-damping-alpha must be finite and in [0, 1]")
    end
    if !(isfinite(config[:w_mean_damping_alpha]) && 0 <= config[:w_mean_damping_alpha] <= 1)
        error("--w-mean-damping-alpha must be finite and in [0, 1]")
    end
    if !(isfinite(config[:w_a_damping_alpha]) && 0 <= config[:w_a_damping_alpha] <= 1)
        error("--w-a-damping-alpha must be finite and in [0, 1]")
    end
    if !(isfinite(config[:likelihood_scale]) && config[:likelihood_scale] > 0)
        error("--likelihood-scale must be finite and positive")
    end
    if !(isfinite(config[:likelihood_tempering_beta]) && config[:likelihood_tempering_beta] > 0)
        error("--likelihood-tempering-beta must be finite and positive")
    end
    if !(config[:candidate_prior] in ("current", "initial"))
        error("--candidate-prior must be current or initial")
    end
    if config[:report_every] < 0
        error("--report-every must be non-negative")
    end
    if !(config[:prior_parameterization] in ("weighted", "mean", "zero"))
        error("--prior-parameterization must be weighted, mean, or zero")
    end
    if !(isfinite(config[:w_mean_prior_precision]) && config[:w_mean_prior_precision] > 0)
        error("--w-mean-prior-precision must be finite and positive")
    end
    if !(isfinite(config[:w_a_prior_precision]) && config[:w_a_prior_precision] > 0)
        error("--w-a-prior-precision must be finite and positive")
    end

    if isnothing(config[:output_prefix])
        niter = config[:projection_niterations]
        prior_token = config[:prior_parameterization]
        if isnothing(config[:batch_size])
            obs_token = numeric_path_token(config[:obs_precision])
            config[:output_prefix] = "test_dataset/viz/relu_projection_niter_$(niter)_obs_$(obs_token)_prior_$(prior_token)"
        else
            batch_size = config[:batch_size]
            batch_iterations = config[:batch_iterations]
            sampling = config[:batch_sampling]
            obs_token = numeric_path_token(config[:obs_precision])
            rho_token = numeric_path_token(config[:prior_forgetting_rho])
            extra_tokens = String[]
            if config[:w_mean_damping_alpha] != 1.0 || config[:w_a_damping_alpha] != 1.0
                push!(extra_tokens, "amean_$(numeric_path_token(config[:w_mean_damping_alpha]))")
                push!(extra_tokens, "agate_$(numeric_path_token(config[:w_a_damping_alpha]))")
            end
            if config[:svi_likelihood_scaling]
                push!(extra_tokens, "sviscale")
            end
            if config[:likelihood_scale] != 1.0
                push!(extra_tokens, "lscale_$(numeric_path_token(config[:likelihood_scale]))")
            end
            if config[:likelihood_tempering_beta] != 1.0
                push!(extra_tokens, "beta_$(numeric_path_token(config[:likelihood_tempering_beta]))")
            end
            if config[:candidate_prior] != "current"
                push!(extra_tokens, "candidate_$(config[:candidate_prior])")
            end
            extra = isempty(extra_tokens) ? "" : "_" * join(extra_tokens, "_")
            config[:output_prefix] = "test_dataset/viz/relu_projection_niter_$(niter)_batch_$(batch_size)_biter_$(batch_iterations)_$(sampling)_obs_$(obs_token)_rho_$(rho_token)_prior_$(prior_token)$(extra)"
        end
    end

    return config
end

@model function xor_relu_direct_projection_niterations(n_neurons, features, y, priors, obs_precision)
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
        y[j] ~ NormalMeanPrecision(out[j], obs_precision)
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

function make_weight_prior(rng, n_features, precision, parameterization)
    Λ = Diagonal(fill(precision, n_features))
    if parameterization == "weighted"
        return MvNormalWeightedMeanPrecision(randn(rng, n_features), Λ)
    elseif parameterization == "mean"
        μ = randn(rng, n_features)
        return MvNormalWeightedMeanPrecision(Λ * μ, Λ)
    elseif parameterization == "zero"
        return MvNormalWeightedMeanPrecision(zeros(n_features), Λ)
    end

    error("Unknown prior parameterization: $parameterization")
end

function make_priors(;
    n_neurons = 4,
    n_features::Int = 3,
    seed::Int = 42,
    parameterization = "weighted",
    w_mean_prior_precision = 1e-4,
    w_a_prior_precision = 1e-3,
)
    rng = StableRNG(seed)

    w_mean = [
        make_weight_prior(rng, n_features, w_mean_prior_precision, parameterization)
        for _ in 1:n_neurons
    ]
    w_a = [
        make_weight_prior(rng, n_features, w_a_prior_precision, parameterization)
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

function precision_diagonal_values(dist)
    Λ = invcov(dist)
    if Λ isa Diagonal
        return Float64.(Λ.diag)
    end
    return Float64.(diag(Matrix(Λ)))
end

function posterior_variance_values(dist)
    return Float64.(var(dist))
end

function weight_uncertainty_metrics(w_mean_dists, w_a_dists)
    w_mean_vars = Float64[]
    w_a_vars = Float64[]
    w_mean_precs = Float64[]
    w_a_precs = Float64[]

    for dist in w_mean_dists
        append!(w_mean_vars, posterior_variance_values(dist))
        append!(w_mean_precs, precision_diagonal_values(dist))
    end
    for dist in w_a_dists
        append!(w_a_vars, posterior_variance_values(dist))
        append!(w_a_precs, precision_diagonal_values(dist))
    end

    return (
        min_var_w_mean = minimum(w_mean_vars),
        median_var_w_mean = median(w_mean_vars),
        max_var_w_mean = maximum(w_mean_vars),
        min_var_w_a = minimum(w_a_vars),
        median_var_w_a = median(w_a_vars),
        max_var_w_a = maximum(w_a_vars),
        max_precision_w_mean = maximum(w_mean_precs),
        max_precision_w_a = maximum(w_a_precs),
    )
end

function effective_gate_count(gate_values)
    total = sum(gate_values)
    if total <= eps(Float64)
        return 0.0
    end
    squared_sum = sum(abs2, gate_values)
    return squared_sum <= eps(Float64) ? 0.0 : total^2 / squared_sum
end

function weight_geometry_metrics(w_means, w_as, features)
    w_mean_slope_norms = [norm(w[2:end]) for w in w_means]
    w_a_slope_norms = [norm(w[2:end]) for w in w_as]
    w_a_abs_biases = [abs(w[1]) for w in w_as]
    gate_values = Float64[]
    effective_counts = Float64[]

    for f in features
        gates = [max(0.0, dot(w_a, f)) for w_a in w_as]
        append!(gate_values, gates)
        push!(effective_counts, effective_gate_count(gates))
    end

    active_fraction = isempty(gate_values) ? NaN : count(>(0.0), gate_values) / length(gate_values)
    mean_effective_count = isempty(effective_counts) ? NaN : mean(effective_counts)
    mean_w_a_slope = mean(w_a_slope_norms)

    return (
        mean_w_mean_slope_norm = mean(w_mean_slope_norms),
        max_w_mean_slope_norm = maximum(w_mean_slope_norms),
        mean_w_a_slope_norm = mean_w_a_slope,
        max_w_a_slope_norm = maximum(w_a_slope_norms),
        mean_abs_w_a_bias = mean(w_a_abs_biases),
        w_a_bias_slope_ratio = mean(w_a_abs_biases) / max(mean_w_a_slope, eps(Float64)),
        gate_active_fraction = active_fraction,
        mean_effective_gate_count = mean_effective_count,
    )
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
        min_var_w_mean = Float64[],
        median_var_w_mean = Float64[],
        max_var_w_mean = Float64[],
        min_var_w_a = Float64[],
        median_var_w_a = Float64[],
        max_var_w_a = Float64[],
        max_precision_w_mean = Float64[],
        max_precision_w_a = Float64[],
        mean_w_mean_slope_norm = Float64[],
        max_w_mean_slope_norm = Float64[],
        mean_w_a_slope_norm = Float64[],
        max_w_a_slope_norm = Float64[],
        mean_abs_w_a_bias = Float64[],
        w_a_bias_slope_ratio = Float64[],
        gate_active_fraction = Float64[],
        mean_effective_gate_count = Float64[],
    )

    for iter in 1:n_iters
        w_mean_dists = result.posteriors[:w_mean][iter]
        w_a_dists = result.posteriors[:w_a][iter]
        w_means = [mean(w_mean_dists[k]) for k in 1:n_neurons]
        w_as = [mean(w_a_dists[k]) for k in 1:n_neurons]

        y_pred = predict_features(features_test, w_means, w_as)
        mse, invalid = finite_mse(y_pred, y_test)
        masses = gate_masses(features_test, w_as)
        fe = iter <= length(free_energy) ? Float64(free_energy[iter]) : NaN
        uncertainty = weight_uncertainty_metrics(w_mean_dists, w_a_dists)
        geometry = weight_geometry_metrics(w_means, w_as, features_test)

        push!(rows, merge((
            iteration = iter,
            free_energy = fe,
            test_mse = mse,
            invalid_predictions = invalid,
            mean_gate_mass = mean(masses),
            min_gate_mass = minimum(masses),
        ), uncertainty, geometry))
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

function finite_mean(values)
    finite_values = Float64[]
    for value in values
        x = Float64(value)
        isfinite(x) && push!(finite_values, x)
    end
    return isempty(finite_values) ? NaN : mean(finite_values)
end

function save_online_free_energy_plots(metrics, config)
    output_prefix = config[:output_prefix]
    mkpath(dirname(output_prefix))

    finite_update_indices = findall(isfinite, metrics.free_energy)
    if isempty(finite_update_indices)
        println("Skipped free-energy plots: no finite FE values were recorded.")
        return nothing
    end

    fe_updates_path = "$(output_prefix)_fe_updates.png"
    plot(
        metrics.update[finite_update_indices],
        metrics.free_energy[finite_update_indices],
        xlabel = "Online update",
        ylabel = "Free energy",
        title = "Free energy by online update",
        legend = false,
        linewidth = 1.5,
        marker = :circle,
        markersize = 2,
    )
    savefig(fe_updates_path)
    println("Free-energy update plot saved to $fe_updates_path")

    epochs = sort(unique(metrics.epoch))
    epoch_mean_fe = Float64[]
    for epoch in epochs
        push!(epoch_mean_fe, finite_mean(metrics.free_energy[metrics.epoch .== epoch]))
    end

    finite_epoch_indices = findall(isfinite, epoch_mean_fe)
    if isempty(finite_epoch_indices)
        println("Skipped epoch-average free-energy plot: no finite epoch averages were recorded.")
        return nothing
    end

    fe_epoch_path = "$(output_prefix)_fe_epoch_mean.png"
    plot(
        epochs[finite_epoch_indices],
        epoch_mean_fe[finite_epoch_indices],
        xlabel = "Epoch",
        ylabel = "Mean free energy",
        title = "Mean free energy by epoch",
        legend = false,
        linewidth = 2,
        marker = :circle,
        markersize = 3,
    )
    savefig(fe_epoch_path)
    println("Epoch-average free-energy plot saved to $fe_epoch_path")

    return nothing
end

function save_online_diagnostic_plots(metrics, config)
    output_prefix = config[:output_prefix]
    mkpath(dirname(output_prefix))

    if nrow(metrics) == 0
        println("Skipped online diagnostic plots: no metrics were recorded.")
        return nothing
    end

    mse_path = "$(output_prefix)_mse_updates.png"
    plot(
        metrics.update,
        metrics.test_mse,
        xlabel = "Online update",
        ylabel = "Test MSE",
        title = "Test MSE by online update",
        legend = false,
        linewidth = 1.5,
        marker = :circle,
        markersize = 2,
    )
    savefig(mse_path)
    println("Test-MSE update plot saved to $mse_path")

    variance_path = "$(output_prefix)_variance_updates.png"
    plot(
        metrics.update,
        metrics.min_var_w_mean,
        xlabel = "Online update",
        ylabel = "Marginal variance",
        label = "min var w_mean",
        yscale = :log10,
        linewidth = 1.5,
    )
    plot!(
        metrics.update,
        metrics.min_var_w_a,
        label = "min var w_a",
        linewidth = 1.5,
    )
    plot!(
        metrics.update,
        metrics.median_var_w_a,
        label = "median var w_a",
        linewidth = 1.2,
        linestyle = :dash,
    )
    savefig(variance_path)
    println("Posterior-variance update plot saved to $variance_path")

    gate_path = "$(output_prefix)_gate_geometry_updates.png"
    plot(
        metrics.update,
        metrics.mean_w_a_slope_norm,
        xlabel = "Online update",
        ylabel = "Gate geometry",
        label = "mean slope norm w_a",
        linewidth = 1.5,
    )
    plot!(
        metrics.update,
        metrics.mean_abs_w_a_bias,
        label = "mean |bias| w_a",
        linewidth = 1.5,
    )
    plot!(
        metrics.update,
        metrics.mean_effective_gate_count,
        label = "mean effective gates",
        linewidth = 1.2,
        linestyle = :dash,
    )
    savefig(gate_path)
    println("Gate-geometry update plot saved to $gate_path")

    collapse_path = "$(output_prefix)_collapse_diagnostic.png"
    p1 = plot(
        metrics.update,
        metrics.test_mse,
        xlabel = "Online update",
        ylabel = "Test MSE",
        title = "Prediction quality",
        legend = false,
        linewidth = 1.5,
    )
    p2 = plot(
        metrics.update,
        metrics.min_var_w_a,
        xlabel = "Online update",
        ylabel = "min var w_a",
        title = "Gate posterior uncertainty",
        legend = false,
        yscale = :log10,
        linewidth = 1.5,
    )
    p3 = plot(
        metrics.update,
        metrics.mean_w_a_slope_norm,
        xlabel = "Online update",
        ylabel = "mean slope norm",
        title = "Gate nonlinearity",
        legend = false,
        linewidth = 1.5,
    )
    plot(p1, p2, p3, layout = (3, 1), size = (900, 900))
    savefig(collapse_path)
    println("Collapse diagnostic plot saved to $collapse_path")

    epochs = sort(unique(metrics.epoch))
    epoch_min_var_w_a = Float64[]
    epoch_median_var_w_a = Float64[]
    epoch_mse = Float64[]
    for epoch in epochs
        epoch_rows = metrics[metrics.epoch .== epoch, :]
        push!(epoch_min_var_w_a, finite_mean(epoch_rows.min_var_w_a))
        push!(epoch_median_var_w_a, finite_mean(epoch_rows.median_var_w_a))
        push!(epoch_mse, finite_mean(epoch_rows.test_mse))
    end

    epoch_path = "$(output_prefix)_epoch_mean_diagnostics.png"
    p_epoch_1 = plot(
        epochs,
        epoch_mse,
        xlabel = "Epoch",
        ylabel = "Mean test MSE",
        title = "Epoch mean test MSE",
        legend = false,
        linewidth = 2,
        marker = :circle,
        markersize = 3,
    )
    p_epoch_2 = plot(
        epochs,
        epoch_min_var_w_a,
        xlabel = "Epoch",
        ylabel = "Mean variance",
        title = "Epoch mean gate variance",
        label = "min var w_a",
        yscale = :log10,
        linewidth = 2,
        marker = :circle,
        markersize = 3,
    )
    plot!(
        p_epoch_2,
        epochs,
        epoch_median_var_w_a,
        label = "median var w_a",
        linewidth = 2,
        marker = :circle,
        markersize = 3,
    )
    plot(p_epoch_1, p_epoch_2, layout = (2, 1), size = (900, 650))
    savefig(epoch_path)
    println("Epoch-mean diagnostic plot saved to $epoch_path")

    return nothing
end

function forget_gaussian_prior(dist, rho)
    if rho == 1.0
        return deepcopy(dist)
    end

    xi = rho .* Vector{Float64}(weightedmean(dist))
    Λ = invcov(dist)
    if Λ isa Diagonal
        return MvNormalWeightedMeanPrecision(xi, Diagonal(rho .* Vector{Float64}(Λ.diag)))
    end
    return MvNormalWeightedMeanPrecision(xi, rho .* Matrix{Float64}(Λ))
end

function apply_prior_forgetting(priors, w_mean_rho, w_a_rho)
    if w_mean_rho == 1.0 && w_a_rho == 1.0
        return deepcopy(priors)
    end

    return Dict{Symbol,Any}(
        :w_mean => [forget_gaussian_prior(prior, w_mean_rho) for prior in priors[:w_mean]],
        :w_a => [forget_gaussian_prior(prior, w_a_rho) for prior in priors[:w_a]],
        :tau => deepcopy(priors[:tau]),
    )
end

function posterior_priors_from_result(result)
    return Dict{Symbol,Any}(
        :w_mean => deepcopy(result.posteriors[:w_mean][end]),
        :w_a => deepcopy(result.posteriors[:w_a][end]),
        :tau => deepcopy(result.posteriors[:tau][end]),
    )
end

function precision_as_matrix(Λ)
    return Λ isa Diagonal ? Matrix{Float64}(Λ) : Matrix{Float64}(Λ)
end

function damp_gaussian_prior(old_dist, post_dist, alpha)
    if alpha == 1.0
        return deepcopy(post_dist)
    elseif alpha == 0.0
        return deepcopy(old_dist)
    end

    xi_old = Vector{Float64}(weightedmean(old_dist))
    xi_post = Vector{Float64}(weightedmean(post_dist))
    xi = (1 - alpha) .* xi_old .+ alpha .* xi_post

    Λ_old = invcov(old_dist)
    Λ_post = invcov(post_dist)
    if Λ_old isa Diagonal && Λ_post isa Diagonal
        diag_old = Vector{Float64}(Λ_old.diag)
        diag_post = Vector{Float64}(Λ_post.diag)
        return MvNormalWeightedMeanPrecision(
            xi,
            Diagonal((1 - alpha) .* diag_old .+ alpha .* diag_post),
        )
    end

    Λ = (1 - alpha) .* precision_as_matrix(Λ_old) .+ alpha .* precision_as_matrix(Λ_post)
    return MvNormalWeightedMeanPrecision(xi, Λ)
end

function damp_posterior_priors(old_priors, posterior_priors, w_mean_alpha, w_a_alpha)
    return Dict{Symbol,Any}(
        :w_mean => [
            damp_gaussian_prior(old_priors[:w_mean][k], posterior_priors[:w_mean][k], w_mean_alpha)
            for k in eachindex(old_priors[:w_mean])
        ],
        :w_a => [
            damp_gaussian_prior(old_priors[:w_a][k], posterior_priors[:w_a][k], w_a_alpha)
            for k in eachindex(old_priors[:w_a])
        ],
        :tau => deepcopy(posterior_priors[:tau]),
    )
end

function update_priors_from_result(result, w_mean_rho = 1.0, w_a_rho = w_mean_rho)
    posterior_priors = posterior_priors_from_result(result)

    return apply_prior_forgetting(posterior_priors, w_mean_rho, w_a_rho)
end

function update_priors_from_result(
    result,
    old_priors,
    w_mean_rho,
    w_a_rho,
    w_mean_alpha,
    w_a_alpha,
)
    posterior_priors = posterior_priors_from_result(result)
    damped_priors = damp_posterior_priors(old_priors, posterior_priors, w_mean_alpha, w_a_alpha)

    return apply_prior_forgetting(damped_priors, w_mean_rho, w_a_rho)
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
    posterior_priors,
)
    y_pred = predict_features(features_test, w_means, w_as)
    mse, invalid = finite_mse(y_pred, y_test)
    masses = gate_masses(features_test, w_as)
    uncertainty = weight_uncertainty_metrics(posterior_priors[:w_mean], posterior_priors[:w_a])
    geometry = weight_geometry_metrics(w_means, w_as, features_test)

    push!(metrics, merge((
        update = update,
        epoch = epoch,
        batch = batch,
        batch_size = batch_size,
        free_energy = free_energy,
        test_mse = mse,
        invalid_predictions = invalid,
        mean_gate_mass = mean(masses),
        min_gate_mass = minimum(masses),
    ), uncertainty, geometry))
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

function online_likelihood_scale(config, n_train, batch_size)
    scale = config[:likelihood_scale] * config[:likelihood_tempering_beta]
    if config[:svi_likelihood_scaling]
        scale *= n_train / batch_size
    end

    return scale
end

function online_candidate_priors(config, current_priors, initial_priors)
    if config[:candidate_prior] == "current"
        return current_priors
    elseif config[:candidate_prior] == "initial"
        return initial_priors
    end

    error("Unknown candidate prior mode: $(config[:candidate_prior])")
end

function infer_callbacks(early_stopper)
    return isnothing(early_stopper) ? nothing : (after_iteration = early_stopper,)
end

function report_online_update!(progress, row, completed_iterations, total_updates, report_every)
    if isnothing(progress)
        if report_every == 0 || (row.update != 1 && row.update != total_updates && row.update % report_every != 0)
            return nothing
        end

        @printf(
            "update=%03d/%03d epoch=%d batch=%d n=%d inner=%d fe=%.4f test_mse=%.6f invalid=%d min_var_w_a=%.3g mean_w_a_slope=%.4g\n",
            row.update,
            total_updates,
            row.epoch,
            row.batch,
            row.batch_size,
            completed_iterations,
            row.free_energy,
            row.test_mse,
            row.invalid_predictions,
            row.min_var_w_a,
            row.mean_w_a_slope_norm,
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
        (:min_var_w_a, row.min_var_w_a),
        (:mean_w_a_slope, row.mean_w_a_slope_norm),
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
    obs_precision = config[:obs_precision]
    w_mean_forgetting_rho = config[:w_mean_forgetting_rho]
    w_a_forgetting_rho = config[:w_a_forgetting_rho]
    w_mean_damping_alpha = config[:w_mean_damping_alpha]
    w_a_damping_alpha = config[:w_a_damping_alpha]
    report_every = config[:report_every]
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
        min_var_w_mean = Float64[],
        median_var_w_mean = Float64[],
        max_var_w_mean = Float64[],
        min_var_w_a = Float64[],
        median_var_w_a = Float64[],
        max_var_w_a = Float64[],
        max_precision_w_mean = Float64[],
        max_precision_w_a = Float64[],
        mean_w_mean_slope_norm = Float64[],
        max_w_mean_slope_norm = Float64[],
        mean_w_a_slope_norm = Float64[],
        max_w_a_slope_norm = Float64[],
        mean_abs_w_a_bias = Float64[],
        w_a_bias_slope_ratio = Float64[],
        gate_active_fraction = Float64[],
        mean_effective_gate_count = Float64[],
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
    likelihood_scale = online_likelihood_scale(config, n_train, batch_size)
    candidate_obs_precision = obs_precision * likelihood_scale
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
            candidate_priors = online_candidate_priors(config, current_priors, initial_priors)

            result = infer(
                model = xor_relu_direct_projection_niterations(
                    n_neurons = n_neurons,
                    priors = candidate_priors,
                    obs_precision = candidate_obs_precision,
                ),
                data = (y = batch_y, features = batch_features),
                constraints = xor_relu_direct_projection_niterations_constraints(projection_niterations),
                initialization = xor_relu_direct_projection_niterations_init(candidate_priors),
                iterations = batch_iterations,
                free_energy = true,
                showprogress = false,
                callbacks = callbacks,
                options = (limit_stack_depth = 100,),
            )

            current_priors = update_priors_from_result(
                result,
                current_priors,
                w_mean_forgetting_rho,
                w_a_forgetting_rho,
                w_mean_damping_alpha,
                w_a_damping_alpha,
            )
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
                current_priors,
            )

            row = metrics[end, :]
            report_online_update!(progress, row, completed_iterations, n_updates, report_every)
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
            candidate_priors = online_candidate_priors(config, current_priors, initial_priors)

            result = infer(
                model = xor_relu_direct_projection_niterations(
                    n_neurons = n_neurons,
                    priors = candidate_priors,
                    obs_precision = candidate_obs_precision,
                ),
                data = (y = batch_y, features = batch_features),
                constraints = xor_relu_direct_projection_niterations_constraints(projection_niterations),
                initialization = xor_relu_direct_projection_niterations_init(candidate_priors),
                iterations = batch_iterations,
                free_energy = true,
                showprogress = false,
                callbacks = callbacks,
                options = (limit_stack_depth = 100,),
            )

            current_priors = update_priors_from_result(
                result,
                current_priors,
                w_mean_forgetting_rho,
                w_a_forgetting_rho,
                w_mean_damping_alpha,
                w_a_damping_alpha,
            )
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
                current_priors,
            )

            row = metrics[end, :]
            report_online_update!(progress, row, completed_iterations, n_updates, report_every)
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
        println("  obs_precision=$(config[:obs_precision]), prior_forgetting_rho=$(config[:prior_forgetting_rho])")
        println("  w_mean_forgetting_rho=$(config[:w_mean_forgetting_rho]), w_a_forgetting_rho=$(config[:w_a_forgetting_rho])")
        candidate_scale = online_likelihood_scale(config, nrow(df_train), config[:batch_size])
        println("  w_mean_damping_alpha=$(config[:w_mean_damping_alpha]), w_a_damping_alpha=$(config[:w_a_damping_alpha])")
        println("  candidate_prior=$(config[:candidate_prior]), likelihood_scale=$candidate_scale, candidate_obs_precision=$(config[:obs_precision] * candidate_scale)")
    else
        println("XOR ReLU Direct: full-batch VMP")
        println("  projection_niterations=$projection_niterations")
        println("  neurons=$n_neurons, outer_iterations=$outer_iterations")
        println("  obs_precision=$(config[:obs_precision])")
    end
    if early_stopping_enabled(config)
        println("  early_stop_window=$(config[:early_stop_window]), early_stop_rtol=$(config[:early_stop_rtol]), early_stop_atol=$(config[:early_stop_atol])")
        println("  early_stop_min_steps=$(config[:early_stop_min_steps]), early_stop_patience=$(config[:early_stop_patience])")
    end
    println("  prior_parameterization=$(config[:prior_parameterization]), w_mean_prior_precision=$(config[:w_mean_prior_precision]), w_a_prior_precision=$(config[:w_a_prior_precision])")
    println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
    println("  dataset=$(config[:dataset])")
    println("=" ^ 78)

    priors = make_priors(
        n_neurons = n_neurons,
        seed = config[:prior_seed],
        parameterization = config[:prior_parameterization],
        w_mean_prior_precision = config[:w_mean_prior_precision],
        w_a_prior_precision = config[:w_a_prior_precision],
    )
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
            save_online_free_energy_plots(metrics, config)
            save_online_diagnostic_plots(metrics, config)
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
            obs_precision = config[:obs_precision],
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

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
