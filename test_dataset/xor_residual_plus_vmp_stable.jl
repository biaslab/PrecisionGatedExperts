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
const OUTPUT_PREFIX = get(ENV, "OUTPUT_PREFIX", "test_dataset/viz/xor_residual_plus_vmp_stable")
const N_NEURONS = parse(Int, get(ENV, "N_NEURONS", "16"))
const N_ITERATIONS = parse(Int, get(ENV, "N_ITERATIONS", "10"))
const TRAIN_FRACTION = parse(Float64, get(ENV, "TRAIN_FRACTION", "0.3"))

const TAU_SKIP = parse(Float64, get(ENV, "TAU_SKIP", "1.0"))
const RESIDUAL_SCALE = parse(Float64, get(ENV, "RESIDUAL_SCALE", "0.25"))

const W_BASE_PRECISION = parse(Float64, get(ENV, "W_BASE_PRECISION", "0.1"))
const W_DELTA_PRECISION = parse(Float64, get(ENV, "W_DELTA_PRECISION", "0.1"))
const W_GATE_PRECISION = parse(Float64, get(ENV, "W_GATE_PRECISION", "1.0"))
const WEIGHT_PRIOR_MEAN_SCALE = parse(Float64, get(ENV, "WEIGHT_PRIOR_MEAN_SCALE", "1.0"))

const TAU_BASE_SHAPE = parse(Float64, get(ENV, "TAU_BASE_SHAPE", "1e6"))
const TAU_BASE_RATE = parse(Float64, get(ENV, "TAU_BASE_RATE", "1.0"))
const TAU_DELTA_SHAPE = parse(Float64, get(ENV, "TAU_DELTA_SHAPE", "1e6"))
const TAU_DELTA_RATE = parse(Float64, get(ENV, "TAU_DELTA_RATE", "1.0"))
const TAU_GATE_SHAPE = parse(Float64, get(ENV, "TAU_GATE_SHAPE", "1e8"))
const TAU_GATE_RATE = parse(Float64, get(ENV, "TAU_GATE_RATE", "100.0"))
const OBS_TAU_SHAPE = parse(Float64, get(ENV, "OBS_TAU_SHAPE", "1e8"))
const OBS_TAU_RATE = parse(Float64, get(ENV, "OBS_TAU_RATE", "1e6"))

const PRIOR_SEED = parse(Int, get(ENV, "PRIOR_SEED", "2027"))
const SPLIT_SEED = parse(Int, get(ENV, "SPLIT_SEED", "2027"))
const FREE_ENERGY = lowercase(get(ENV, "FREE_ENERGY", "true")) in ("1", "true", "yes")

@model function xor_residual_plus_vmp_stable(n_neurons, features, residual_features, y, priors, tau_skip)
    local w_base, w_delta, w_gate, base, delta, mu, za, gamma
    local tau_base, tau_delta, tau_gate, obs_tau, out

    tau_base ~ priors[:tau_base]
    tau_delta ~ priors[:tau_delta]
    tau_gate ~ priors[:tau_gate]
    obs_tau ~ priors[:obs_tau]
    w_base ~ priors[:w_base]

    for k = 1:n_neurons
        w_delta[k] ~ priors[:w_delta][k]
        w_gate[k] ~ priors[:w_gate][k]
    end

    for j = 1:length(y)
        base[j] ~ softdot(features[j], w_base, tau_base)
        out[j] ~ NormalMeanPrecision(base[j], tau_skip)

        for k = 1:n_neurons
            delta[k, j] ~ softdot(residual_features[j], w_delta[k], tau_delta)
            mu[k, j] := base[j] + delta[k, j]

            za[k, j] ~ softdot(features[j], w_gate[k], tau_gate) where {meta = LowRankMeta()}
            gamma[k, j] ~ ReLU(za[k, j])
            out[j] ~ NormalMeanPrecision(mu[k, j], gamma[k, j])
        end

        y[j] ~ NormalMeanPrecision(out[j], obs_tau)
    end
end

@constraints function xor_residual_plus_vmp_stable_constraints()
    q(w_base, w_delta, w_gate, base, delta, mu, za, gamma, tau_base, tau_delta, tau_gate, obs_tau, out) =
        q(w_base, base, w_delta, delta, mu, out)q(w_gate)q(za, gamma)q(tau_base)q(tau_delta)q(tau_gate)q(obs_tau)

    q(base)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(delta)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(mu)::ProjectedTo(NormalMeanVariance, parameters = ProjectionParameters(strategy = ClosedFormStrategy()))
    q(za)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
    q(gamma)::ProjectedTo(Gamma, parameters = ProjectionParameters(strategy = ClosedFormStrategy(EnzymeBackend())))
end

@initialization function xor_residual_plus_vmp_stable_init(priors)
    q(w_base) = priors[:w_base]
    q(w_delta) = deepcopy(priors[:w_delta])
    q(w_gate) = deepcopy(priors[:w_gate])
    q(base) = NormalMeanVariance(0.5, 1.0)
    q(delta) = NormalMeanVariance(0.0, 1.0)
    q(mu) = NormalMeanVariance(0.5, 1.0)
    q(za) = GammaShapeScale(2.0, 1.0)
    q(gamma) = GammaShapeScale(2.0, 1.0)
    q(tau_base) = priors[:tau_base]
    q(tau_delta) = priors[:tau_delta]
    q(tau_gate) = priors[:tau_gate]
    q(obs_tau) = priors[:obs_tau]
    q(out) = NormalMeanVariance(0.5, 1.0)

    μ(base) = NormalMeanVariance(0.5, 10.0)
    μ(delta) = NormalMeanVariance(0.0, 10.0)
    μ(mu) = NormalMeanVariance(0.5, 10.0)
end

function make_weight_prior(rng, n_features, precision)
    precision_matrix = Diagonal(fill(precision, n_features))
    actual_mean = WEIGHT_PRIOR_MEAN_SCALE .* randn(rng, n_features)
    return MvNormalWeightedMeanPrecision(precision_matrix * actual_mean, precision_matrix)
end

function make_priors(; n_neurons = N_NEURONS, n_features = 3, seed = PRIOR_SEED)
    rng = StableRNG(seed)
    w_base = make_weight_prior(rng, n_features, W_BASE_PRECISION)

    # Keep RNG consumption aligned with the earlier residual/direct comparisons.
    _unused_w_mean = [make_weight_prior(rng, n_features, 1e-4) for _ in 1:n_neurons]

    return Dict{Symbol,Any}(
        :w_base => w_base,
        :w_delta => [make_weight_prior(rng, n_features, W_DELTA_PRECISION) for _ in 1:n_neurons],
        :w_gate => [make_weight_prior(rng, n_features, W_GATE_PRECISION) for _ in 1:n_neurons],
        :tau_base => GammaShapeRate(TAU_BASE_SHAPE, TAU_BASE_RATE),
        :tau_delta => GammaShapeRate(TAU_DELTA_SHAPE, TAU_DELTA_RATE),
        :tau_gate => GammaShapeRate(TAU_GATE_SHAPE, TAU_GATE_RATE),
        :obs_tau => GammaShapeRate(OBS_TAU_SHAPE, OBS_TAU_RATE),
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

function scalar_posterior_mean(result, name::Symbol, iteration)
    haskey(result.posteriors, name) || return NaN
    return Float64(mean(result.posteriors[name][iteration]))
end

function trace_for(result, features_test, y_test)
    rows = DataFrame(
        iteration = Int[],
        test_mse = Float64[],
        free_energy = Float64[],
        obs_tau = Float64[],
        tau_base = Float64[],
        tau_delta = Float64[],
        tau_gate = Float64[],
        w_delta_max_norm = Float64[],
        w_gate_max_norm = Float64[],
    )
    free_energy = FREE_ENERGY ? result.free_energy : Real[]

    for iteration in 1:length(result.posteriors[:w_gate])
        predict_fn = predictor_at(result, iteration)
        w_deltas = [mean(result.posteriors[:w_delta][iteration][k]) for k in 1:N_NEURONS]
        w_gates = [mean(result.posteriors[:w_gate][iteration][k]) for k in 1:N_NEURONS]
        fe = iteration <= length(free_energy) ? Float64(free_energy[iteration]) : NaN
        push!(
            rows,
            (
                iteration,
                mse_for(predict_fn, features_test, y_test),
                fe,
                scalar_posterior_mean(result, :obs_tau, iteration),
                scalar_posterior_mean(result, :tau_base, iteration),
                scalar_posterior_mean(result, :tau_delta, iteration),
                scalar_posterior_mean(result, :tau_gate, iteration),
                maximum(norm.(w_deltas)),
                maximum(norm.(w_gates)),
            ),
        )
    end

    return rows
end

function save_trace_plot(trace)
    outpath = "$(OUTPUT_PREFIX)_mse_by_iteration.png"
    p = plot(
        trace.iteration,
        trace.test_mse;
        marker = :circle,
        xlabel = "iteration",
        ylabel = "test MSE",
        title = "XOR residual plus VMP-stable",
        label = "test",
        size = (760, 440),
    )
    savefig(p, outpath)
    return outpath
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
    plot(p1, p2; layout = (1, 2), size = (1000, 420), plot_title = "residual_plus_vmp_stable")
    savefig(outpath)
    return outpath
end

# function main()
df = CSV.read(DATASET, DataFrame)
df_train, df_test = split_train_test(df)
features_train = build_features(df_train)
residual_features_train = build_residual_features(features_train)
features_test = build_features(df_test)
priors = make_priors()

println("=" ^ 78)
println("XOR residual_plus VMP-stable")
println("  n_train=$(nrow(df_train)), n_test=$(nrow(df_test))")
println("  neurons=$N_NEURONS, iterations=$N_ITERATIONS")
println("  tau_skip=$TAU_SKIP, residual_scale=$RESIDUAL_SCALE")
println("  weight precisions: base=$W_BASE_PRECISION, delta=$W_DELTA_PRECISION, gate=$W_GATE_PRECISION")
println("  tau_gate=GammaShapeRate($TAU_GATE_SHAPE, $TAU_GATE_RATE)")
println("  obs_tau=GammaShapeRate($OBS_TAU_SHAPE, $OBS_TAU_RATE)")
println("=" ^ 78)

result = infer(
    model = xor_residual_plus_vmp_stable(n_neurons = N_NEURONS, priors = priors, tau_skip = TAU_SKIP),
    data = (y = df_train.OT, features = features_train, residual_features = residual_features_train),
    constraints = xor_residual_plus_vmp_stable_constraints(),
    initialization = xor_residual_plus_vmp_stable_init(priors),
    iterations = N_ITERATIONS,
    free_energy = FREE_ENERGY,
    showprogress = true,
    options = (limit_stack_depth = 100,),
)

trace = trace_for(result, features_test, df_test.OT)
baseline_mse = mean((0.52 .- df_test.OT) .^ 2)
best_idx = argmin(trace.test_mse)
final_predictor = predictor_at(result, nrow(trace))
best_predictor = predictor_at(result, trace.iteration[best_idx])

mkpath(dirname(OUTPUT_PREFIX))
trace_path = "$(OUTPUT_PREFIX)_trace.csv"
CSV.write(trace_path, trace)
plot_path = save_trace_plot(trace)
final_heatmap = save_heatmap("final", final_predictor, features_test, df_test)
best_heatmap = save_heatmap("best_iter_$(trace.iteration[best_idx])", best_predictor, features_test, df_test)

@printf("Baseline MSE constant 0.52 = %.6f\n", baseline_mse)
@printf("Final test MSE = %.6f at iteration %d\n", trace.test_mse[end], trace.iteration[end])
@printf("Best diagnostic test MSE = %.6f at iteration %d\n", trace.test_mse[best_idx], trace.iteration[best_idx])
@printf("Final obs_tau = %.6f\n", trace.obs_tau[end])
@printf("Final tau_gate = %.6f\n", trace.tau_gate[end])
@printf("Final w_delta max norm = %.6f\n", trace.w_delta_max_norm[end])
@printf("Final w_gate max norm = %.6f\n", trace.w_gate_max_norm[end])
println("Trace saved to $trace_path")
println("MSE plot saved to $plot_path")
println("Final heatmap saved to $final_heatmap")
println("Best heatmap saved to $best_heatmap")

plot(1:length(result.free_energy), result.free_energy)
# end

# main()
