#!/usr/bin/env julia

"""
Adaptive mixture of local experts (Jacobs,Jordan,Nowlan,Hinton)

Usage:
    julia scripts/adaptive_mixture_local_experts.jl
"""

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using ProgressMeter
using ADTypes
using Optimisers
using Statistics
using LinearAlgebra
using Lux
using Reactant
using Plots
using Random
using ProbabilisticEnsembling



# -----------------------------------------------------------------------------
# Adaptive Mixture of Local Experts (Jacobs et al., 1991)
# -----------------------------------------------------------------------------

function stable_softmax(logits)
    z = logits .- maximum(logits)
    p = exp.(z)
    return p ./ sum(p)
end

function gating_probs(gating, ps, st, x)
    logits, _ = gating(Float32.(x), ps, st)
    return stable_softmax(vec(logits))
end

function moe_predict(predictions_vec, gating, ps, st, x)
    probs = gating_probs(gating, ps, st, x)
    preds = hcat(predictions_vec...)
    return preds * probs
end

function moe_var(predictions_vec, gating, ps, st, x)
    probs = gating_probs(gating, ps, st, x)
    preds = hcat(predictions_vec...)
    return (preds .- preds * probs).^2 * probs
end

function moe_objective(gating, ps, st, data)
    preds, x, y = data
    probs = gating_probs(gating, ps, st, x)
    losses = [sum(abs2, pred .- y) for pred in preds]
    return dot(probs, losses), st, (;)
end

function average_moe_loss(predictions_vec, features, y, gating, ps, st)
    st_eval = Lux.testmode(st)
    total = 0.0f0
    for j in eachindex(features)
        probs = gating_probs(gating, ps, st_eval, features[j])
        losses = [sum(abs2, pred .- y[j]) for pred in predictions_vec[:, j]]
        total += dot(probs, losses)
    end
    return total / length(features)
end

function train_moe!(
    predictions_train_vec, features_train, y_train,
    predictions_monitor_vec, features_monitor, y_monitor,
    gating, opt;
    n_epochs=100, patience=50, min_delta=1f-6, monitor_label="val"
)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, gating)
    train_state = Lux.Training.TrainState(gating, ps, st, opt)
    ad = AutoEnzyme()

    y_train_f32 = [Float32.(y) for y in y_train]
    features_train_f32 = [Float32.(x) for x in features_train]
    predictions_train_f32 = [Float32.(predictions_train_vec[i, j]) for i in axes(predictions_train_vec, 1), j in axes(predictions_train_vec, 2)]
    y_monitor_f32 = [Float32.(y) for y in y_monitor]
    features_monitor_f32 = [Float32.(x) for x in features_monitor]
    predictions_monitor_f32 = [Float32.(predictions_monitor_vec[i, j]) for i in axes(predictions_monitor_vec, 1), j in axes(predictions_monitor_vec, 2)]

    best_monitor_loss = Inf32
    best_epoch = 0
    best_ps = train_state.parameters
    best_st = train_state.states
    patience_counter = 0

    @showprogress for epoch in 1:n_epochs
        for j in eachindex(features_train_f32)
            data_j = (predictions_train_f32[:, j], features_train_f32[j], y_train_f32[j])
            (_, _, _, train_state) = Lux.Training.single_train_step!(ad, moe_objective, data_j, train_state)
        end

        train_loss = average_moe_loss(
            predictions_train_f32, features_train_f32, y_train_f32,
            gating, train_state.parameters, train_state.states
        )
        monitor_loss = average_moe_loss(
            predictions_monitor_f32, features_monitor_f32, y_monitor_f32,
            gating, train_state.parameters, train_state.states
        )
        @info "Gating epoch" epoch train_loss monitor_label monitor_loss

        if monitor_loss < best_monitor_loss - min_delta
            best_monitor_loss = monitor_loss
            best_epoch = epoch
            best_ps = train_state.parameters
            best_st = train_state.states
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                @info "Early stopping gating training" epoch best_epoch monitor_label best_monitor_loss
                break
            end
        end
    end

    return (parameters=best_ps, states=best_st, best_epoch=best_epoch, best_monitor_loss=best_monitor_loss)
end

function parse_bool_flag(v::AbstractString, flag::AbstractString)
    lv = lowercase(v)
    lv == "true" && return true
    lv == "false" && return false
    error("Flag $(flag) expects true|false, got: $(v)")
end

function parse_int_flag(v::AbstractString, flag::AbstractString)
    try
        return parse(Int, v)
    catch
        error("Flag $(flag) expects an integer, got: $(v)")
    end
end

function parse_cli_args(args::Vector{String})
    train_set = true
    layers = 1
    model_paths = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--train_set"
            i == length(args) && error("Missing value for --train_set (expected true|false)")
            train_set = parse_bool_flag(args[i + 1], "--train_set")
            i += 2
        elseif arg == "--layers"
            i == length(args) && error("Missing value for --layers (expected integer >= 1)")
            layers = parse_int_flag(args[i + 1], "--layers")
            layers < 1 && error("Flag --layers expects integer >= 1, got: $(layers)")
            i += 2
        elseif startswith(arg, "--")
            error("Unknown flag: $(arg)")
        else
            push!(model_paths, arg)
            i += 1
        end
    end
    return (; train_set, layers, model_paths)
end

reactant_device() = (
    try
        Reactant.default_device()
    catch
        Lux.cpu_device()
    end
)
cpu_device() = Lux.cpu_device()

function same_scaler(s1, s2; atol=1.0f-6)
    length(s1.μ) == length(s2.μ) || return false
    length(s1.σ) == length(s2.σ) || return false
    return maximum(abs.(s1.μ .- s2.μ)) ≤ atol && maximum(abs.(s1.σ .- s2.σ)) ≤ atol
end

function predict_unscaled(model, ps, st, X_scaled; dev=reactant_device())
    Xd = dev(Float32.(X_scaled))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)
    model_compiled = if dev isa ReactantDevice
        @compile model(Xd, ps_d, st_test)
    else
        model
    end
    y_sc, _ = model_compiled(Xd, ps_d, st_test)
    y_sc = cpu_device()(y_sc)
    return Array(y_sc)
end

function to_vecs(Y::AbstractMatrix)
    return [Vector{Float64}(Y[:, j]) for j in 1:size(Y, 2)]
end

function make_features(X_scaled)
    n = size(X_scaled, 3)
    feats = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        x_last_cos = map(cos, X_scaled[:, end, j])
        x_last_sin = map(sin, X_scaled[:, end, j])
        x_last = Float64.(X_scaled[:, end, j])
        feats[j] = vcat(1.0, x_last, x_last_cos, x_last_sin)
    end
    return feats
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    parsed = parse_cli_args(ARGS)
    model_paths = parsed.model_paths
    train_set = parsed.train_set
    layers = parsed.layers

    if length(model_paths) < 2
        println("Usage: julia scripts/adaptive_mixture_local_experts.jl [--train_set true|false] [--layers N] <model1.jld2> <model2.jld2> [more...]")
        return
    end

    models = map(load_jld2_model, model_paths)

    base_meta = models[1].meta
    for m in models[2:end]
        if m.meta.dataset != base_meta.dataset || m.meta.seq_len != base_meta.seq_len || m.meta.horizon != base_meta.horizon
            error("All models must share dataset, seq_len, and horizon. Got $(m.meta.dataset), seq_len=$(m.meta.seq_len), horizon=$(m.meta.horizon)")
        end
        if !same_scaler(m.meta.scaler, base_meta.scaler)
            error("All models must share the same scaler (train split).")
        end
    end

    @info "Loading dataset" dataset = base_meta.dataset seq_len = base_meta.seq_len horizon = base_meta.horizon

    data_dir = joinpath(@__DIR__, "..", "data")
    ds_path = joinpath(data_dir, String(base_meta.dataset))
    Xmat, _ = load_ett(ds_path)

    X3, Y2 = make_sequences(Xmat; seq_len=Int(base_meta.seq_len), horizon=Int(base_meta.horizon))
    split = base_meta.split
    Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = base_meta.scaler
    Xtr_s = scale_inputs(scaler, Xtr)
    Xval_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Ytr_sc = scale_targets(scaler, Ytr)
    Yval_sc = scale_targets(scaler, Yval)
    Yte_sc = scale_targets(scaler, Yte)

    n_forecasters = length(models)
    n_train = size(Xtr, 3)
    n_val = size(Xval, 3)
    n_test = size(Xte, 3)
    d = size(Ytr, 1)
    ot_idx = d  # OT is last column

    # Add two constant baselines: per-dimension min and max from train targets
    n_total = n_forecasters + 2
    predictions_train = Array{Float64}(undef, n_total, d, n_train)
    predictions_val = Array{Float64}(undef, n_total, d, n_val)
    predictions_test = Array{Float64}(undef, n_total, d, n_test)

    @info "Running forecasters" n_forecasters n_train n_val n_test output_dim = d

    for (i, m) in enumerate(models)
        model = build_model(m.model_type, m.config)
        yhat_tr_sc = predict_unscaled(model, m.parameters, m.states, Xtr_s)
        yhat_val_sc = predict_unscaled(model, m.parameters, m.states, Xval_s)
        yhat_te_sc = predict_unscaled(model, m.parameters, m.states, Xte_s)

        predictions_train[i, :, :] = Float64.(yhat_tr_sc)
        predictions_val[i, :, :] = Float64.(yhat_val_sc)
        predictions_test[i, :, :] = Float64.(yhat_te_sc)
        @info "Forecaster ready" index = i model_type = m.model_type path = model_paths[i]
    end

    y_train = to_vecs(Float64.(Ytr_sc))
    y_val = to_vecs(Float64.(Yval_sc))
    y_test = to_vecs(Float64.(Yte_sc))

    use_ot_target_training = occursin("ETTh", String(base_meta.dataset))

    y_q10 = [quantile(Float64.(view(Ytr_sc, i, :)), 0.1) for i in 1:d]
    y_q90 = [quantile(Float64.(view(Ytr_sc, i, :)), 0.9) for i in 1:d]
    idx_min = n_forecasters + 1
    idx_max = n_forecasters + 2
    for j in 1:n_train
        predictions_train[idx_min, :, j] = y_q10
        predictions_train[idx_max, :, j] = y_q90
    end
    for j in 1:n_val
        predictions_val[idx_min, :, j] = y_q10
        predictions_val[idx_max, :, j] = y_q90
    end
    for j in 1:n_test
        predictions_test[idx_min, :, j] = y_q10
        predictions_test[idx_max, :, j] = y_q90
    end
    @info "Added constant baselines" q10_idx = idx_min q90_idx = idx_max

    predictions_train_vec = Array{Vector{Float64}}(undef, n_total, n_train)
    predictions_val_vec = Array{Vector{Float64}}(undef, n_total, n_val)
    predictions_test_vec = Array{Vector{Float64}}(undef, n_total, n_test)
    for i in 1:n_total
        for j in 1:n_train
            predictions_train_vec[i, j] = Vector{Float64}(predictions_train[i, :, j])
        end
        for j in 1:n_val
            predictions_val_vec[i, j] = Vector{Float64}(predictions_val[i, :, j])
        end
        for j in 1:n_test
            predictions_test_vec[i, j] = Vector{Float64}(predictions_test[i, :, j])
        end
    end

    predictions_train_vec_moe = predictions_train_vec
    predictions_val_vec_moe = predictions_val_vec
    y_train_moe = y_train
    y_val_moe = y_val
    if use_ot_target_training
        predictions_train_vec_moe = Array{Vector{Float64}}(undef, n_total, n_train)
        predictions_val_vec_moe = Array{Vector{Float64}}(undef, n_total, n_val)
        for i in 1:n_total
            for j in 1:n_train
                predictions_train_vec_moe[i, j] = [predictions_train[i, ot_idx, j]]
            end
            for j in 1:n_val
                predictions_val_vec_moe[i, j] = [predictions_val[i, ot_idx, j]]
            end
        end
        y_train_moe = [Vector{Float64}([Ytr_sc[ot_idx, j]]) for j in 1:n_train]
        y_val_moe = [Vector{Float64}([Yval_sc[ot_idx, j]]) for j in 1:n_val]
    end

    features_train = make_features(Xtr_s)
    features_val = make_features(Xval_s)
    features_test = make_features(Xte_s)
    n_features = length(features_train[1])

    hidden_state = 64
    gating = if layers == 1
        Chain(Dense(n_features => n_total))
    else
        gating_layers = Any[Dense(n_features => hidden_state)]
        for _ in 2:(layers - 1)
            push!(gating_layers, Dense(hidden_state => hidden_state))
        end
        push!(gating_layers, Dense(hidden_state => n_total))
        Chain(gating_layers...)
    end

    @info "Step 1: Training adaptive mixture of local experts (MLE)" n_features = n_features layers = layers hidden_state = hidden_state target_scope = (use_ot_target_training ? "OT-only" : "all-columns")

    opt = Optimisers.Adam(1f-3)
    gating_state = if train_set
        train_moe!(
            predictions_train_vec_moe, features_train, y_train_moe,
            predictions_val_vec_moe, features_val, y_val_moe,
            gating, opt;
            n_epochs=100, patience=50, min_delta=1f-6, monitor_label="val"
        )
    else
        train_moe!(
            predictions_val_vec_moe, features_val, y_val_moe,
            predictions_train_vec_moe, features_train, y_train_moe,
            gating, opt;
            n_epochs=100, patience=1, min_delta=1f-3, monitor_label="train"
        )
    end
    @info "Best gating checkpoint" train_set best_epoch = gating_state.best_epoch best_monitor_loss = gating_state.best_monitor_loss

    @info "Step 2: Testing adaptive mixture of local experts (MLE)"

    ensemble_mean = hcat([moe_predict(predictions_test_vec[:, j], gating, gating_state.parameters, gating_state.states, features_test[j]) for j in 1:n_test]...)
    ensemble_std = sqrt.(hcat([moe_var(predictions_test_vec[:,j], gating, gating_state.parameters, gating_state.states, features_test[j]) for j in 1:n_test]...))
    gating_weights = hcat([gating_probs(gating, gating_state.parameters, gating_state.states, features_test[j]) for j in 1:n_test]...)

    y_test_mat = Float64.(Yte_sc)
    y_test_ot_sc = y_test_mat[ot_idx:ot_idx, :]
    ensemble_mean_ot_sc = ensemble_mean[ot_idx:ot_idx, :]
    use_ot_metrics = occursin("ETTh", String(base_meta.dataset))
    metrics_scope = "all-columns"
    y_eval = y_test_mat
    ensemble_eval = ensemble_mean
    if use_ot_metrics
        metrics_scope = "OT-only"
        y_eval = y_test_ot_sc
        ensemble_eval = ensemble_mean_ot_sc
    end
    ensemble_metrics = (
        mse=mse_mv(ensemble_eval, y_eval),
        mae=mae_mv(ensemble_eval, y_eval),
        mean_std=mean(ensemble_std),
    )

    @info "Step 3: Performance comparison on test"
    @info "Metric scope" dataset = base_meta.dataset scope = metrics_scope
    individual = []
    for i in 1:n_total
        yhat = predictions_test[i, :, :]
        yhat_ot_sc = yhat[ot_idx:ot_idx, :]
        yhat_eval = yhat
        if use_ot_metrics
            yhat_eval = yhat_ot_sc
        end
        push!(individual, (
            path=i <= n_forecasters ? model_paths[i] : (i == idx_min ? "const_q10_train" : "const_q90_train"),
            mse=mse_mv(yhat_eval, y_eval),
            mae=mae_mv(yhat_eval, y_eval),
        ))
    end

    simple_avg = vec(mean(predictions_test; dims=1)) |> x -> reshape(x, d, n_test)
    simple_avg_ot_sc = simple_avg[ot_idx:ot_idx, :]
    simple_eval = simple_avg
    if use_ot_metrics
        simple_eval = simple_avg_ot_sc
    end
    simple_metrics = (
        mse=mse_mv(simple_eval, y_eval),
        mae=mae_mv(simple_eval, y_eval),
    )

    @info "Dynamic ensemble metrics" ensemble_metrics...
    @info "Simple average metrics" simple_metrics...
    println("METRIC|kind=dynamic|mse=$(ensemble_metrics.mse)|mae=$(ensemble_metrics.mae)")
    println("METRIC|kind=average|mse=$(simple_metrics.mse)|mae=$(simple_metrics.mae)")
    for (i, m) in enumerate(individual)
        @info "Forecaster metrics" index = i path = m.path mse = m.mse mae = m.mae
        println("METRIC|kind=forecaster|index=$(i)|path=$(m.path)|mse=$(m.mse)|mae=$(m.mae)")
    end

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    x_test = 1:n_test
    y_test_mat_T = permutedims(y_test_mat, (2, 1))              # n_test x d
    ensemble_mean_T = permutedims(ensemble_mean, (2, 1))        # n_test x d
    simple_avg_T = permutedims(simple_avg, (2, 1))              # n_test x d

    p1 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Dynamic Ensemble vs Individual (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright
    )
    plot!(p1, x_test, ensemble_mean_T[:, 1],
        ribbon=2 .* ensemble_std,
        label="Dynamic ±2σ",
        lw=2, color=:blue, fillalpha=0.3
    )

    colors = [:red, :green, :orange, :purple, :brown, :pink, :cyan, :gray, :black]
    for i in 1:n_total
        plot!(p1, x_test, predictions_test[i, 1, :],
            label="F$(i)", ls=:dash, alpha=0.6, color=colors[mod1(i, length(colors))]
        )
    end

    p2 = plot(x_test, y_test_mat_T[:, 1],
        label="True (dim 1)", lw=2, color=:black, ls=:dot,
        title="Simple Avg vs Dynamic (Dim 1)",
        xlabel="t", ylabel="y",
        legend=:topright
    )
    plot!(p2, x_test, simple_avg_T[:, 1],
        label="Simple Avg", lw=2, color=:orange, ls=:dash
    )
    plot!(p2, x_test, ensemble_mean_T[:, 1],
        label="Dynamic", lw=2, color=:blue
    )

    p3 = plot(title="Dynamic Precision Weights",
        xlabel="t", ylabel="Gate softmax probabilities",
        legend=:topright
    )
    for i in 1:n_total
        plot!(p3, x_test, gating_weights[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    p4 = plot(title="Uncertainty (Dynamic)",
        xlabel="t", ylabel="Standard Deviation (σ)",
        legend=:topright
    )
    plot!(p4, x_test, sum(ensemble_std,dims=1)', label="Dynamic σ", lw=2, color=:blue)

    all_mses = vcat([m.mse for m in individual], [simple_metrics.mse, ensemble_metrics.mse])
    all_labels = vcat(["F$(i)" for i in 1:n_total], ["Simple Avg", "Dynamic"])
    bar_colors = vcat(fill(:gray, n_total), [:orange, :blue])
    p5 = bar(1:length(all_mses), all_mses,
        title=use_ot_metrics ? "MSE Comparison (Test Set, OT only)" : "MSE Comparison (Test Set, all columns)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45
    )

    p6 = plot(title="Normalized Dynamic Weights",
        xlabel="t", ylabel="Weight (normalized)",
        legend=:outerright
    )
    for i in 1:n_total
        plot!(p6, x_test, gating_weights[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    plt = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1200, 1200))
    plot_file = "viz/adaptive_mixture_local_experts_dynamic_$(base_meta.dataset).png"
    savefig(plt, plot_file)
    @info "Saved visualization" file = plot_file

    @info "Done"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
