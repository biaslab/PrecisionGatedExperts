#!/usr/bin/env julia

"""
Adaptive mixture of local experts (Jacobs,Jordan,Nowlan,Hinton)

Usage:
    julia scripts/adaptive_mixture_local_experts.jl 
"""

using Revise
using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Statistics
using LinearAlgebra
using Lux
using Reactant
using Plots

includet("../src/ProbabilisticEnsembling.jl"); using .ProbabilisticEnsembling


# -----------------------------------------------------------------------------
# Adaptive Mixture of Local Experts (Jacobs et al., 1991)
# -----------------------------------------------------------------------------

using Flux
using Statistics

# Set input_dim and n_experts as needed
const input_dim =  size(features, 2)  # or set manually
const n_experts = 3  # Change as needed

# Define expert models (e.g., simple MLPs)
experts = [Chain(Dense(input_dim, 16, relu), Dense(16, 1)) for _ in 1:n_experts]

# Gating network: softmax over linear functions
gating = Chain(Dense(input_dim, n_experts))

# Softmax function for gating outputs
function gating_probs(gating, x)
    Flux.softmax(gating(x))
end

# Mixture of experts prediction
function moe_predict(experts, gating, x)
    ps = gating_probs(gating, x)
    ys = hcat([expert(x) for expert in experts]...)
    sum(ps .* ys; dims=2)
end

# Loss: negative log-likelihood (MSE for regression, weighted by gating)
function moe_loss(experts, gating, x, y)
    ps = gating_probs(gating, x)
    ys = hcat([expert(x) for expert in experts]...)
    # Weighted MSE for each expert, summed over experts
    mse = sum(ps .* (ys .- y).^2; dims=2)
    mean(mse)
end

# Training function
function train_moe!(experts, gating, data, opt, n_epochs=100)
    ps = Flux.params([experts; gating])
    for epoch in 1:n_epochs
        for (x, y) in data
            gs = Flux.gradient(ps) do
                moe_loss(experts, gating, x, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
    end
end

# Example usage (replace with your data loading)
# data = [(x, y), ...]  # x: input, y: target
# opt = ADAM(1e-3)
# train_moe!(experts, gating, data, opt)
# ŷ = moe_predict(experts, gating, x)

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
    if length(ARGS) < 2
        println("Usage: julia scripts/adaptive_mixture_local_experts.jl <model1.jld2> <model2.jld2> [more...]")
        return
    end

    model_paths = ARGS

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
    Xtr_s = scale_inputs(scaler, Xval)
    Xte_s = scale_inputs(scaler, Xte)
    Yval_sc = scale_targets(scaler, Yval)
    Yte_sc = scale_targets(scaler, Yte)


    n_forecasters = length(models)
    n_train = size(Xval, 3)
    n_test = size(Xte, 3)
    d = size(Yval, 1)
    ot_idx = d  # OT is last column

    # Add two constant baselines: per-dimension min and max from train targets
    n_total = n_forecasters + 2
    predictions_train = Array{Float64}(undef, n_total, d, n_train)
    predictions_test = Array{Float64}(undef, n_total, d, n_test)

    @info "Running forecasters" n_forecasters n_train n_test output_dim = d

    for (i, m) in enumerate(models)
        model = build_model(m.model_type, m.config)
        yhat_tr_sc = predict_unscaled(model, m.parameters, m.states, Xtr_s)
        yhat_te_sc = predict_unscaled(model, m.parameters, m.states, Xte_s)

        predictions_train[i, :, :] = Float64.(yhat_tr_sc)
        predictions_test[i, :, :] = Float64.(yhat_te_sc)
        @info "Forecaster ready" index = i model_type = m.model_type path = model_paths[i]
    end

    y_train = to_vecs(Float64.(Yval_sc))
    y_test = to_vecs(Float64.(Yte_sc))

    y_q10 = [quantile(Float64.(view(Yval_sc, i, :)), 0.1) for i in 1:d]
    y_q90 = [quantile(Float64.(view(Yval_sc, i, :)), 0.9) for i in 1:d]
    idx_min = n_forecasters + 1
    idx_max = n_forecasters + 2
    for j in 1:n_train
        predictions_train[idx_min, :, j] = y_q10
        predictions_train[idx_max, :, j] = y_q90
    end
    for j in 1:n_test
        predictions_test[idx_min, :, j] = y_q10
        predictions_test[idx_max, :, j] = y_q90
    end
    @info "Added constant baselines" q10_idx = idx_min q90_idx = idx_max

    predictions_train_vec = Array{Vector{Float64}}(undef, n_total, n_train)
    predictions_test_vec = Array{Vector{Float64}}(undef, n_total, n_test)
    for i in 1:n_total
        for j in 1:n_train
            predictions_train_vec[i, j] = Vector{Float64}(predictions_train[i, :, j])
        end
        for j in 1:n_test
            predictions_test_vec[i, j] = Vector{Float64}(predictions_test[i, :, j])
        end
    end

    features_train = make_features(Xtr_s)
    features_test = make_features(Xte_s)
    n_features = length(features_train[1])
    #N_FEATURES[] = n_features


    @info "Step 1: Training adaptive mixture of local experts (MLE)" n_features = n_features
    # Prepare data for Flux: features_train and y_train as tuples
    train_data = [(Float32.(features_train[j]), Float32.(y_train[j])) for j in 1:n_train]
    opt = Flux.ADAM(1e-3)
    train_moe!(experts, gating, train_data, opt, n_epochs=100)

    @info "Step 2: Testing adaptive mixture of local experts (MLE)"
    # Predict on test set
    ensemble_mean = hcat([moe_predict(experts, gating, Float32.(features_test[j])) for j in 1:n_test]...)
    # For uncertainty, you could compute the variance across experts weighted by gating probabilities
    function moe_var(experts, gating, x)
        ps = gating_probs(gating, x)
        ys = hcat([expert(x) for expert in experts]...)
        μ = sum(ps .* ys; dims=2)
        var = sum(ps .* (ys .- μ).^2; dims=2)
        return var
    end
    ensemble_std = sqrt.(hcat([moe_var(experts, gating, Float32.(features_test[j])) for j in 1:n_test]...))

    # For visualization, you can also extract the gating weights for each test sample
    gating_weights = hcat([gating_probs(gating, Float32.(features_test[j])) for j in 1:n_test]...)

    y_test_mat = Float64.(Yte_sc)
    y_test_ot_sc = y_test_mat[ot_idx:ot_idx, :]
    ensemble_mean_ot_sc = ensemble_mean[ot_idx:ot_idx, :]
    ensemble_metrics = (
        mse=mse_mv(ensemble_mean_ot_sc, y_test_ot_sc),
        mae=mae_mv(ensemble_mean_ot_sc, y_test_ot_sc),
        rmse=rmse_mv(ensemble_mean_ot_sc, y_test_ot_sc),
        r2=r2_mv(ensemble_mean_ot_sc, y_test_ot_sc),
        mape=mape_mv(ensemble_mean_ot_sc, y_test_ot_sc),
        smape=smape_mv(ensemble_mean_ot_sc, y_test_ot_sc),
        mse_all=mse_mv(ensemble_mean, y_test_mat),
        mae_all=mae_mv(ensemble_mean, y_test_mat),
        rmse_all=rmse_mv(ensemble_mean, y_test_mat),
        r2_all=r2_mv(ensemble_mean, y_test_mat),
        mape_all=mape_mv(ensemble_mean, y_test_mat),
        smape_all=smape_mv(ensemble_mean, y_test_mat),
        mean_std=mean(ensemble_std),
    )

    @info "Step 3: Performance comparison on test"
    individual = []
    for i in 1:n_total
        yhat = predictions_test[i, :, :]
        yhat_ot_sc = yhat[ot_idx:ot_idx, :]
        push!(individual, (
            path=i <= n_forecasters ? model_paths[i] : (i == idx_min ? "const_q10_train" : "const_q90_train"),
            mse=mse_mv(yhat_ot_sc, y_test_ot_sc),
            mae=mae_mv(yhat_ot_sc, y_test_ot_sc),
            rmse=rmse_mv(yhat_ot_sc, y_test_ot_sc),
            r2=r2_mv(yhat_ot_sc, y_test_ot_sc),
            mape=mape_mv(yhat_ot_sc, y_test_ot_sc),
            smape=smape_mv(yhat_ot_sc, y_test_ot_sc),
            mse_all=mse_mv(yhat, y_test_mat),
            mae_all=mae_mv(yhat, y_test_mat),
            rmse_all=rmse_mv(yhat, y_test_mat),
            r2_all=r2_mv(yhat, y_test_mat),
            mape_all=mape_mv(yhat, y_test_mat),
            smape_all=smape_mv(yhat, y_test_mat),
        ))
    end

    simple_avg = vec(mean(predictions_test; dims=1)) |> x -> reshape(x, d, n_test)
    simple_avg_ot_sc = simple_avg[ot_idx:ot_idx, :]
    simple_metrics = (
        mse=mse_mv(simple_avg_ot_sc, y_test_ot_sc),
        mae=mae_mv(simple_avg_ot_sc, y_test_ot_sc),
        rmse=rmse_mv(simple_avg_ot_sc, y_test_ot_sc),
        r2=r2_mv(simple_avg_ot_sc, y_test_ot_sc),
        mape=mape_mv(simple_avg_ot_sc, y_test_ot_sc),
        smape=smape_mv(simple_avg_ot_sc, y_test_ot_sc),
        mse_all=mse_mv(simple_avg, y_test_mat),
        mae_all=mae_mv(simple_avg, y_test_mat),
        rmse_all=rmse_mv(simple_avg, y_test_mat),
        r2_all=r2_mv(simple_avg, y_test_mat),
        mape_all=mape_mv(simple_avg, y_test_mat),
        smape_all=smape_mv(simple_avg, y_test_mat),
    )

    @info "Dynamic ensemble metrics" ensemble_metrics...
    @info "Simple average metrics" simple_metrics...
    for (i, m) in enumerate(individual)
        @info "Forecaster metrics" index = i path = m.path mse = m.mse mae = m.mae rmse = m.rmse r2 = m.r2 mape = m.mape smape = m.smape
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
        xlabel="t", ylabel="γ(x) = exp(w'x)",
        legend=:topright
    )
    for i in 1:n_total
        plot!(p3, x_test, γ_values_all[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
    end

    p4 = plot(title="Uncertainty (Dynamic)",
        xlabel="t", ylabel="Standard Deviation (σ)",
        legend=:topright
    )
    plot!(p4, x_test, ensemble_std, label="Dynamic σ", lw=2, color=:blue)

    all_mses = vcat([m.mse for m in individual], [simple_metrics.mse, ensemble_metrics.mse])
    all_labels = vcat(["F$(i)" for i in 1:n_total], ["Simple Avg", "Dynamic"])
    bar_colors = vcat(fill(:gray, n_total), [:orange, :blue])
    p5 = bar(1:length(all_mses), all_mses,
        title="MSE Comparison (Test Set, avg over dims)",
        xlabel="Method", ylabel="MSE",
        xticks=(1:length(all_mses), all_labels),
        legend=false, color=bar_colors,
        xrotation=45
    )

    normalized_weights = γ_values_all ./ sum(γ_values_all; dims=1)
    p6 = plot(title="Normalized Dynamic Weights",
        xlabel="t", ylabel="Weight (normalized)",
        legend=:outerright
    )
    for i in 1:n_total
        plot!(p6, x_test, normalized_weights[i, :], label="F$(i)", lw=2, color=colors[mod1(i, length(colors))])
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
