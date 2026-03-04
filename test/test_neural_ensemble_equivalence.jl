#!/usr/bin/env julia
"""
Compare old script logic vs new NeuralEnsembleSpecifier pipeline.
Verifies that intermediate data and final metrics are identical.
"""

using ProbabilisticEnsembling
using ProbabilisticEnsembling: NeuralEnsembleSpecifier, Univariate, SimpleFeatures,
    before_neural_ensemble, build_gating_network, train_moe!, moe_predict, moe_var,
    gating_probs, load_dataset, predict_unscaled, find_column_index,
    generate_expert_predictions_three_splits, make_features
using Random
using Statistics
using Lux
using Optimisers
using LinearAlgebra

# ============================================================================
# OLD SCRIPT LOGIC (inlined from dynamic_neural_ensemble_adaptive_mixture_local_experts.jl)
# ============================================================================

function old_script_prepare_data(model_paths)
    models = map(load_jld2_model, model_paths)
    base_meta = models[1].meta

    Xmat, _ = load_ett(joinpath("data", String(base_meta.dataset)))
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
    ot_idx = d

    n_total = n_forecasters + 2
    predictions_train = Array{Float64}(undef, n_total, d, n_train)
    predictions_val = Array{Float64}(undef, n_total, d, n_val)
    predictions_test = Array{Float64}(undef, n_total, d, n_test)

    for (i, m) in enumerate(models)
        model = build_model(m.model_type, m.config)
        predictions_train[i, :, :] = Float64.(predict_unscaled(model, m.parameters, m.states, Xtr_s))
        predictions_val[i, :, :] = Float64.(predict_unscaled(model, m.parameters, m.states, Xval_s))
        predictions_test[i, :, :] = Float64.(predict_unscaled(model, m.parameters, m.states, Xte_s))
    end

    y_q10 = [quantile(Float64.(view(Ytr_sc, i, :)), 0.1) for i in 1:d]
    y_q90 = [quantile(Float64.(view(Ytr_sc, i, :)), 0.9) for i in 1:d]
    idx_min = n_forecasters + 1
    idx_max = n_forecasters + 2
    for j in 1:n_train; predictions_train[idx_min, :, j] = y_q10; predictions_train[idx_max, :, j] = y_q90; end
    for j in 1:n_val;   predictions_val[idx_min, :, j] = y_q10;   predictions_val[idx_max, :, j] = y_q90;   end
    for j in 1:n_test;  predictions_test[idx_min, :, j] = y_q10;  predictions_test[idx_max, :, j] = y_q90;  end

    # to_vecs
    to_vecs(Y) = [Vector{Float64}(Y[:, j]) for j in 1:size(Y, 2)]
    y_train = to_vecs(Float64.(Ytr_sc))
    y_val = to_vecs(Float64.(Yval_sc))

    # vec format
    predictions_train_vec = Array{Vector{Float64}}(undef, n_total, n_train)
    predictions_val_vec = Array{Vector{Float64}}(undef, n_total, n_val)
    predictions_test_vec = Array{Vector{Float64}}(undef, n_total, n_test)
    for i in 1:n_total
        for j in 1:n_train; predictions_train_vec[i, j] = Vector{Float64}(predictions_train[i, :, j]); end
        for j in 1:n_val;   predictions_val_vec[i, j] = Vector{Float64}(predictions_val[i, :, j]);     end
        for j in 1:n_test;  predictions_test_vec[i, j] = Vector{Float64}(predictions_test[i, :, j]);   end
    end

    # Restrict to OT column for training
    predictions_train_vec_moe = Array{Vector{Float64}}(undef, n_total, n_train)
    predictions_val_vec_moe = Array{Vector{Float64}}(undef, n_total, n_val)
    for i in 1:n_total
        for j in 1:n_train; predictions_train_vec_moe[i, j] = [predictions_train[i, ot_idx, j]]; end
        for j in 1:n_val;   predictions_val_vec_moe[i, j] = [predictions_val[i, ot_idx, j]];     end
    end
    y_train_moe = [Vector{Float64}([Ytr_sc[ot_idx, j]]) for j in 1:n_train]
    y_val_moe = [Vector{Float64}([Yval_sc[ot_idx, j]]) for j in 1:n_val]

    # Features (old script's make_features = SimpleFeatures)
    function old_make_features(X_scaled)
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
    features_train = old_make_features(Xtr_s)
    features_val = old_make_features(Xval_s)
    features_test = old_make_features(Xte_s)

    return (
        predictions_train_vec_moe = predictions_train_vec_moe,
        predictions_val_vec_moe = predictions_val_vec_moe,
        predictions_test_vec = predictions_test_vec,
        y_train_moe = y_train_moe,
        y_val_moe = y_val_moe,
        y_test_mat = Float64.(Yte_sc),
        features_train = features_train,
        features_val = features_val,
        features_test = features_test,
        n_total = n_total,
        n_features = length(features_train[1]),
        ot_idx = ot_idx,
    )
end

# ============================================================================
# COMPARISON
# ============================================================================

function arrays_match(a, b, label; atol=1e-10)
    if size(a) != size(b)
        println("  FAIL [$label] sizes differ: $(size(a)) vs $(size(b))")
        return false
    end
    maxdiff = maximum(abs.(a .- b))
    ok = maxdiff <= atol
    status = ok ? "OK" : "FAIL"
    println("  $status [$label] max_diff=$maxdiff")
    return ok
end

function vecs_match(a::Vector{Vector{Float64}}, b::Vector{Vector{Float64}}, label; atol=1e-10)
    if length(a) != length(b)
        println("  FAIL [$label] lengths differ: $(length(a)) vs $(length(b))")
        return false
    end
    maxdiff = maximum(maximum(abs.(a[i] .- b[i])) for i in eachindex(a))
    ok = maxdiff <= atol
    status = ok ? "OK" : "FAIL"
    println("  $status [$label] max_diff=$maxdiff")
    return ok
end

function vec_matrix_match(a::Array{Vector{Float64},2}, b::Array{Vector{Float64},2}, label; atol=1e-10)
    if size(a) != size(b)
        println("  FAIL [$label] sizes differ: $(size(a)) vs $(size(b))")
        return false
    end
    maxdiff = maximum(maximum(abs.(a[i,j] .- b[i,j])) for i in axes(a,1), j in axes(a,2))
    ok = maxdiff <= atol
    status = ok ? "OK" : "FAIL"
    println("  $status [$label] max_diff=$maxdiff")
    return ok
end

function main()
    model_paths = [
        "models/ETTh1_h96_s96_CNN_enzyme.jld2",
        "models/ETTh1_h96_s96_MLP_enzyme.jld2",
        "models/ETTh1_h96_s96_LSTM_enzyme.jld2",
        "models/ETTh1_h96_s96_DLinear_enzyme.jld2",
        "models/ETTh1_h96_s96_NConv_enzyme.jld2",
    ]

    # --- Step 1: Prepare data both ways ---
    println("=== Step 1: Prepare data (OLD way) ===")
    old = old_script_prepare_data(model_paths)

    println("\n=== Step 2: Prepare data (NEW way) ===")
    spec = NeuralEnsembleSpecifier(
        Univariate(),
        "OT",
        Val{:ETTh1}(),
        "data/ETTh1.csv",
        96,
        model_paths,
        true,                # train_set
        1,                   # gating_layers
        64,                  # gating_hidden_dim
        100,                 # n_epochs
        50,                  # patience
        Float32(1e-6),       # min_delta
        Float32(1e-3),       # learning_rate
        SimpleFeatures(),
        [10.0, 90.0],        # selected_quantiles
        "saved_neural_ensemble_models",
    )
    new = before_neural_ensemble(spec)

    # --- Step 3: Compare intermediate data ---
    println("\n=== Step 3: Compare intermediate data ===")
    all_ok = true
    all_ok &= vec_matrix_match(old.predictions_train_vec_moe, new.predictions_train_vec_moe, "predictions_train_vec_moe")
    all_ok &= vec_matrix_match(old.predictions_val_vec_moe, new.predictions_val_vec_moe, "predictions_val_vec_moe")
    all_ok &= vecs_match(old.y_train_moe, new.y_train_moe, "y_train_moe")
    all_ok &= vecs_match(old.y_val_moe, new.y_val_moe, "y_val_moe")
    all_ok &= vecs_match(old.features_train, new.features_train, "features_train")
    all_ok &= vecs_match(old.features_val, new.features_val, "features_val")
    all_ok &= vecs_match(old.features_test, new.features_test, "features_test")
    all_ok &= arrays_match(old.y_test_mat, new.y_test_mat, "y_test_mat")
    println("  n_total: old=$(old.n_total) new=$(new.n_total)")
    println("  n_features: old=$(old.n_features) new=$(new.n_features)")
    println("  ot_idx: old=$(old.ot_idx) new=$(new.col_idx)")

    if !all_ok
        println("\nFAILED: Intermediate data differs. Fix data pipeline before comparing training.")
        return
    end
    println("\nAll intermediate data matches!")

    # --- Step 4: Train with same seed ---
    println("\n=== Step 4: Train OLD with seed 42 ===")
    Random.seed!(42)
    old_gating = Chain(Dense(old.n_features => old.n_total))
    old_opt = Optimisers.Adam(1f-3)
    old_state = train_moe!(
        old.predictions_train_vec_moe, old.features_train, old.y_train_moe,
        old.predictions_val_vec_moe, old.features_val, old.y_val_moe,
        old_gating, old_opt;
        n_epochs=100, patience=50, min_delta=1f-6, monitor_label="val",
    )

    println("\n=== Step 5: Train NEW with seed 42 ===")
    Random.seed!(42)
    new_gating = build_gating_network(new.n_features, new.n_total, 1, 64)
    new_opt = Optimisers.Adam(1f-3)
    new_state = train_moe!(
        new.predictions_train_vec_moe, new.features_train, new.y_train_moe,
        new.predictions_val_vec_moe, new.features_val, new.y_val_moe,
        new_gating, new_opt;
        n_epochs=100, patience=50, min_delta=1f-6, monitor_label="val",
    )

    # --- Step 5: Compare training results ---
    println("\n=== Step 6: Compare training results ===")
    println("  best_epoch: old=$(old_state.best_epoch) new=$(new_state.best_epoch)")
    println("  best_monitor_loss: old=$(old_state.best_monitor_loss) new=$(new_state.best_monitor_loss)")

    # --- Step 6: Compare test metrics ---
    println("\n=== Step 7: Compare test metrics ===")
    n_test = size(old.predictions_test_vec, 2)
    ot_idx = old.ot_idx

    # OLD metrics
    old_ensemble_mean = hcat([moe_predict(old.predictions_test_vec[:, j], old_gating, old_state.parameters, old_state.states, old.features_test[j]) for j in 1:n_test]...)
    old_ensemble_std = sqrt.(hcat([moe_var(old.predictions_test_vec[:, j], old_gating, old_state.parameters, old_state.states, old.features_test[j]) for j in 1:n_test]...))
    old_y_eval = old.y_test_mat[ot_idx:ot_idx, :]
    old_ens_eval = old_ensemble_mean[ot_idx:ot_idx, :]
    old_mse = mse_mv(old_ens_eval, old_y_eval)
    old_mae = mae_mv(old_ens_eval, old_y_eval)

    # NEW metrics
    new_ensemble_mean = hcat([moe_predict(new.predictions_test_vec[:, j], new_gating, new_state.parameters, new_state.states, new.features_test[j]) for j in 1:n_test]...)
    new_ensemble_std = sqrt.(hcat([moe_var(new.predictions_test_vec[:, j], new_gating, new_state.parameters, new_state.states, new.features_test[j]) for j in 1:n_test]...))
    new_col_idx = new.col_idx
    new_y_eval = new.y_test_mat[new_col_idx:new_col_idx, :]
    new_ens_eval = new_ensemble_mean[new_col_idx:new_col_idx, :]
    new_mse = mse_mv(new_ens_eval, new_y_eval)
    new_mae = mae_mv(new_ens_eval, new_y_eval)

    println("  OLD:  MSE=$(old_mse)  MAE=$(old_mae)")
    println("  NEW:  MSE=$(new_mse)  MAE=$(new_mae)")
    println("  DIFF: MSE=$(abs(old_mse - new_mse))  MAE=$(abs(old_mae - new_mae))")

    mse_ok = abs(old_mse - new_mse) < 1e-6
    mae_ok = abs(old_mae - new_mae) < 1e-6
    println("\n=== RESULT ===")
    println("MSE match: $mse_ok")
    println("MAE match: $mae_ok")

    if mse_ok && mae_ok
        println("SUCCESS: Old script and new pipeline produce identical results!")
    else
        println("WARNING: Results differ. Check pipeline logic.")
    end
end

main()
