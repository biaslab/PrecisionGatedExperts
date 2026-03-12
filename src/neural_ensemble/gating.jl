# ---------------------------------------------------------------------------
# Adaptive Mixture of Local Experts — Gating Network
# (Jacobs, Jordan, Nowlan, Hinton 1991)
# ---------------------------------------------------------------------------

using Lux
using Random
using Optimisers
using ProgressMeter

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
    return (preds .- preds * probs) .^ 2 * probs
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

function build_gating_network(n_features::Int, n_experts::Int, layers::Int, hidden_dim::Int)
    if layers == 1
        return Chain(Dense(n_features => n_experts))
    end
    gating_layers = Any[Dense(n_features => hidden_dim)]
    for _ = 2:(layers-1)
        push!(gating_layers, Dense(hidden_dim => hidden_dim))
    end
    push!(gating_layers, Dense(hidden_dim => n_experts))
    return Chain(gating_layers...)
end

function train_moe!(
    predictions_train_vec,
    features_train,
    y_train,
    predictions_monitor_vec,
    features_monitor,
    y_monitor,
    gating,
    opt;
    n_epochs,
    patience,
    min_delta,
    monitor_label,
)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, gating)
    train_state = Lux.Training.TrainState(gating, ps, st, opt)
    ad = AutoEnzyme()

    y_train_f32 = [Float32.(y) for y in y_train]
    features_train_f32 = [Float32.(x) for x in features_train]
    predictions_train_f32 = [
        Float32.(predictions_train_vec[i, j]) for
        i in axes(predictions_train_vec, 1), j in axes(predictions_train_vec, 2)
    ]
    y_monitor_f32 = [Float32.(y) for y in y_monitor]
    features_monitor_f32 = [Float32.(x) for x in features_monitor]
    predictions_monitor_f32 = [
        Float32.(predictions_monitor_vec[i, j]) for
        i in axes(predictions_monitor_vec, 1), j in axes(predictions_monitor_vec, 2)
    ]

    best_monitor_loss = Inf32
    best_epoch = 0
    best_ps = train_state.parameters
    best_st = train_state.states
    patience_counter = 0

    @showprogress for epoch = 1:n_epochs
        for j in eachindex(features_train_f32)
            data_j = (predictions_train_f32[:, j], features_train_f32[j], y_train_f32[j])
            (_, _, _, train_state) =
                Lux.Training.single_train_step!(ad, moe_objective, data_j, train_state)
        end

        train_loss = average_moe_loss(
            predictions_train_f32,
            features_train_f32,
            y_train_f32,
            gating,
            train_state.parameters,
            train_state.states,
        )
        monitor_loss = average_moe_loss(
            predictions_monitor_f32,
            features_monitor_f32,
            y_monitor_f32,
            gating,
            train_state.parameters,
            train_state.states,
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

    return (
        parameters = best_ps,
        states = best_st,
        best_epoch = best_epoch,
        best_monitor_loss = best_monitor_loss,
    )
end
