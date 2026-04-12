#!/usr/bin/env julia

# Lux neural network for OT REGRESSION (not classification).
# Same XOR dataset, but predicting continuous OT from (x1, x2).
# Baseline to see how well a classical NN can do regression on this problem.

using ADTypes
using CSV
using DataFrames
using Enzyme
using Lux
using NNlib: softmax
using Optimisers
using Plots
using Random
using StableRNGs
using Statistics

function regression_metrics(y_true, y_pred)
    err = y_pred .- y_true
    mse = mean(err .^ 2)
    rmse = sqrt(mse)
    mae = mean(abs.(err))
    return (; mse, rmse, mae)
end

function build_model(kind::Symbol; hidden_dim::Int = 16)
    if kind == :one_layer
        return Chain(Dense(2 => 1))
    elseif kind == :two_layer
        return Chain(
            Dense(2 => hidden_dim, tanh),
            Dense(hidden_dim => 1),
        )
    elseif kind == :three_layer
        return Chain(
            Dense(2 => hidden_dim, tanh),
            Dense(hidden_dim => hidden_dim, tanh),
            Dense(hidden_dim => 1),
        )
    elseif kind == :two_layer_relu
        return Chain(
            Dense(2 => hidden_dim, relu),
            Dense(hidden_dim => 1),
        )
    elseif kind == :two_layer_softmax
        return Chain(
            Dense(2 => hidden_dim),
            x -> softmax(x; dims = 1),
            Dense(hidden_dim => 1),
        )
    elseif kind == :three_layer_relu
        return Chain(
            Dense(2 => hidden_dim, relu),
            Dense(hidden_dim => hidden_dim, relu),
            Dense(hidden_dim => 1),
        )
    elseif kind == :three_layer_softmax
        return Chain(
            Dense(2 => hidden_dim),
            x -> softmax(x; dims = 1),
            Dense(hidden_dim => hidden_dim),
            x -> softmax(x; dims = 1),
            Dense(hidden_dim => 1),
        )
    end
    error("Unsupported model kind: $kind")
end

function regression_loss(model, ps, st, data)
    x, y = data
    pred, st_ = model(x, ps, st)
    loss = mean((pred .- y) .^ 2)
    return loss, st_, (; pred = pred)
end

function fit_regressor(
    df_train::DataFrame;
    kind::Symbol = :two_layer,
    epochs::Int = 2000,
    learning_rate::Float32 = 1f-3,
    patience::Int = 200,
    hidden_dim::Int = 16,
    seed::Int = 2026,
    optimizer = Optimisers.Descent(1f-3),
)
    model = build_model(kind; hidden_dim = hidden_dim)
    rng = StableRNG(seed)
    ps, st = Lux.setup(rng, model)
    train_state = Lux.Training.TrainState(model, ps, st, optimizer)
    ad = AutoEnzyme()

    x = permutedims(Float32.(Matrix(df_train[:, [:x1, :x2]])))
    y = reshape(Float32.(df_train.OT), 1, nrow(df_train))

    best_loss = Inf32
    best_epoch = 0
    best_ps = train_state.parameters
    best_st = train_state.states
    patience_counter = 0

    for epoch in 1:epochs
        (_, loss, _, train_state) = Lux.Training.single_train_step!(
            ad, regression_loss, (x, y), train_state
        )

        if loss < best_loss
            best_loss = loss
            best_epoch = epoch
            best_ps = train_state.parameters
            best_st = train_state.states
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                @info "Early stopping" kind epoch best_epoch best_loss
                break
            end
        end

        if epoch == 1 || epoch % 500 == 0
            @info "Training regressor" kind epoch loss
        end
    end

    return (
        model = model,
        parameters = best_ps,
        states = Lux.testmode(best_st),
        best_epoch = best_epoch,
        best_loss = best_loss,
    )
end

function predict(fitted, df::DataFrame)
    x = permutedims(Float32.(Matrix(df[:, [:x1, :x2]])))
    pred, _ = fitted.model(x, fitted.parameters, fitted.states)
    return vec(pred)
end

function predict_xy(fitted, x1_vals, x2_vals)
    x = permutedims(Float32.(hcat(x1_vals, x2_vals)))
    pred, _ = fitted.model(x, fitted.parameters, fitted.states)
    return vec(pred)
end

function split_dataset(df::DataFrame; train_fraction::Float64 = 0.8, seed::Int = 2026)
    rng = StableRNG(seed)
    idx = randperm(rng, nrow(df))
    n_train = round(Int, train_fraction * nrow(df))
    return df[idx[1:n_train], :], df[idx[n_train+1:end], :]
end

function load_dataset(path::AbstractString)
    isfile(path) || error("Dataset not found: $path.")
    CSV.read(path, DataFrame)
end

function add_true_boundary!(p, x_range)
    xs = range(x_range...; length = 300)
    plot!(p, xs, zero.(xs); linewidth = 2.5, color = :black, linestyle = :dash, label = "")
    vline!(p, [0.0]; linewidth = 2.5, color = :black, linestyle = :dash, label = "true")
end

function make_panels(df_test, y_pred, fitted, title_prefix; n_grid::Int = 60)
    margin = 0.3
    x1r = (minimum(df_test.x1) - margin, maximum(df_test.x1) + margin)
    x2r = (minimum(df_test.x2) - margin, maximum(df_test.x2) + margin)

    x1s = range(x1r...; length = n_grid)
    x2s = range(x2r...; length = n_grid)
    x1_grid = Float64[]
    x2_grid = Float64[]
    for x2 in x2s, x1 in x1s
        push!(x1_grid, x1)
        push!(x2_grid, x2)
    end

    y_grid = predict_xy(fitted, x1_grid, x2_grid)
    mean_grid = reshape(y_grid, n_grid, n_grid)

    p1 = contourf(
        x1s, x2s, mean_grid';
        color = :RdBu, levels = 20,
        xlabel = "x1", ylabel = "x2",
        title = "$title_prefix — predicted y",
        colorbar_title = "y",
    )
    add_true_boundary!(p1, x1r)

    sq_err = (y_pred .- df_test.OT) .^ 2
    clim = quantile(sq_err, 0.95)
    p2 = scatter(
        df_test.x1, df_test.x2;
        marker_z = sq_err,
        color = :hot, clims = (0, clim),
        markersize = 4, markerstrokewidth = 0, alpha = 0.9,
        xlabel = "x1", ylabel = "x2",
        title = "$title_prefix — (y-y_hat)^2",
        colorbar_title = "squared error",
        label = "",
    )
    add_true_boundary!(p2, x1r)

    return p1, p2
end

slugify(s::AbstractString) = replace(lowercase(strip(s)), r"[^a-z0-9]+" => "_")

function main()
    dataset_configs = [
        (name = "xor", path = "test_dataset/xor_dataset.csv"),
        (name = "xor_simple", path = "test_dataset/xor_simple_dataset.csv"),
    ]

    optimizer_configs = [
        (name = "Descent", builder = (lr) -> Optimisers.Descent(lr)),
        (name = "Momentum", builder = (lr) -> Optimisers.Momentum(lr, 0.9f0)),
    ]

    configs = [
        (kind = :one_layer, hidden_dim = 16, label = "1-layer linear", activation = "linear"),
        (kind = :two_layer, hidden_dim = 8, label = "2-layer Tanh h=8", activation = "tanh"),
        (kind = :two_layer, hidden_dim = 16, label = "2-layer Tanh h=16", activation = "tanh"),
        (kind = :two_layer, hidden_dim = 32, label = "2-layer Tanh h=32", activation = "tanh"),
        (kind = :three_layer, hidden_dim = 16, label = "3-layer Tanh h=16", activation = "tanh"),
        (kind = :two_layer_relu, hidden_dim = 8, label = "2-layer ReLU h=8", activation = "relu"),
        (kind = :two_layer_relu, hidden_dim = 16, label = "2-layer ReLU h=16", activation = "relu"),
        (kind = :two_layer_relu, hidden_dim = 32, label = "2-layer ReLU h=32", activation = "relu"),
        (kind = :three_layer_relu, hidden_dim = 16, label = "3-layer ReLU h=16", activation = "relu"),
        (kind = :two_layer_softmax, hidden_dim = 8, label = "2-layer softmax h=8", activation = "softmax"),
        (kind = :two_layer_softmax, hidden_dim = 16, label = "2-layer softmax h=16", activation = "softmax"),
        (kind = :two_layer_softmax, hidden_dim = 32, label = "2-layer softmax h=32", activation = "softmax"),
        (kind = :three_layer_softmax, hidden_dim = 16, label = "3-layer softmax h=16", activation = "softmax"),
    ]

    metrics_rows = DataFrame(
        dataset = String[],
        optimizer_method = String[],
        activation = String[],
        method = String[],
        mse = Float64[],
        rmse = Float64[],
        mae = Float64[],
        best_epoch = Union{Missing,Int}[],
        train_loss = Union{Missing,Float64}[],
    )

    grouped_panels = Dict{Tuple{String,String,String},Vector{Any}}()

    for (didx, ds_cfg) in enumerate(dataset_configs)
        df = load_dataset(ds_cfg.path)
        df_train, df_test = split_dataset(df; seed = 2027 + didx)

        y_oracle = ifelse.(df_test.regime .== 1, df_test.pred_a, df_test.pred_b)
        oracle_metrics = regression_metrics(df_test.OT, y_oracle)

        println("\nOT Regression on $(uppercase(ds_cfg.name)) dataset")
        println("========================================")
        println(
            "  oracle metrics: " *
            "MSE=$(round(oracle_metrics.mse; digits = 6)), " *
            "RMSE=$(round(oracle_metrics.rmse; digits = 6)), " *
            "MAE=$(round(oracle_metrics.mae; digits = 6))"
        )
        println("  n_train = $(nrow(df_train)), n_test = $(nrow(df_test))")
        println()

        for opt_cfg in optimizer_configs
            push!(
                metrics_rows,
                (
                    dataset = ds_cfg.name,
                    optimizer_method = opt_cfg.name,
                    activation = "oracle",
                    method = "oracle_switch",
                    mse = oracle_metrics.mse,
                    rmse = oracle_metrics.rmse,
                    mae = oracle_metrics.mae,
                    best_epoch = missing,
                    train_loss = missing,
                ),
            )

            for cfg in configs
                @info "Training" dataset = ds_cfg.name optimizer = opt_cfg.name label = cfg.label
                fitted = fit_regressor(
                    df_train;
                    kind = cfg.kind,
                    hidden_dim = cfg.hidden_dim,
                    learning_rate = 1f-3,
                    seed = 2026,
                    optimizer = opt_cfg.builder(1f-3),
                )
                y_pred = predict(fitted, df_test)
                pred_metrics = regression_metrics(df_test.OT, y_pred)

                println(
                    "[$(ds_cfg.name) | $(opt_cfg.name)] $(cfg.label) | " *
                    "MSE = $(round(pred_metrics.mse; digits = 6)) | " *
                    "RMSE = $(round(pred_metrics.rmse; digits = 6)) | " *
                    "MAE = $(round(pred_metrics.mae; digits = 6)) | " *
                    "best_epoch = $(fitted.best_epoch) | " *
                    "train_loss = $(round(fitted.best_loss; digits=6))"
                )

                push!(
                    metrics_rows,
                    (
                        dataset = ds_cfg.name,
                        optimizer_method = opt_cfg.name,
                        activation = cfg.activation,
                        method = cfg.label,
                        mse = pred_metrics.mse,
                        rmse = pred_metrics.rmse,
                        mae = pred_metrics.mae,
                        best_epoch = fitted.best_epoch,
                        train_loss = fitted.best_loss,
                    ),
                )

                title_prefix = "$(cfg.label) ($(ds_cfg.name), $(opt_cfg.name))"
                p1, p2 = make_panels(df_test, y_pred, fitted, title_prefix)
                group_key = (cfg.activation, ds_cfg.name, opt_cfg.name)
                if !haskey(grouped_panels, group_key)
                    grouped_panels[group_key] = Any[]
                end
                push!(grouped_panels[group_key], p1)
                push!(grouped_panels[group_key], p2)
            end
        end
    end

    mkpath("test_dataset/viz")
    for (key, panels) in grouped_panels
        activation, dataset_name, optimizer_name = key
        n = length(panels)
        p_all = plot(
            panels...;
            layout = (max(1, div(n, 2)), 2),
            size = (1200, 300 * max(1, div(n, 2))),
            margin = 5Plots.mm,
        )
        out_path = "test_dataset/viz/lux_regression_$(slugify(dataset_name))_$(slugify(optimizer_name))_$(slugify(activation)).png"
        savefig(p_all, out_path)
        @info "Saved grouped plot" dataset = dataset_name optimizer = optimizer_name activation = activation path = out_path
    end

    metrics_out = "test_dataset/viz/lux_regression_metrics.csv"
    CSV.write(metrics_out, metrics_rows)
    @info "Saved metrics CSV" path = metrics_out
end

main()
