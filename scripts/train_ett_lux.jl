#!/usr/bin/env julia

# Train LSTM and 1D-CNN with Lux.jl on ETTh1/ETTh2 and save models.

using Pkg
using Random, Statistics
using CSV, DataFrames
using Lux
using Optimisers
using Zygote
using NNlib
using BSON: @save

# Optional CUDA support if available
const HAS_CUDA = try
    Base.find_package("CUDA") !== nothing && (Base.require(Main, :CUDA); true)
catch
    false
end
if HAS_CUDA
    import CUDA
end

function get_device()
    if HAS_CUDA && CUDA.functional()
        return (to=x -> CUDA.cu(x), on="cuda")
    else
        return (to=x -> x, on="cpu")
    end
end

include(joinpath(@__DIR__, "utils.jl"))
using .Utils: StandardScaler, fit_scaler, scale_inputs, scale_targets, inverse_targets,
    find_dataset_csv, load_ett, make_sequences, train_val_test_split,
    batches, mse, mae, rmse, r2, mape, compute_metrics

# LSTM via explicit time scan over LSTMCell to avoid version-specific wrappers
function lstm_forward_batch(lstm_cell, head, ps_cell, st_cell, ps_head, st_head, xb)
    stc = st_cell
    h = nothing
    @inbounds for t in 1:size(xb, 2)
        xt = @view xb[:, t, :]
        h, stc = Lux.apply(lstm_cell, xt, ps_cell, stc)
    end
    # Some Lux versions return (h, c) as the output of LSTMCell; take hidden
    if h isa Tuple
        h = h[1]
    end
    y, sth = Lux.apply(head, h, ps_head, st_head)
    return y, (stc, sth)
end

function train_lstm!(input_dim::Int, out_dim::Int, Xtr, Ytr, Xval, Yval; hidden::Int=64, epochs=5, batchsize=128, lr=1e-3, device=get_device())
    to, devname = device
    rng = Random.default_rng()
    lstm_cell = LSTMCell(input_dim => hidden)
    head = Chain(Dense(hidden => 32, relu), Dense(32 => out_dim))
    ps_cell, st_cell = Lux.setup(rng, lstm_cell)
    ps_head, st_head = Lux.setup(rng, head)
    opt_cell = Optimisers.setup(Optimisers.Adam(lr), ps_cell)
    opt_head = Optimisers.setup(Optimisers.Adam(lr), ps_head)

    Xtr_d, Ytr_d = to(Xtr), to(Ytr)
    Xval_d, Yval_d = to(Xval), to(Yval)

    best_val = Inf
    best_ep = 0
    best_ps_cell = deepcopy(ps_cell)
    best_ps_head = deepcopy(ps_head)

    for ep in 1:epochs
        # Shuffle along batch dimension
        idx = randperm(size(Xtr_d, 3))
        Xtr_d = Xtr_d[:, :, idx]
        Ytr_d = Ytr_d[:, idx]

        lsum = 0.0
        nb = 0
        for (xb, yb) in batches(Xtr_d, Ytr_d, batchsize)
            nb += 1
            # Loss and grads wrt both param sets
            function loss_fun(pc, ph)
                ŷ, _ = lstm_forward_batch(lstm_cell, head, pc, st_cell, ph, st_head, xb)
                return mean((ŷ .- yb) .^ 2)
            end
            g_cell, g_head = Zygote.gradient(loss_fun, ps_cell, ps_head)
            opt_cell, ps_cell = Optimisers.update(opt_cell, ps_cell, g_cell)
            opt_head, ps_head = Optimisers.update(opt_head, ps_head, g_head)
            # Log loss
            lsum += loss_fun(ps_cell, ps_head)
        end
        # Validation
        function val_loss_fun()
            ŷ, _ = lstm_forward_batch(lstm_cell, head, ps_cell, st_cell, ps_head, st_head, Xval_d)
            mean((ŷ .- Yval_d) .^ 2)
        end
        v = val_loss_fun()
        if v < best_val
            best_val = v
            best_ep = ep
            best_ps_cell = deepcopy(ps_cell)
            best_ps_head = deepcopy(ps_head)
        end
        @info "[LSTM] epoch=$(ep) train_mse=$(lsum/nb) val_mse=$(v) on=$(devname)"
    end
    # Load best params
    ps_cell = best_ps_cell
    ps_head = best_ps_head
    return (lstm_cell=lstm_cell, head=head, ps_cell=ps_cell, st_cell=st_cell, ps_head=ps_head, st_head=st_head, best_epoch=best_ep, best_val_mse=best_val)
end

function build_cnn_layers(input_dim::Int, out_dim::Int)
    # Use strided convs to downsample, avoid separate pooling layers
    conv1 = Conv((7,), input_dim => 64, relu, pad=(1,), stride=(2,))
    conv2 = Conv((7,), 64 => 64, relu, pad=(1,), stride=(2,))
    fc1 = Dense(64 => 32, relu)
    fc2 = Dense(32 => out_dim)
    return conv1, conv2, fc1, fc2
end

function cnn_forward_batch(layers, params, states, xb)
    (conv1, conv2, fc1, fc2) = layers
    (ps1, ps2, ps3, ps4) = params
    (st1, st2, st3, st4) = states
    # Permute to (seq, channels, N) for this NNlib version
    x = permutedims(xb, (2, 1, 3))
    x, st1 = Lux.apply(conv1, x, ps1, st1)
    x, st2 = Lux.apply(conv2, x, ps2, st2)
    # Global average over time dimension (first)
    x = mean(x; dims=1)  # (1, C, N)
    x = reshape(x, size(x, 2), size(x, 3)) # (C, N)
    x, st3 = Lux.apply(fc1, x, ps3, st3)
    y, st4 = Lux.apply(fc2, x, ps4, st4)
    return y, (st1, st2, st3, st4)
end

function train_cnn!(input_dim::Int, out_dim::Int, Xtr, Ytr, Xval, Yval; epochs=5, batchsize=128, lr=1e-3, device=get_device())
    to, devname = device
    rng = Random.default_rng()
    layers = build_cnn_layers(input_dim, out_dim)
    ps = map(l -> Lux.setup(rng, l)[1], layers)
    st = map(l -> Lux.setup(rng, l)[2], layers)
    opts = (
        Optimisers.setup(Optimisers.Adam(lr), ps[1]),
        Optimisers.setup(Optimisers.Adam(lr), ps[2]),
        Optimisers.setup(Optimisers.Adam(lr), ps[3]),
        Optimisers.setup(Optimisers.Adam(lr), ps[4]),
    )

    Xtr_d, Ytr_d = to(Xtr), to(Ytr)
    Xval_d, Yval_d = to(Xval), to(Yval)

    best_val = Inf
    best_ep = 0
    best_ps = deepcopy(ps)

    for ep in 1:epochs
        idx = randperm(size(Xtr_d, 3))
        Xtr_d = Xtr_d[:, :, idx]
        Ytr_d = Ytr_d[:, idx]
        lsum = 0.0
        nb = 0
        for (xb, yb) in batches(Xtr_d, Ytr_d, batchsize)
            nb += 1
            loss_fun(ps1, ps2, ps3, ps4) = begin
                ŷ, _ = cnn_forward_batch(layers, (ps1, ps2, ps3, ps4), st, xb)
                mean((ŷ .- yb) .^ 2)
            end
            g = Zygote.gradient(loss_fun, ps...)
            opt1, p1 = Optimisers.update(opts[1], ps[1], g[1])
            opt2, p2 = Optimisers.update(opts[2], ps[2], g[2])
            opt3, p3 = Optimisers.update(opts[3], ps[3], g[3])
            opt4, p4 = Optimisers.update(opts[4], ps[4], g[4])
            opts = (opt1, opt2, opt3, opt4)
            ps = (p1, p2, p3, p4)
            lsum += loss_fun(ps...)
        end
        # Validation
        function val_loss_fun()
            ŷ, _ = cnn_forward_batch(layers, ps, st, Xval_d)
            mean((ŷ .- Yval_d) .^ 2)
        end
        v = val_loss_fun()
        if v < best_val
            best_val = v
            best_ep = ep
            best_ps = deepcopy(ps)
        end
        @info "[CNN] epoch=$(ep) train_mse=$(lsum/nb) val_mse=$(v) on=$(devname)"
    end
    ps = best_ps
    return (layers=layers, ps=ps, st=st, best_epoch=best_ep, best_val_mse=best_val)
end

function train!(model, Xtr, Ytr, Xval, Yval; epochs=5, batchsize=128, lr=1e-3, device=get_device())
    to, devname = device
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.setup(Optimisers.Adam(lr), ps)

    todev(x) = to(x)
    Xtr_d, Ytr_d = todev(Xtr), todev(Ytr)
    Xval_d, Yval_d = todev(Xval), todev(Yval)

    for ep in 1:epochs
        # Shuffle
        idx = randperm(size(Xtr_d, 3))
        Xtr_d = Xtr_d[:, :, idx]
        Ytr_d = Ytr_d[:, idx]
        # Batches
        lsum = 0.0
        nb = 0
        for (xb, yb) in batches(Xtr_d, Ytr_d, batchsize)
            nb += 1
            # Forward for loss logging (do not carry state across batches)
            ŷ, _ = Lux.apply(model, xb, ps, st)
            l = mean((ŷ .- yb) .^ 2)
            # Compute gradients w.r.t ps
            gs = Zygote.gradient(ps) do p
                ŷ2, _ = Lux.apply(model, xb, p, st)
                mean((ŷ2 .- yb) .^ 2)
            end
            opt, ps = Optimisers.update(opt, ps, gs)
            lsum += l
        end
        # Validation
        val_loss = mean((Lux.apply(model, Xval_d, ps, st)[1] .- Yval_d) .^ 2)
        @info "[$(nameof(typeof(model)))] epoch=$(ep) train_mse=$(lsum/nb) val_mse=$(val_loss) on=$(devname)"
    end

    return ps, st
end

function save_model(save_path::AbstractString, model, ps, st, meta)
    mkpath(dirname(save_path))
    @save save_path model ps st meta
end

function main()
    # Settings
    data_dir = joinpath(@__DIR__, "..", "data")
    models_dir = joinpath(@__DIR__, "..", "models")
    datasets = ["ETTh1", "ETTh2"]
    seq_len = parse(Int, get(ENV, "SEQ_LEN", "96"))
    horizons = let hs = get(ENV, "HORIZONS", "")
        h = get(ENV, "HORIZON", "")
        if !isempty(hs)
            parse.(Int, split(hs, [',', ';', ' ']; keepempty=false))
        elseif !isempty(h)
            [parse(Int, h)]
        else
            [96, 192, 336, 720]
        end
    end
    epochs = parse(Int, get(ENV, "EPOCHS", "5"))
    batchsize = parse(Int, get(ENV, "BATCHSIZE", "128"))
    lr = parse(Float64, get(ENV, "LR", "1e-3"))

    device = get_device()

    for ds in datasets
        ds_path = joinpath(data_dir, ds)
        @info "Loading dataset $(ds_path)"
        Xmat, feat_cols = load_ett(ds_path)
        @info "Loaded dataset $(ds_path)"
        for H in horizons
            @info "Building sequences with horizon=$(H)"
            X3, Y2 = make_sequences(Xmat; seq_len=seq_len, horizon=H)
            Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(0.6, 0.2, 0.2))
            scaler = fit_scaler(Xtr)
            Xtr = scale_inputs(scaler, Xtr)
            Xval = scale_inputs(scaler, Xval)
            Xte = scale_inputs(scaler, Xte)
            # scale targets per-feature too (multivariate)
            Ytr = scale_targets(scaler, Ytr)
            Yval = scale_targets(scaler, Yval)
            Yte_sc = scale_targets(scaler, Yte)

            input_dim = size(Xtr, 1)
            out_dim = size(Ytr, 1)
            target_stats = nothing

            # LSTM via manual scan over time
            @info "Training LSTM on $(ds) horizon=$(H)"
            lstm_art = train_lstm!(input_dim, out_dim, Xtr, Ytr, Xval, Yval; hidden=64, epochs=epochs, batchsize=batchsize, lr=lr, device=device)
            # Predict and compute test metrics on original scale
            function predict_lstm(X)
                Xd = device.to(X)
                ŷ, _ = lstm_forward_batch(lstm_art.lstm_cell, lstm_art.head, lstm_art.ps_cell, lstm_art.st_cell, lstm_art.ps_head, lstm_art.st_head, Xd)
                Array(ŷ)
            end
            ŷ_sc = predict_lstm(Xte)
            ŷ = inverse_targets(scaler, ŷ_sc)
            y = inverse_targets(scaler, Yte_sc)
            lstm_test_metrics = (
                mse=mse(ŷ, y),
                mae=mae(ŷ, y),
                rmse=rmse(ŷ, y),
                r2=r2(ŷ, y),
                mape=mape(ŷ, y),
            )
            save_model(joinpath(models_dir, "$(ds)_h$(H)_LSTM.bson"), :LSTM_manual, lstm_art, nothing, (
                dataset=ds, model="LSTM", seq_len=seq_len, horizon=H, features=feat_cols,
                targets=feat_cols, scaler=scaler, target_stats=target_stats, split=(train=0.6, val=0.2, test=0.2),
                sizes=(train=size(Xtr, 3), val=size(Xval, 3), test=size(Xte, 3)),
                val_best=(epoch=lstm_art.best_epoch, mse=lstm_art.best_val_mse),
                test_metrics=lstm_test_metrics,
            ))
            @info "[LSTM $(ds) h=$(H)] test metrics" lstm_test_metrics

            # CNN manual forward layers
            @info "Training CNN on $(ds) horizon=$(H)"
            cnn_art = train_cnn!(input_dim, out_dim, Xtr, Ytr, Xval, Yval; epochs=epochs, batchsize=batchsize, lr=lr, device=device)
            # Predict on test and compute metrics (original scale)
            function predict_cnn(X)
                Xd = device.to(X)
                ŷ, _ = cnn_forward_batch(cnn_art.layers, cnn_art.ps, cnn_art.st, Xd)
                Array(ŷ)
            end
            ŷ_sc = predict_cnn(Xte)
            ŷ = inverse_targets(scaler, ŷ_sc)
            y = inverse_targets(scaler, Yte_sc)
            cnn_test_metrics = (
                mse=mse(ŷ, y),
                mae=mae(ŷ, y),
                rmse=rmse(ŷ, y),
                r2=r2(ŷ, y),
                mape=mape(ŷ, y),
            )
            save_model(joinpath(models_dir, "$(ds)_h$(H)_CNN.bson"), :CNN_manual, cnn_art, nothing, (
                dataset=ds, model="CNN1D", seq_len=seq_len, horizon=H, features=feat_cols,
                targets=feat_cols, scaler=scaler, target_stats=target_stats, split=(train=0.6, val=0.2, test=0.2),
                sizes=(train=size(Xtr, 3), val=size(Xval, 3), test=size(Xte, 3)),
                val_best=(epoch=cnn_art.best_epoch, mse=cnn_art.best_val_mse),
                test_metrics=cnn_test_metrics,
            ))
            @info "[CNN $(ds) h=$(H)] test metrics" cnn_test_metrics
        end
    end
    @info "All models saved under $(models_dir)"
end

main()
