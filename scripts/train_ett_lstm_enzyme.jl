#!/usr/bin/env julia

# Train LSTM with Lux.jl + Enzyme/Reactant on ETTh1/ETTh2 and save models.

using ADTypes
using Lux
using JLD2
using MLUtils
using Optimisers
using Printf
using Reactant
using Random
using Statistics
using ProbabilisticEnsembling

# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------

const lossfn = MSELoss()

function compute_loss(model, ps, st, (x, y))
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (; y_pred=ŷ)
end

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

function train_lstm(
    model, train_loader, val_loader;
    epochs::Int=25,
    lr::Float32=0.001f0,
    patience::Int=5,
    dev=reactant_device()
)
    cdev = cpu_device()

    ps, st = Lux.setup(Random.default_rng(), model) |> dev
    train_state = Lux.Training.TrainState(model, ps, st, Adam(lr))

    # Compile model for inference
    model_compiled = if dev isa ReactantDevice
        sample_x = first(train_loader)[1]
        @compile model(sample_x, ps, Lux.testmode(st))
    else
        model
    end

    ad = dev isa ReactantDevice ? AutoEnzyme() : AutoZygote()

    best_val_loss = Inf32
    best_epoch = 0
    best_ps = cdev(ps)
    best_st = cdev(st)
    patience_counter = 0

    for epoch in 1:epochs
        GC.gc()
        # Training
        total_loss = 0.0f0
        total_samples = 0

        for (x, y) in train_loader
            GC.gc()
            (_, loss, _, train_state) = Lux.Training.single_train_step!(
                ad, lossfn, (x, y), train_state
            )
            batch_size = size(y, 2)
            total_loss += loss * batch_size
            total_samples += batch_size
        end

        train_loss = total_loss / total_samples

        # Validation
        val_loss = 0.0f0
        val_samples = 0
        st_ = Lux.testmode(train_state.states)

        for (x, y) in val_loader
            ŷ, st_ = model_compiled(x, train_state.parameters, st_)
            batch_size = size(y, 2)
            val_loss += lossfn(ŷ, y) * batch_size
            val_samples += batch_size
        end

        val_loss = val_loss / val_samples

        @printf("Epoch [%3d]: Train Loss %.6f | Val Loss %.6f\n", epoch, train_loss, val_loss)

        # Early stopping check
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            best_ps = cdev(train_state.parameters)
            best_st = cdev(train_state.states)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= patience
                @printf("Early stopping at epoch %d. Best epoch: %d\n", epoch, best_epoch)
                break
            end
        end
    end

    return (
        parameters=best_ps,
        states=best_st,
        best_epoch=best_epoch,
        best_val_mse=best_val_loss
    )
end

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

struct LazyWindowDataset
    X::Matrix{Float32}
    starts::Vector{Int}
    scaler::StandardScaler
    seq_len::Int
    horizon::Int
end

MLUtils.numobs(ds::LazyWindowDataset) = length(ds.starts)

function MLUtils.getobs(ds::LazyWindowDataset, i::Int)
    s = ds.starts[i]
    e = s + ds.seq_len - 1
    t = e + ds.horizon

    μx = reshape(ds.scaler.μ, :, 1)
    σx = reshape(ds.scaler.σ, :, 1)

    x = @views (ds.X[:, s:e] .- μx) ./ σx
    y = @views (ds.X[:, t] .- ds.scaler.μ) ./ ds.scaler.σ
    return Float32.(x), Float32.(y)
end

function MLUtils.getobs(ds::LazyWindowDataset, idxs::AbstractVector{<:Integer})
    f = size(ds.X, 1)
    b = length(idxs)
    Xb = Array{Float32}(undef, f, ds.seq_len, b)
    Yb = Array{Float32}(undef, f, b)

    μ = ds.scaler.μ
    σ = ds.scaler.σ

    @inbounds for j in 1:b
        s = ds.starts[idxs[j]]
        e = s + ds.seq_len - 1
        t = e + ds.horizon
        @views Xb[:, :, j] = (ds.X[:, s:e] .- reshape(μ, :, 1)) ./ reshape(σ, :, 1)
        @views Yb[:, j] = (ds.X[:, t] .- μ) ./ σ
    end
    return Xb, Yb
end

function split_window_starts(T::Int, seq_len::Int, horizon::Int; ratios=(0.6, 0.2, 0.2))
    N = T - seq_len - horizon + 1
    N >= 3 || error("Time series too short for seq_len=$(seq_len), horizon=$(horizon).")

    r1, r2, r3 = ratios
    @assert abs(r1 + r2 + r3 - 1.0) < 1e-6 "Ratios must sum to 1.0"

    # Keep exactly the same counting logic as train_val_test_split in src/utils.jl
    n_tr = round(Int, N * r1)
    n_va = round(Int, N * r2)
    n_te = N - n_tr - n_va
    n_tr = max(n_tr, 1)
    n_va = max(n_va, 1)
    n_te = max(n_te, 1)
    @assert n_tr + n_va + n_te <= N "Invalid split sizes for N=$(N), ratios=$(ratios)"

    train_starts = collect(1:n_tr)
    val_starts = collect(n_tr+1:n_tr+n_va)
    test_starts = collect(n_tr+n_va+1:N)
    return train_starts, val_starts, test_starts
end

function compare_old_vs_current_split_counts(
    T::Int,
    seq_len::Int,
    horizon::Int,
    train_starts::Vector{Int},
    val_starts::Vector{Int},
    test_starts::Vector{Int};
    ratios=(0.6, 0.2, 0.2)
)
    # Old path count logic:
    # N = size(make_sequences(X), 3) == T - seq_len - horizon + 1
    # then train_val_test_split uses round/max as below.
    N = T - seq_len - horizon + 1
    r1, r2, r3 = ratios
    @assert abs(r1 + r2 + r3 - 1.0) < 1e-6 "Ratios must sum to 1.0"
    old_train = max(round(Int, N * r1), 1)
    old_val = max(round(Int, N * r2), 1)
    old_test = max(N - old_train - old_val, 1)
    @assert old_train + old_val + old_test <= N "Old split logic is invalid for this configuration."

    new_train = length(train_starts)
    new_val = length(val_starts)
    new_test = length(test_starts)

    same = (old_train == new_train) && (old_val == new_val) && (old_test == new_test)
    return (
        same=same,
        old=(train=old_train, val=old_val, test=old_test),
        new=(train=new_train, val=new_val, test=new_test),
        total_windows=N
    )
end

function fit_scaler_from_starts(X::Matrix{Float32}, starts::Vector{Int}, seq_len::Int)
    f = size(X, 1)
    total = length(starts) * seq_len
    μ = zeros(Float32, f)

    # Accumulate in the same column-wise traversal order as reshape(Xtr, f, :).
    for s in starts
        for off in 0:seq_len-1
            t = s + off
            @inbounds @views μ .+= X[:, t]
        end
    end
    μ ./= Float32(total)

    # Unbiased variance (corrected=true), then add epsilon exactly like old scaler.
    ss = zeros(Float32, f)
    for s in starts
        for off in 0:seq_len-1
            t = s + off
            @inbounds @views begin
                d = X[:, t] .- μ
                ss .+= d .* d
            end
        end
    end
    σ = sqrt.(ss ./ Float32(max(total - 1, 1))) .+ 1.0f-6

    return StandardScaler(μ, σ)
end

function materialize_scaled_windows(
    X::Matrix{Float32},
    starts::Vector{Int},
    scaler::StandardScaler;
    seq_len::Int,
    horizon::Int
)
    f = size(X, 1)
    n = length(starts)
    X3 = Array{Float32}(undef, f, seq_len, n)
    Y2 = Array{Float32}(undef, f, n)
    μ = scaler.μ
    σ = scaler.σ

    @inbounds for i in 1:n
        s = starts[i]
        e = s + seq_len - 1
        t = e + horizon
        @views X3[:, :, i] = (X[:, s:e] .- reshape(μ, :, 1)) ./ reshape(σ, :, 1)
        @views Y2[:, i] = (X[:, t] .- μ) ./ σ
    end

    return X3, Y2
end

function create_dataloaders(
    X::Matrix{Float32},
    train_starts::Vector{Int},
    val_starts::Vector{Int},
    scaler::StandardScaler;
    seq_len::Int,
    horizon::Int,
    batchsize::Int=128,
    dev=reactant_device()
)
    train_ds = LazyWindowDataset(X, train_starts, scaler, seq_len, horizon)
    val_ds = LazyWindowDataset(X, val_starts, scaler, seq_len, horizon)

    train_loader = DataLoader(train_ds; batchsize=batchsize, shuffle=true, partial=false) |> dev
    val_loader = DataLoader(val_ds; batchsize=batchsize, shuffle=false, partial=false) |> dev
    return train_loader, val_loader
end

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

function compute_test_metrics(model, ps, st, Xte, Yte, scaler; dev=reactant_device())
    cdev = cpu_device()

    Xte_d = dev(Float32.(Xte))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)

    # Compile for test inference
    model_compiled = if dev isa ReactantDevice
        @compile model(Xte_d, ps_d, st_test)
    else
        model
    end

    ŷ_sc, _ = model_compiled(Xte_d, ps_d, st_test)
    ŷ_sc = cdev(ŷ_sc)

    # Inverse transform to original scale
    ŷ = inverse_targets(scaler, Array(ŷ_sc))
    y = inverse_targets(scaler, Array(Yte))

    return (
        mse=mse(ŷ, y),
        mae=mae(ŷ, y),
        rmse=rmse(ŷ, y),
        r2=r2(ŷ, y),
        mape=mape(ŷ, y)
    )
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    # Settings
    data_dir = joinpath("..", "data")
    models_dir = joinpath("..", "models")
    mkpath(models_dir)

    datasets = ["electricity"]
    seq_len = parse(Int, get(ENV, "SEQ_LEN", "96"))
    horizons = let hs = get(ENV, "HORIZONS", "")
        h = get(ENV, "HORIZON", "")
        if !isempty(hs)
            parse.(Int, split(hs, [',', ';', ' ']; keepempty=false))
        elseif !isempty(h)
            [parse(Int, h)]
        else
            [192]
        end
    end

    epochs = parse(Int, get(ENV, "EPOCHS", "10"))
    batchsize = parse(Int, get(ENV, "BATCHSIZE", "128"))
    lr = parse(Float32, get(ENV, "LR", "0.001"))
    hidden_dim = parse(Int, get(ENV, "HIDDEN", "64"))
    mlp_hidden = parse(Int, get(ENV, "MLP_HIDDEN", string(hidden_dim)))
    mlp_depth = parse(Int, get(ENV, "MLP_DEPTH", "2"))
    cnn_channels = parse(Int, get(ENV, "CNN_CHANNELS", "64"))
    cnn_kernel = parse(Int, get(ENV, "CNN_KERNEL", "7"))
    cnn_stride = parse(Int, get(ENV, "CNN_STRIDE", "2"))
    patience = parse(Int, get(ENV, "PATIENCE", "50"))

    dev = reactant_device()
    cdev = cpu_device()
    @info "Using device: $(typeof(dev))"

    for ds in datasets
        ds_path = joinpath(data_dir, ds)
        @info "Loading dataset" dataset = ds_path

        if occursin("ETTh", ds)
            ratio_ds = (0.6, 0.2, 0.2)
        else
            ratio_ds = (0.7, 0.1, 0.2)
        end

        Xmat, feat_cols = load_ett(ds_path)
        Xmat = Float32.(Xmat)
        @info "Loaded dataset" n_timesteps = size(Xmat, 2) n_features = length(feat_cols)

        for H in horizons
            seq_len = H
            @info "Training LSTM" dataset = ds horizon = H seq_len = seq_len ratio = ratio_ds

            train_starts, val_starts, test_starts = split_window_starts(
                size(Xmat, 2), seq_len, H; ratios=ratio_ds
            )
            split_check = compare_old_vs_current_split_counts(
                size(Xmat, 2), seq_len, H, train_starts, val_starts, test_starts; ratios=ratio_ds
            )
            @info "Split count check" dataset = ds horizon = H total_windows = split_check.total_windows old = split_check.old new = split_check.new same = split_check.same
            split_check.same || error("Split mismatch detected for $(ds), horizon=$(H). old=$(split_check.old), new=$(split_check.new)")

            scaler = fit_scaler_from_starts(Xmat, train_starts, seq_len)
            Xte, Yte_sc = materialize_scaled_windows(
                Xmat, test_starts, scaler; seq_len=seq_len, horizon=H
            )

            input_dim = size(Xmat, 1)
            out_dim = input_dim

            @info "Data shapes" input_dim = input_dim out_dim = out_dim train = length(train_starts) val = length(val_starts) test = length(test_starts)

            @info "Training LSTMOneStep" dataset = ds horizon = H seq_len = seq_len n_steps = 8

            lstm8_model = TimeSeriesLSTMOneStep(input_dim, seq_len, hidden_dim, out_dim; n_steps=8)
            lstm8_train_loader, lstm8_val_loader = create_dataloaders(
                Xmat, train_starts, val_starts, scaler;
                seq_len=seq_len, horizon=H, batchsize=batchsize, dev=dev
            )

            lstm8_result = train_lstm(
                lstm8_model, lstm8_train_loader, lstm8_val_loader;
                epochs=epochs,
                lr=lr,
                patience=patience,
                dev=dev
            )

            lstm8_test_metrics = compute_test_metrics(
                lstm8_model, lstm8_result.parameters, lstm8_result.states, Xte, Yte_sc, scaler; dev=dev
            )
            @info "LSTMOneStep test metrics" dataset = ds horizon = H lstm8_test_metrics...

            lstm8_model_path = joinpath(models_dir, "$(ds)_h$(H)_s$(seq_len)_LSTMOneStep8_enzyme.jld2")
            jldsave(lstm8_model_path;
                model_type=:TimeSeriesLSTMOneStep,
                parameters=lstm8_result.parameters,
                states=lstm8_result.states,
                config=(
                    input_dim=input_dim,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    n_steps=8
                ),
                meta=(
                    dataset=ds,
                    model="LSTMOneStep",
                    seq_len=seq_len,
                    horizon=H,
                    features=feat_cols,
                    targets=feat_cols,
                    scaler=scaler,
                    split=(train=ratio_ds[1], val=ratio_ds[2], test=ratio_ds[3]),
                    sizes=(train=length(train_starts), val=length(val_starts), test=length(test_starts)),
                    val_best=(epoch=lstm8_result.best_epoch, mse=lstm8_result.best_val_mse),
                    test_metrics=lstm8_test_metrics
                )
            )
            @info "LSTMOneStep model saved" path = lstm8_model_path

            GC.gc()

            @info "Training CNN" dataset = ds horizon = H seq_len = seq_len

            cnn_model = TimeSeriesCNN(input_dim, out_dim; channels=cnn_channels, k=cnn_kernel, stride=cnn_stride)
            cnn_train_loader, cnn_val_loader = create_dataloaders(
                Xmat, train_starts, val_starts, scaler;
                seq_len=seq_len, horizon=H, batchsize=batchsize, dev=dev
            )

            cnn_result = train_lstm(
                cnn_model, cnn_train_loader, cnn_val_loader;
                epochs=epochs,
                lr=lr,
                patience=patience,
                dev=dev
            )

            cnn_test_metrics = compute_test_metrics(cnn_model, cnn_result.parameters, cnn_result.states, Xte, Yte_sc, scaler; dev=dev)
            @info "CNN test metrics" dataset = ds horizon = H cnn_test_metrics...

            cnn_model_path = joinpath(models_dir, "$(ds)_h$(H)_s$(seq_len)_CNN_enzyme.jld2")
            jldsave(cnn_model_path;
                model_type=:TimeSeriesCNN,
                parameters=cnn_result.parameters,
                states=cnn_result.states,
                config=(
                    input_dim=input_dim,
                    channels=cnn_channels,
                    kernel=cnn_kernel,
                    stride=cnn_stride,
                    out_dim=out_dim
                ),
                meta=(
                    dataset=ds,
                    model="CNN1D",
                    seq_len=seq_len,
                    horizon=H,
                    features=feat_cols,
                    targets=feat_cols,
                    scaler=scaler,
                    split=(train=ratio_ds[1], val=ratio_ds[2], test=ratio_ds[3]),
                    sizes=(train=length(train_starts), val=length(val_starts), test=length(test_starts)),
                    val_best=(epoch=cnn_result.best_epoch, mse=cnn_result.best_val_mse),
                    test_metrics=cnn_test_metrics
                )
            )
            @info "CNN model saved" path = cnn_model_path

            GC.gc()

            @info "Training NLinear" dataset = ds horizon = H seq_len = seq_len

            nlinear_model = TimeSeriesNLinear(input_dim, seq_len, out_dim)
            nlinear_train_loader, nlinear_val_loader = create_dataloaders(
                Xmat, train_starts, val_starts, scaler;
                seq_len=seq_len, horizon=H, batchsize=batchsize, dev=dev
            )

            nlinear_result = train_lstm(
                nlinear_model, nlinear_train_loader, nlinear_val_loader;
                epochs=epochs,
                lr=lr,
                patience=patience,
                dev=dev
            )

            nlinear_test_metrics = compute_test_metrics(
                nlinear_model, nlinear_result.parameters, nlinear_result.states, Xte, Yte_sc, scaler; dev=dev
            )
            @info "NLinear test metrics" dataset = ds horizon = H nlinear_test_metrics...

            nlinear_model_path = joinpath(models_dir, "$(ds)_h$(H)_s$(seq_len)_MLP_enzyme.jld2")
            jldsave(nlinear_model_path;
                model_type=:TimeSeriesNLinear,
                parameters=nlinear_result.parameters,
                states=nlinear_result.states,
                config=(
                    input_dim=input_dim,
                    seq_len=seq_len,
                    bias=true,
                    out_dim=out_dim
                ),
                meta=(
                    dataset=ds,
                    model="NLinear",
                    seq_len=seq_len,
                    horizon=H,
                    features=feat_cols,
                    targets=feat_cols,
                    scaler=scaler,
                    split=(train=ratio_ds[1], val=ratio_ds[2], test=ratio_ds[3]),
                    sizes=(train=length(train_starts), val=length(val_starts), test=length(test_starts)),
                    val_best=(epoch=nlinear_result.best_epoch, mse=nlinear_result.best_val_mse),
                    test_metrics=nlinear_test_metrics
                )
            )
            @info "NLinear model saved" path = nlinear_model_path
        end
    end

    @info "All models saved" directory = models_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
