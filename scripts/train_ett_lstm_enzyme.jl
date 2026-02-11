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
        # Training
        total_loss = 0.0f0
        total_samples = 0

        for (x, y) in train_loader
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

function create_dataloaders(Xtr, Ytr, Xval, Yval; batchsize::Int=128, dev=reactant_device())
    # Xtr shape: (features, seq_len, n_samples)
    # Ytr shape: (out_features, n_samples)

    train_loader = DataLoader(
        (Float32.(Xtr), Float32.(Ytr));
        batchsize=batchsize,
        shuffle=true,
        partial=false
    ) |> dev

    val_loader = DataLoader(
        (Float32.(Xval), Float32.(Yval));
        batchsize=batchsize,
        shuffle=false,
        partial=false
    ) |> dev

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
    data_dir = joinpath(@__DIR__, "..", "data")
    models_dir = joinpath(@__DIR__, "..", "models")
    mkpath(models_dir)

    datasets = ["traffic"]
    seq_len = parse(Int, get(ENV, "SEQ_LEN", "96"))
    horizons = let hs = get(ENV, "HORIZONS", "")
        h = get(ENV, "HORIZON", "")
        if !isempty(hs)
            parse.(Int, split(hs, [',', ';', ' ']; keepempty=false))
        elseif !isempty(h)
            [parse(Int, h)]
        else
            [720]
        end
    end

    epochs = parse(Int, get(ENV, "EPOCHS", "50"))
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
        @info "Loaded dataset" n_samples = size(Xmat, 1) n_features = length(feat_cols)

        for H in horizons
            @info "Training LSTM" dataset = ds horizon = H seq_len = seq_len ratio = ratio_ds

            # Build sequences
            X3, Y2 = make_sequences(Xmat; seq_len=seq_len, horizon=H)
            Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=ratio_ds)

            # Scale data
            scaler = fit_scaler(Xtr)
            Xtr = scale_inputs(scaler, Xtr)
            Xval = scale_inputs(scaler, Xval)
            Xte = scale_inputs(scaler, Xte)
            Ytr = scale_targets(scaler, Ytr)
            Yval = scale_targets(scaler, Yval)
            Yte_sc = scale_targets(scaler, Yte)

            input_dim = size(Xtr, 1)
            out_dim = size(Ytr, 1)

            @info "Data shapes" input_dim = input_dim out_dim = out_dim train = size(Xtr, 3) val = size(Xval, 3) test = size(Xte, 3)

            # Create model and dataloaders
            model = TimeSeriesLSTM(input_dim, hidden_dim, out_dim)
            train_loader, val_loader = create_dataloaders(Xtr, Ytr, Xval, Yval; batchsize=batchsize, dev=dev)

            # Train
            result = train_lstm(
                model, train_loader, val_loader;
                epochs=epochs,
                lr=lr,
                patience=patience,
                dev=dev
            )

            # Compute test metrics
            test_metrics = compute_test_metrics(model, result.parameters, result.states, Xte, Yte_sc, scaler; dev=dev)
            @info "Test metrics" dataset = ds horizon = H test_metrics...

            # Save model
            model_path = joinpath(models_dir, "$(ds)_h$(H)_LSTM_enzyme.jld2")
            jldsave(model_path;
                model_type=:TimeSeriesLSTM,
                parameters=result.parameters,
                states=result.states,
                config=(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim
                ),
                meta=(
                    dataset=ds,
                    model="LSTM",
                    seq_len=seq_len,
                    horizon=H,
                    features=feat_cols,
                    targets=feat_cols,
                    scaler=scaler,
                    split=(train=0.6, val=0.2, test=0.2),
                    sizes=(train=size(Xtr, 3), val=size(Xval, 3), test=size(Xte, 3)),
                    val_best=(epoch=result.best_epoch, mse=result.best_val_mse),
                    test_metrics=test_metrics
                )
            )
            @info "Model saved" path = model_path

            @info "Training CNN" dataset = ds horizon = H seq_len = seq_len

            cnn_model = TimeSeriesCNN(input_dim, out_dim; channels=cnn_channels, k=cnn_kernel, stride=cnn_stride)
            cnn_train_loader, cnn_val_loader = create_dataloaders(Xtr, Ytr, Xval, Yval; batchsize=batchsize, dev=dev)

            cnn_result = train_lstm(
                cnn_model, cnn_train_loader, cnn_val_loader;
                epochs=epochs,
                lr=lr,
                patience=patience,
                dev=dev
            )

            cnn_test_metrics = compute_test_metrics(cnn_model, cnn_result.parameters, cnn_result.states, Xte, Yte_sc, scaler; dev=dev)
            @info "CNN test metrics" dataset = ds horizon = H cnn_test_metrics...

            cnn_model_path = joinpath(models_dir, "$(ds)_h$(H)_CNN_enzyme.jld2")
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
                    split=(train=0.6, val=0.2, test=0.2),
                    sizes=(train=size(Xtr, 3), val=size(Xval, 3), test=size(Xte, 3)),
                    val_best=(epoch=cnn_result.best_epoch, mse=cnn_result.best_val_mse),
                    test_metrics=cnn_test_metrics
                )
            )
            @info "CNN model saved" path = cnn_model_path

            @info "Training MLP" dataset = ds horizon = H seq_len = seq_len

            mlp_model = TimeSeriesMLP(input_dim, seq_len, out_dim; hidden_dims=mlp_hidden, depth=mlp_depth)
            mlp_train_loader, mlp_val_loader = create_dataloaders(Xtr, Ytr, Xval, Yval; batchsize=batchsize, dev=dev)

            mlp_result = train_lstm(
                mlp_model, mlp_train_loader, mlp_val_loader;
                epochs=epochs,
                lr=lr,
                patience=patience,
                dev=dev
            )

            mlp_test_metrics = compute_test_metrics(mlp_model, mlp_result.parameters, mlp_result.states, Xte, Yte_sc, scaler; dev=dev)
            @info "MLP test metrics" dataset = ds horizon = H mlp_test_metrics...

            mlp_model_path = joinpath(models_dir, "$(ds)_h$(H)_MLP_enzyme.jld2")
            jldsave(mlp_model_path;
                model_type=:TimeSeriesMLP,
                parameters=mlp_result.parameters,
                states=mlp_result.states,
                config=(
                    input_dim=input_dim,
                    seq_len=seq_len,
                    hidden_dim=mlp_hidden,
                    depth=mlp_depth,
                    out_dim=out_dim
                ),
                meta=(
                    dataset=ds,
                    model="MLP",
                    seq_len=seq_len,
                    horizon=H,
                    features=feat_cols,
                    targets=feat_cols,
                    scaler=scaler,
                    split=(train=0.6, val=0.2, test=0.2),
                    sizes=(train=size(Xtr, 3), val=size(Xval, 3), test=size(Xte, 3)),
                    val_best=(epoch=mlp_result.best_epoch, mse=mlp_result.best_val_mse),
                    test_metrics=mlp_test_metrics
                )
            )
            @info "MLP model saved" path = mlp_model_path
        end
    end

    @info "All models saved" directory = models_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
