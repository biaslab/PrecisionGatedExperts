#!/usr/bin/env julia

# Train a reconstruction-only CNN autoencoder / variational autoencoder on sliding windows.

using ADTypes
using JLD2
using Lux
using MLUtils
using Optimisers
using Printf
using Random
using Reactant
using ProbabilisticEnsembling

const lossfn = MSELoss()

struct LazyReconstructionDataset
    X::Matrix{Float32}
    starts::Vector{Int}
    scaler::StandardScaler
    seq_len::Int
end

MLUtils.numobs(ds::LazyReconstructionDataset) = length(ds.starts)

function MLUtils.getobs(ds::LazyReconstructionDataset, i::Int)
    s = ds.starts[i]
    e = s + ds.seq_len - 1
    μx = reshape(ds.scaler.μ, :, 1)
    σx = reshape(ds.scaler.σ, :, 1)
    x = @views (ds.X[:, s:e] .- μx) ./ σx
    x = Float32.(x)
    return x, x
end

function MLUtils.getobs(ds::LazyReconstructionDataset, idxs::AbstractVector{<:Integer})
    f = size(ds.X, 1)
    b = length(idxs)
    Xb = Array{Float32}(undef, f, ds.seq_len, b)
    μ = ds.scaler.μ
    σ = ds.scaler.σ
    μx = reshape(μ, :, 1)
    σx = reshape(σ, :, 1)

    @inbounds for j in 1:b
        s = ds.starts[idxs[j]]
        e = s + ds.seq_len - 1
        @views Xb[:, :, j] = (ds.X[:, s:e] .- μx) ./ σx
    end
    return Xb, Xb
end

function split_window_starts(T::Int, seq_len::Int; ratios = (0.6, 0.2, 0.2))
    N = T - seq_len + 1
    N >= 3 || error("Time series too short for seq_len=$(seq_len).")

    r1, r2, r3 = ratios
    @assert abs(r1 + r2 + r3 - 1.0) < 1e-6 "Ratios must sum to 1.0"

    n_tr = max(round(Int, N * r1), 1)
    n_va = max(round(Int, N * r2), 1)
    n_te = max(N - n_tr - n_va, 1)
    @assert n_tr + n_va + n_te <= N "Invalid split sizes for N=$(N), ratios=$(ratios)"

    train_starts = collect(1:n_tr)
    val_starts = collect(n_tr+1:n_tr+n_va)
    test_starts = collect(n_tr+n_va+1:N)
    return train_starts, val_starts, test_starts
end

function fit_scaler_from_starts(X::Matrix{Float32}, starts::Vector{Int}, seq_len::Int)
    f = size(X, 1)
    total = length(starts) * seq_len
    μ = zeros(Float32, f)

    for s in starts
        for off in 0:seq_len-1
            @inbounds @views μ .+= X[:, s+off]
        end
    end
    μ ./= Float32(total)

    ss = zeros(Float32, f)
    for s in starts
        for off in 0:seq_len-1
            @inbounds @views begin
                d = X[:, s+off] .- μ
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
)
    f = size(X, 1)
    n = length(starts)
    X3 = Array{Float32}(undef, f, seq_len, n)
    μx = reshape(scaler.μ, :, 1)
    σx = reshape(scaler.σ, :, 1)

    @inbounds for i in 1:n
        s = starts[i]
        e = s + seq_len - 1
        @views X3[:, :, i] = (X[:, s:e] .- μx) ./ σx
    end
    return X3
end

function create_dataloaders(
    X::Matrix{Float32},
    train_starts::Vector{Int},
    val_starts::Vector{Int},
    scaler::StandardScaler;
    seq_len::Int,
    batchsize::Int = 128,
    dev = reactant_device(),
)
    train_ds = LazyReconstructionDataset(X, train_starts, scaler, seq_len)
    val_ds = LazyReconstructionDataset(X, val_starts, scaler, seq_len)
    train_loader = DataLoader(train_ds; batchsize = batchsize, shuffle = true, partial = false) |> dev
    val_loader = DataLoader(val_ds; batchsize = batchsize, shuffle = false, partial = false) |> dev
    return train_loader, val_loader
end

function inverse_inputs(s::StandardScaler, X3::AbstractArray{<:Real,3})
    μ = reshape(s.μ, :, 1, 1)
    σ = reshape(s.σ, :, 1, 1)
    return X3 .* σ .+ μ
end

function reconstruction_objective(model, ps, st, data)
    x, y = data
    ŷ, st_ = model(x, ps, st)
    loss = lossfn(ŷ, y)
    return loss, st_, (; recon_loss = loss, kl_loss = 0.0f0)
end

function vae_objective(model, ps, st, data)
    x, y, beta = data
    ŷ, μ, logvar, st_ = vae_forward(model, x, ps, st; sample = false)
    recon = lossfn(ŷ, y)
    kl = vae_kl_loss(μ, logvar)
    loss = recon + beta * kl
    return loss, st_, (; recon_loss = recon, kl_loss = kl)
end

function train_reconstruction_model(
    model,
    train_loader,
    val_loader;
    model_kind::Symbol = :ae,
    vae_beta::Float32 = 1.0f-3,
    epochs::Int = 25,
    lr::Float32 = 0.001f0,
    patience::Int = 5,
    dev = reactant_device(),
)
    cdev = cpu_device()
    ps, st = Lux.setup(Random.default_rng(), model) |> dev
    train_state = Lux.Training.TrainState(model, ps, st, AdamW(lr))

    sample_x = first(train_loader)[1]
    model_compiled = if dev isa ReactantDevice && model_kind != :vae
        @compile model(sample_x, ps, Lux.testmode(st))
    else
        model
    end
    vae_infer_compiled = if dev isa ReactantDevice && model_kind == :vae
        vae_infer(x, ps_, st_) = vae_forward(model, x, ps_, st_; sample = false)
        @compile vae_infer(sample_x, ps, Lux.testmode(st))
    else
        nothing
    end

    ad = dev isa ReactantDevice ? AutoEnzyme() : AutoZygote()
    best_val_loss = Inf32
    best_epoch = 0
    best_ps = cdev(ps)
    best_st = cdev(st)
    patience_counter = 0

    for epoch in 1:epochs
        GC.gc()
        total_loss = 0.0f0
        total_recon = 0.0f0
        total_kl = 0.0f0
        total_samples = 0

        for (x, y) in train_loader
            data = model_kind == :vae ? (x, y, vae_beta) : (x, y)
            objective = model_kind == :vae ? vae_objective : reconstruction_objective
            (_, loss, stats, train_state) =
                Lux.Training.single_train_step!(ad, objective, data, train_state)
            bsz = size(y, 3)
            total_loss += loss * bsz
            total_recon += stats.recon_loss * bsz
            total_kl += stats.kl_loss * bsz
            total_samples += bsz
        end
        train_loss = total_loss / total_samples
        train_recon = total_recon / total_samples
        train_kl = total_kl / total_samples

        val_loss = 0.0f0
        val_recon = 0.0f0
        val_kl = 0.0f0
        val_samples = 0
        st_ = Lux.testmode(train_state.states)
        for (x, y) in val_loader
            bsz = size(y, 3)
            if model_kind == :vae
                if dev isa ReactantDevice
                    ŷ, μ, logvar, st_ = vae_infer_compiled(x, train_state.parameters, st_)
                else
                    ŷ, μ, logvar, st_ =
                        vae_forward(model, x, train_state.parameters, st_; sample = false)
                end
                recon = lossfn(ŷ, y)
                kl = vae_kl_loss(μ, logvar)
                val_loss += (recon + vae_beta * kl) * bsz
                val_recon += recon * bsz
                val_kl += kl * bsz
            else
                ŷ, st_ = model_compiled(x, train_state.parameters, st_)
                recon = lossfn(ŷ, y)
                val_loss += recon * bsz
                val_recon += recon * bsz
            end
            val_samples += bsz
        end
        val_loss /= val_samples
        val_recon /= val_samples
        val_kl /= val_samples

        if model_kind == :vae
            @printf(
                "Epoch [%3d]: Train Loss %.6f (Recon %.6f KL %.6f) | Val Loss %.6f (Recon %.6f KL %.6f)\n",
                epoch,
                train_loss,
                train_recon,
                train_kl,
                val_loss,
                val_recon,
                val_kl,
            )
        else
            @printf("Epoch [%3d]: Train Loss %.6f | Val Loss %.6f\n", epoch, train_loss, val_loss)
        end

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
        parameters = best_ps,
        states = best_st,
        best_epoch = best_epoch,
        best_val_mse = best_val_loss,
    )
end

function compute_reconstruction_metrics(model, ps, st, Xte_sc, scaler; dev = reactant_device())
    cdev = cpu_device()
    Xte_d = dev(Float32.(Xte_sc))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)

    model_compiled = if dev isa ReactantDevice && !(model isa TimeSeriesVAE)
        @compile model(Xte_d, ps_d, st_test)
    else
        model
    end
    vae_infer_compiled = if dev isa ReactantDevice && (model isa TimeSeriesVAE)
        vae_infer(x, ps_, st_) = vae_forward(model, x, ps_, st_; sample = false)
        @compile vae_infer(Xte_d, ps_d, st_test)
    else
        nothing
    end

    ŷ_sc = if model isa TimeSeriesVAE
        if dev isa ReactantDevice
            y, _, _, _ = vae_infer_compiled(Xte_d, ps_d, st_test)
            y
        else
            y, _, _, _ = vae_forward(model, Xte_d, ps_d, st_test; sample = false)
            y
        end
    else
        y, _ = model_compiled(Xte_d, ps_d, st_test)
        y
    end
    ŷ_sc = cdev(ŷ_sc)

    ŷ = inverse_inputs(scaler, Array(ŷ_sc))
    y = inverse_inputs(scaler, Array(Xte_sc))
    return (
        mse = mse(ŷ, y),
        mae = mae(ŷ, y),
        rmse = rmse(ŷ, y),
    )
end

function main()
    data_dir = joinpath( "data")
    models_dir = joinpath("models")
    mkpath(models_dir)

    datasets = let ds = get(ENV, "DATASETS", "ETTh2, ETTh1, exchange_rate")
        split(ds, [',', ';', ' ']; keepempty = false)
    end

    seq_len = parse(Int, get(ENV, "SEQ_LEN", "96"))
    epochs = parse(Int, get(ENV, "EPOCHS", "50"))
    batchsize = parse(Int, get(ENV, "BATCHSIZE", "128"))
    lr = parse(Float32, get(ENV, "LR", "0.001"))
    patience = parse(Int, get(ENV, "PATIENCE", "20"))

    ae_channels = parse(Int, get(ENV, "AE_CHANNELS", "32"))
    ae_hidden = parse(Int, get(ENV, "AE_HIDDEN", "64"))
    ae_latent = parse(Int, get(ENV, "AE_LATENT", "32"))
    ae_kernel = parse(Int, get(ENV, "AE_KERNEL", "5"))
    model_kind = Symbol(lowercase(get(ENV, "MODEL_KIND", "vae")))
    model_kind in (:ae, :vae) || error("MODEL_KIND must be one of: ae, vae")
    vae_beta = parse(Float32, get(ENV, "VAE_BETA", "0.001"))

    device_kind = lowercase(get(ENV, "DEVICE", "reactant"))
    dev = device_kind == "cpu" ? cpu_device() : reactant_device()
    @info "Using device: $(typeof(dev))"

    for ds in datasets
        ds_path = joinpath(data_dir, ds)
        @info "Loading dataset" dataset = ds_path

        ratio_ds = occursin("ETTh", ds) ? (0.6, 0.2, 0.2) : (0.7, 0.1, 0.2)
        Xmat, feat_cols = load_ett(ds_path)
        Xmat = Float32.(Xmat)

        train_starts, val_starts, test_starts = split_window_starts(
            size(Xmat, 2),
            seq_len;
            ratios = ratio_ds,
        )
        scaler = fit_scaler_from_starts(Xmat, train_starts, seq_len)

        train_loader, val_loader = create_dataloaders(
            Xmat,
            train_starts,
            val_starts,
            scaler;
            seq_len = seq_len,
            batchsize = batchsize,
            dev = dev,
        )

        input_dim = size(Xmat, 1)
        model = if model_kind == :vae
            TimeSeriesVAE(
                input_dim,
                seq_len;
                channels = ae_channels,
                hidden_dims = ae_hidden,
                latent_dim = ae_latent,
                kernel_size = ae_kernel,
            )
        else
            TimeSeriesAutoEncoder(
                input_dim,
                seq_len;
                channels = ae_channels,
                hidden_dims = ae_hidden,
                latent_dim = ae_latent,
                kernel_size = ae_kernel,
            )
        end

        @info "Training reconstruction model" dataset = ds model_kind = model_kind seq_len = seq_len channels = ae_channels hidden = ae_hidden latent = ae_latent kernel = ae_kernel vae_beta = vae_beta
        result = train_reconstruction_model(
            model,
            train_loader,
            val_loader;
            model_kind = model_kind,
            vae_beta = vae_beta,
            epochs = epochs,
            lr = lr,
            patience = patience,
            dev = dev,
        )

        Xte_sc = materialize_scaled_windows(Xmat, test_starts, scaler; seq_len = seq_len)
        test_metrics = compute_reconstruction_metrics(
            model,
            result.parameters,
            result.states,
            Xte_sc,
            scaler;
            dev = dev,
        )
        @info "Reconstruction test metrics" dataset = ds model_kind = model_kind test_metrics...

        model_name = model_kind == :vae ? "TimeSeriesVAE" : "TimeSeriesAutoEncoder"
        model_type = model_kind == :vae ? :TimeSeriesVAE : :TimeSeriesAutoEncoder
        model_file_tag = model_kind == :vae ? "VAE" : "AutoEncoder"
        model_path = joinpath(models_dir, "$(ds)_s$(seq_len)_$(model_file_tag)_enzyme.jld2")
        jldsave(
            model_path;
            model_type = model_type,
            parameters = result.parameters,
            states = result.states,
            config = (
                input_dim = input_dim,
                seq_len = seq_len,
                channels = ae_channels,
                hidden_dim = ae_hidden,
                latent_dim = ae_latent,
                kernel_size = ae_kernel,
                model_kind = model_kind,
                vae_beta = vae_beta,
            ),
            meta = (
                dataset = ds,
                model = model_name,
                task = "reconstruction",
                seq_len = seq_len,
                features = feat_cols,
                scaler = scaler,
                split = (train = ratio_ds[1], val = ratio_ds[2], test = ratio_ds[3]),
                sizes = (
                    train = length(train_starts),
                    val = length(val_starts),
                    test = length(test_starts),
                ),
                val_best = (epoch = result.best_epoch, mse = result.best_val_mse),
                test_metrics = test_metrics,
            ),
        )
        @info "Reconstruction model saved" path = model_path model_kind = model_kind
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
