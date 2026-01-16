using CSV, DataFrames
using Statistics, Random
using Lux

export StandardScaler, fit_scaler, scale_inputs, inverse_transform_target, scale_targets, inverse_targets
export find_dataset_csv, load_ett
export make_sequences
export train_val_test_split
export batches
export mse, mae, rmse, r2, mape, compute_metrics

struct StandardScaler
    μ::Vector{Float32}
    σ::Vector{Float32}
end

function fit_scaler(X::AbstractArray{<:Real,3})
    f, t, n = size(X)
    X2 = reshape(X, f, t*n)
    μ = vec(mean(X2; dims=2))
    σ = vec(std(X2; dims=2, corrected=true)) .+ 1f-6
    return StandardScaler(Float32.(μ), Float32.(σ))
end

function scale_inputs(s::StandardScaler, X)
    μ = reshape(s.μ, :, 1, 1)
    σ = reshape(s.σ, :, 1, 1)
    return (X .- μ) ./ σ
end

inverse_transform_target(μ::Float32, σ::Float32, yhat) = yhat .* σ .+ μ

scale_targets(s::StandardScaler, Y) = begin
    μ = reshape(s.μ, :, 1)
    σ = reshape(s.σ, :, 1)
    (Y .- μ) ./ σ
end

inverse_targets(s::StandardScaler, Y) = begin
    μ = reshape(s.μ, :, 1)
    σ = reshape(s.σ, :, 1)
    Y .* σ .+ μ
end

function find_dataset_csv(path::AbstractString)
    if endswith(lowercase(path), ".csv") && isfile(path)
        return path
    elseif isdir(path)
        csvs = filter(p -> endswith(lowercase(p), ".csv"), readdir(path; join=true))
        isempty(csvs) && error("No CSV in $(path)")
        return first(csvs)
    else
        isfile(path * ".csv") && return path * ".csv"
        error("Dataset not found at $(path)")
    end
end

function load_ett(path::AbstractString)
    csv_path = find_dataset_csv(path)
    df = CSV.read(csv_path, DataFrame)
    num_cols = [n for (n, c) in zip(names(df), eachcol(df)) if eltype(c) <: Real]
    isempty(num_cols) && error("No numeric columns in $(csv_path)")
    feat_cols = num_cols
    X = Matrix{Float32}(permutedims(Matrix(df[:, feat_cols]))) # (features, time)
    return X, feat_cols
end

function make_sequences(X::AbstractMatrix{<:Real}; seq_len::Int=96, horizon::Int=1)
    f, T = size(X)
    last_start = T - seq_len - horizon + 1
    last_start < 1 && error("Time series too short for seq_len=$(seq_len), horizon=$(horizon)")
    N = last_start
    X3 = Array{Float32}(undef, f, seq_len, N)
    Y2 = Array{Float32}(undef, f, N)
    @inbounds for i in 1:N
        s = i
        e = s + seq_len - 1
        @views X3[:, :, i] = Float32.(X[:, s:e])
        @views Y2[:, i] = Float32.(X[:, e + horizon])
    end
    return X3, Y2
end

function train_val_test_split(X, Y; ratios=(0.6, 0.2, 0.2))
    N = size(X, 3)
    r1, r2, r3 = ratios
    @assert abs(r1 + r2 + r3 - 1.0) < 1e-6 "Ratios must sum to 1.0"
    n_tr = round(Int, N * r1)
    n_va = round(Int, N * r2)
    n_te = N - n_tr - n_va
    n_tr = max(n_tr, 1)
    n_va = max(n_va, 1)
    n_te = max(n_te, 1)
    @assert n_tr + n_va + n_te <= N
    Xtr = X[:, :, 1:n_tr]
    Ytr = Y[:, 1:n_tr]
    Xva = X[:, :, n_tr+1:n_tr+n_va]
    Yva = Y[:, n_tr+1:n_tr+n_va]
    Xte = X[:, :, n_tr+n_va+1:n_tr+n_va+n_te]
    Yte = Y[:, n_tr+n_va+1:n_tr+n_va+n_te]
    return Xtr, Ytr, Xva, Yva, Xte, Yte
end

batches(X, Y, batchsize) = begin
    N = size(X, 3)
    [ (X[:, :, i:min(i+batchsize-1, N)], Y[:, i:min(i+batchsize-1, N)]) for i in 1:batchsize:N ]
end

mse(ŷ, y) = mean((ŷ .- y).^2)
mae(ŷ, y) = mean(abs.(ŷ .- y))
rmse(ŷ, y) = sqrt(mse(ŷ, y))
function r2(ŷ, y)
    sse = sum((ŷ .- y).^2)
    sst = sum((y .- mean(y)).^2) + eps()
    return 1 .- sse ./ sst
end
mape(ŷ, y; ϵ=1f-6) = mean(abs.((ŷ .- y) ./ (abs.(y) .+ ϵ))) * 100

function compute_metrics(model, ps, st, X_scaled, Y_scaled, scaler::StandardScaler, to)
    Xd, Yd = to(X_scaled), to(Y_scaled)
    ŷ_scaled = Lux.apply(model, Xd, ps, st)[1]
    # back to CPU and inverse-transform
    ŷ = inverse_targets(scaler, Array(ŷ_scaled))
    y = inverse_targets(scaler, Array(Yd))
    return (
        mse = mse(ŷ, y),
        mae = mae(ŷ, y),
        rmse = rmse(ŷ, y),
        r2 = r2(ŷ, y),
        mape = mape(ŷ, y),
    )
end