#!/usr/bin/env julia

using JLD2
using Printf
using ProbabilisticEnsembling

const MODELS_DIR = joinpath(@__DIR__, "..", "models")
const DATA_DIR = joinpath(@__DIR__, "..", "data")
const DATASETS = Set(["ETTh1", "ETTh2"])

function is_forecasting_checkpoint(path::String)
    name = basename(path)
    return occursin(r"^ETTh[12]_h\d+_s\d+_.+_enzyme\.jld2$", name)
end

function metric_tuple(yhat, y)
    return (
        mse=mse(yhat, y),
        mae=mae(yhat, y),
        rmse=rmse(yhat, y),
        r2=r2(yhat, y),
        mape=mape(yhat, y),
        smape=smape(yhat, y),
    )
end

function print_metrics(io, label, m)
    @printf(
        io,
        "%s mse=%.6f mae=%.6f \n",
        label,
        m.mse,
        m.mae,
    )
end

function evaluate_model(path::String)
    is_forecasting_checkpoint(path) || return nothing

    saved = load_jld2_model(path)
    meta = saved.meta

    dataset = String(meta.dataset)
    dataset in DATASETS || return nothing

    csv_path = joinpath(DATA_DIR, dataset * ".csv")
    Xmat, feat_cols = load_ett(csv_path)
    ot_idx = ProbabilisticEnsembling.find_column_index(feat_cols, "OT")

    seq_len = Int(meta.seq_len)
    horizon = Int(meta.horizon)
    split = meta.split

    X3, Y2 = make_sequences(Xmat; seq_len=seq_len, horizon=horizon)
    _, _, _, _, Xte, Yte =
        train_val_test_split(X3, Y2; ratios=(split.train, split.val, split.test))

    scaler = meta.scaler
    Xte_s = scale_inputs(scaler, Xte)
    Yte_s = scale_targets(scaler, Yte)

    model = build_model(saved.model_type, saved.config)
    yhat = ProbabilisticEnsembling.predict_unscaled(
        model,
        saved.parameters,
        saved.states,
        Xte_s,
    )

    yhat_ot_scaled = vec(Float64.(yhat[ot_idx, :]))
    y_ot_scaled = vec(Float64.(Yte_s[ot_idx, :]))

    μ = Float64(scaler.μ[ot_idx])
    σ = Float64(scaler.σ[ot_idx])

    yhat_ot = yhat_ot_scaled .* σ .+ μ
    y_ot = y_ot_scaled .* σ .+ μ

    return (
        file=basename(path),
        dataset=dataset,
        horizon=horizon,
        model_type=String(saved.model_type),
        scaled=metric_tuple(yhat_ot_scaled, y_ot_scaled),
        original=metric_tuple(yhat_ot, y_ot),
        n_test=length(y_ot),
    )
end

function main()
    files = sort(filter(f -> endswith(f, ".jld2"), readdir(MODELS_DIR; join=true)))
    results = filter(!isnothing, evaluate_model.(files))

    println("OT-only metrics for ETTh1 / ETTh2 models")
    println("scaled = directly comparable to current ensemble metrics")
    println("original = metrics on the original OT scale")
    println()

    for r in sort(results, by=x -> (x.dataset, x.horizon, x.file))
        println("file=$(r.file) dataset=$(r.dataset) horizon=$(r.horizon) n_test=$(r.n_test)")
        print_metrics(stdout, "  scaled  ", r.scaled)
        #print_metrics(stdout, "  original", r.original)
        println()
    end
end

main()
