#!/usr/bin/env julia

# Inference for LSTM/CNN Lux models saved by scripts/train_ett_lstm_enzyme.jl (JLD2)

using ADTypes
using Lux
using JLD2
using MLUtils
using Reactant
using Random
using Statistics
using ProbabilisticEnsembling
using ExponentialFamilyProjection  # ensure scaler types deserialize


# -----------------------------------------------------------------------------

# Helpers

# -----------------------------------------------------------------------------

reactant_device() = (
    try
        Reactant.default_device()
    catch
        cpu_device()
    end
)
cpu_device() = Lux.cpu_device()


function eval_on_test(model, ps, st, meta)
    # Rebuild dataset and splits per meta
    data_dir = joinpath(@__DIR__, "..", "data")
    ds_path = joinpath(data_dir, String(meta.dataset))
    Xmat, _ = load_ett(ds_path)

    X3, Y2 = make_sequences(Xmat; seq_len=Int(meta.seq_len), horizon=Int(meta.horizon))
    Xtr, Ytr, Xval, Yval, Xte, Yte = train_val_test_split(X3, Y2; ratios=(meta.split.train, meta.split.val, meta.split.test))

    scaler = meta.scaler
    # Scale inputs/targets and evaluate on test split
    Xte_s = scale_inputs(scaler, Xte)
    Yte_s = scale_targets(scaler, Yte)

    dev = reactant_device()
    cdev = cpu_device()
    Xd = dev(Float32.(Xte_s))
    st_test = Lux.testmode(st) |> dev
    ps_d = dev(ps)

    # Compile for inference on Reactant
    model_compiled = if dev isa ReactantDevice
        @compile model(Xd, ps_d, st_test)
    else
        model
    end

    ŷ_sc, _ = model_compiled(Xd, ps_d, st_test)
    ŷ_sc = cdev(ŷ_sc)

    ŷ = inverse_targets(scaler, Array(ŷ_sc))
    y = inverse_targets(scaler, Array(Yte_s))

    return (
        mse=mse(ŷ_sc, Yte_s),
        mae=mae(ŷ_sc, Yte_s),
        rmse=rmse(ŷ_sc, Yte_s),
        r2=r2(ŷ_sc, Yte_s),
        mape=mape(ŷ_sc, Yte_s),
        smape=smape(ŷ_sc, Yte_s),
        n_test=size(Xte, 3),
        n_features=size(Xte, 1),
    )
end

function main()
    if length(ARGS) < 1
        println("Usage: julia scripts/infer_ett_enzyme.jl <model_path.jld2>")
        return
    end
    model_path = ARGS[1]
    @info "Loading JLD2 model" path = model_path

    @assert isfile(model_path) "Model file not found: $(model_path)"
    data = JLD2.jldopen(model_path, "r") do f
        (
            model_type=read(f, "model_type"),
            parameters=read(f, "parameters"),
            states=read(f, "states"),
            config=read(f, "config"),
            meta=read(f, "meta"),
        )
    end

    model = build_model(data.model_type, data.config)
    metrics = eval_on_test(model, data.parameters, data.states, data.meta)

    @info "Test metrics" model = data.model_type dataset = data.meta.dataset horizon = data.meta.horizon metrics...
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
