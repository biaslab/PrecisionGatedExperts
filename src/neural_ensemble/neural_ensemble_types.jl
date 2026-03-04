# ---------------------------------------------------------------------------
# Neural Ensemble Types
# ---------------------------------------------------------------------------

struct NeuralEnsembleSpecifier{P,D,F}
    prediction_type::P
    column::Union{String,Nothing}
    dataset::D
    dataset_path::String
    horizon::Int
    experts::Vector{String}
    train_set::Bool
    gating_layers::Int
    gating_hidden_dim::Int
    n_epochs::Int
    patience::Int
    min_delta::Float32
    learning_rate::Float32
    feature_type::F
    selected_quantiles::Vector{Float64}
    save_dir::String
end
