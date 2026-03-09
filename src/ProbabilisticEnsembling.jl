module ProbabilisticEnsembling

using RxInfer
using StableRNGs

include("utils.jl")
include("log.jl")
include("extra.jl")
include("models.jl")

# low rank normal message type (must precede softdot which uses it)
include("low_rank_normal.jl")
# low rank softdot
include("low_rank_softdot.jl")
# low rank diagonal product rules (ADF diagonal-precision accumulator)
include("low_rank_diagonal.jl")

# shared infrastructure (must precede model files)
include("model_zoo/model_types.jl")
include("model_zoo/features.jl")
include("model_zoo/shared_pipeline.jl")
include("model_zoo/model_specifier.jl")

# model implementations
# static
include("model_zoo/static/model_type.jl")
include("model_zoo/static/univariate.jl")
include("model_zoo/static/multivariate.jl")
include("model_zoo/static/pipeline.jl")

# dynamic
include("model_zoo/dynamic/model_type.jl")
include("model_zoo/dynamic/univariate.jl")
include("model_zoo/dynamic/multivariate.jl")
include("model_zoo/dynamic/pipeline.jl")

# dynamic_diagonal
include("model_zoo/dynamic_diagonal/model_type.jl")
include("model_zoo/dynamic_diagonal/univariate.jl")
include("model_zoo/dynamic_diagonal/multivariate.jl")
include("model_zoo/dynamic_diagonal/pipeline.jl")

# noisy experts
include("model_zoo/noisy_experts/model_type.jl")
include("model_zoo/noisy_experts/univariate.jl")
include("model_zoo/noisy_experts/pipeline.jl")

# neural ensemble pipeline (Adaptive Mixture of Local Experts)
include("neural_ensemble/neural_ensemble_types.jl")
include("neural_ensemble/gating.jl")
include("neural_ensemble/neural_pipeline.jl")
include("neural_ensemble/neural_ensemble_specifier.jl")

# viz utils
include("dist_normalized_weights.jl")

end
