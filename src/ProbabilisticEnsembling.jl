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

# shared infrastructure (must precede model files)
include("model_zoo/model_types.jl")
include("model_zoo/features.jl")
include("model_zoo/shared_pipeline.jl")
include("model_zoo/model_specifier.jl")

# model implementations
# static
include("model_zoo/univariate_y/static/model_type.jl")
include("model_zoo/univariate_y/static/static.jl")
include("model_zoo/multivariate_y/static/static.jl")
include("model_zoo/univariate_y/static/pipeline.jl")

# dynamic
include("model_zoo/univariate_y/dynamic/model_type.jl")
include("model_zoo/univariate_y/dynamic/dynamic.jl")
include("model_zoo/multivariate_y/dynamic/dynamic.jl")
include("model_zoo/univariate_y/dynamic/pipeline.jl")

# neural ensemble pipeline (Adaptive Mixture of Local Experts)
include("neural_ensemble/neural_ensemble_types.jl")
include("neural_ensemble/gating.jl")
include("neural_ensemble/neural_pipeline.jl")
include("neural_ensemble/neural_ensemble_specifier.jl")

# viz utils
include("dist_normalized_weights.jl")

end
