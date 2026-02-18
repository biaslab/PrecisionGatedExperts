module ProbabilisticEnsembling

using RxInfer

include("utils.jl")
include("log.jl")
include("extra.jl")
include("models.jl")

# low rank normal message type (must precede softdot which uses it)
include("low_rank_normal.jl")
# low rank softdot
include("low_rank_softdot.jl")

# model zoo univariate_y
include("model_zoo/univariate_y/static_ensemble.jl")

# model zoo multivariate_y
include("model_zoo/multivariate_y/static_ensemble.jl")

# model running
include("model_zoo/model_specifier.jl")

end
