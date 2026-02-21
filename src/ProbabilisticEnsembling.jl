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

# model zoo univariate_y
include("model_zoo/univariate_y/static_ensemble.jl")
include("model_zoo/univariate_y/dynamic.jl")
include("model_zoo/univariate_y/hierarchial.jl")
include("model_zoo/univariate_y/deep.jl")

# model zoo multivariate_y
include("model_zoo/multivariate_y/static_ensemble.jl")
include("model_zoo/multivariate_y/dynamic.jl")
include("model_zoo/multivariate_y/hierarchial.jl")
include("model_zoo/multivariate_y/deep.jl")

# model types & prior parsing
include("model_zoo/model_types.jl")

# model running
include("model_zoo/model_specifier.jl")

end
