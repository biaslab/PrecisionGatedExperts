import BayesBase: PointMass
using ReactiveMP
using ReactiveMP: score, AverageEnergy, SoftDot

export LowRankMeta

struct LowRankMeta end

include("low_rank_softdot/mean_field.jl")
include("low_rank_softdot/structured.jl")
