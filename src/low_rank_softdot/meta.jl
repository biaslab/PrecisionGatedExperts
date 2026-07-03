import BayesBase: PointMass
using ReactiveMP
using ReactiveMP: score, AverageEnergy, SoftDot
using LinearAlgebra: BLAS

using LowRankMatrices

export LowRankMeta

struct LowRankMeta end
