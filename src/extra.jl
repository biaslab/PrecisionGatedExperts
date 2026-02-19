# Unavoidable type piracy due to RxInfer design

using BayesBase

function BayesBase.prod(::GenericProd, left::ProductOf{L,R}, ::Missing) where {L,R}
    return left
end

function BayesBase.prod(::GenericProd, ::Missing, right::ProductOf{L,R}) where {L,R}
    return right
end
