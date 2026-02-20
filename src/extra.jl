# Unavoidable type piracy due to RxInfer design

using BayesBase

function BayesBase.prod(::GenericProd, left::ProductOf{L,R}, ::Missing) where {L,R}
    return left
end

function BayesBase.prod(::GenericProd, ::Missing, right::ProductOf{L,R}) where {L,R}
    return right
end

"""
    SubsampleData(original_data, subsample_size)

A special structure that if passed as `data` inside of RxInfer will subsample 
the original dataset on every variational iteration. 

To subsample the structure uses stable `rng`. The seed of the `rng` is instantiated 
automatically with `subsample_size`, this is useful when you have different datasets 
passed to `data` that must be "synced" together. 

If `original_data` is of type `Vector` the procedure simply samples the
elements of that vector. If `original_data` is of type `Matrix` the procedure 
subsamples the columns, so the number of rows stays the same.
"""
struct SubsampledData{R,T}
    original_data::T
    subsample_size::Int
    rng::R
    sampled_ids::Vector{Int}
end

function SubsampledData(original_data::T, subsample_size::Int) where {T}
    rng = StableRNGs.StableRNG(subsample_size)
    sampled_ids = zeros(Int, subsample_size)
    return SubsampledData(original_data, subsample_size, rng, sampled_ids)
end

function RxInfer.get_data(d::SubsampledData)
    return subsample_data(d.rng, d.sampled_ids, d.original_data, d.subsample_size)
end

function subsample_data(rng, sampled_ids, data::Vector, subsample_size::Int)
    indx = 1:size(data, 1)
    sample!(rng, indx, sampled_ids; replace=false)
    return view(data, sampled_ids)
end

function subsample_data(rng, sampled_ids, data::Matrix, subsample_size::Int)
    colindx = 1:size(data, 2)
    sample!(rng, colindx, sampled_ids; replace=false)
    return view(data, :, sampled_ids)
end
