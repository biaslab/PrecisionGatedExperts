export FrozenDistribution

# Unavoidable type piracy due to RxInfer design

using BayesBase

function BayesBase.prod(::GenericProd, left::ProductOf{L,R}, ::Missing) where {L,R}
    return left
end

function BayesBase.prod(::GenericProd, ::Missing, right::ProductOf{L,R}) where {L,R}
    return right
end

"""
    SubsampledData(original_data, subsample_size; repeat_batch = 1)

A special structure that if passed as `data` inside of RxInfer will subsample
the original dataset on every variational iteration.

To subsample the structure uses stable `rng`. The seed of the `rng` is instantiated
automatically with `subsample_size`, this is useful when you have different datasets
passed to `data` that must be "synced" together.

If `original_data` is of type `Vector` the procedure simply samples the
elements of that vector. If `original_data` is of type `Matrix` the procedure
subsamples the columns, so the number of rows stays the same.

`repeat_batch` controls how many consecutive calls reuse the same sampled batch
before resampling. Default is `1` (resample every call).
"""
struct SubsampledData{R,T}
    original_data::T
    subsample_size::Int
    rng::R
    sampled_ids::Vector{Int}
    repeat_batch::Int
    call_count::Base.RefValue{Int}
    cached_data::Base.RefValue{Any}
end

function SubsampledData(
    original_data::T,
    subsample_size::Int,
    repeat_batch::Int = 1,
) where {T}
    rng = StableRNGs.StableRNG(subsample_size)
    sampled_ids = zeros(Int, subsample_size)
    return SubsampledData(
        original_data,
        subsample_size,
        rng,
        sampled_ids,
        repeat_batch,
        Ref(0),
        Ref{Any}(nothing),
    )
end

function SubsampledData(original_data::T, subsample_size::Int, ::Nothing) where {T}
    rng = StableRNGs.StableRNG(subsample_size)
    sampled_ids = zeros(Int, subsample_size)
    return SubsampledData(
        original_data,
        subsample_size,
        rng,
        sampled_ids,
        1,
        Ref(0),
        Ref{Any}(nothing),
    )
end

function RxInfer.get_data(d::SubsampledData)
    d.call_count[] += 1
    if d.call_count[] == 1 || d.call_count[] > d.repeat_batch
        d.cached_data[] =
            subsample_data(d.rng, d.sampled_ids, d.original_data, d.subsample_size)
        d.call_count[] = 1
    end
    return d.cached_data[]
end

function subsample_data(rng, sampled_ids, data::Vector, subsample_size::Int)
    indx = 1:size(data, 1)
    sample!(rng, indx, sampled_ids; replace = false)
    return view(data, sampled_ids)
end

function subsample_data(rng, sampled_ids, data::Matrix, subsample_size::Int)
    colindx = 1:size(data, 2)
    sample!(rng, colindx, sampled_ids; replace = false)
    return view(data, :, sampled_ids)
end
