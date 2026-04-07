using Test

using ProbabilisticEnsembling
using BayesBase
using ExponentialFamily
using LinearAlgebra

@testset "Low-rank product" begin
    dense_left = MvNormalWeightedMeanPrecision(
        [10.0, 20.0, 30.0],
        Matrix(Diagonal([3.0, 5.0, 7.0])),
    )

    right = LowRankNormalWeightedMeanPrecision(
        [1.0, 2.0, 4.0],
        [1.0, 2.0, 4.0],
        1.0,
    )

    dense_result = prod(PreserveTypeProd(Distribution), dense_left, right)

    @test weightedmean(dense_result) == [11.0, 22.0, 34.0]
    @test Matrix(BayesBase.invcov(dense_result)) == [
        4.0 2.0 4.0
        2.0 9.0 8.0
        4.0 8.0 23.0
    ]

    @test begin
        diagonal_result = prod(
            PreserveTypeProd(Distribution),
            MvNormalWeightedMeanPrecision(
                [10.0, 20.0, 30.0],
                Diagonal([3.0, 5.0, 7.0]),
            ),
            LowRankNormalWeightedMeanPrecision([1.0, 2.0, 4.0], [1.0, 2.0, 4.0], 1.0),
        )

        weightedmean(diagonal_result) == weightedmean(dense_result) &&
            Matrix(BayesBase.invcov(diagonal_result)) == Matrix(BayesBase.invcov(dense_result))
    end

    @test begin
        symmetric_result = prod(
            PreserveTypeProd(Distribution),
            MvNormalWeightedMeanPrecision(
                [10.0, 20.0, 30.0],
                Symmetric(Matrix(Diagonal([3.0, 5.0, 7.0]))),
            ),
            LowRankNormalWeightedMeanPrecision([1.0, 2.0, 4.0], [1.0, 2.0, 4.0], 1.0),
        )

        weightedmean(symmetric_result) == weightedmean(dense_result) &&
            Matrix(BayesBase.invcov(symmetric_result)) == Matrix(BayesBase.invcov(dense_result))
    end
end

@testset "Low-rank diagonal update" begin
    @test begin
        diagonal_update_result = prod(
            PreserveTypeProd(Distribution),
            MvNormalWeightedMeanPrecision(
                [10.0, 20.0, 30.0],
                Matrix(Diagonal([3.0, 5.0, 7.0])),
            ),
            LowRankDiagonalUpdate(
                [1.0, 2.0, 4.0],
                [1.0, 2.0, 4.0],
                1.0,
            ),
        )

        weightedmean(diagonal_update_result) == [11.0, 22.0, 34.0] &&
            Matrix(BayesBase.invcov(diagonal_update_result)) == [
                4.0 2.0 4.0
                2.0 9.0 8.0
                4.0 8.0 23.0
            ]
    end
end
