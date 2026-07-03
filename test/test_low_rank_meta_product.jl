using Test

using ProbabilisticEnsembling
using BayesBase
using ExponentialFamily
using LinearAlgebra
using ReactiveMP

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

@testset "Low-rank likelihood messages can multiply before prior" begin
    first_message = LowRankNormalWeightedMeanPrecision(
        [1.0, 2.0],
        [1.0, 2.0],
        0.5,
    )
    second_message = LowRankNormalWeightedMeanPrecision(
        [3.0, 5.0],
        [2.0, -1.0],
        2.0,
    )
    prior = MvNormalMeanScalePrecision([0.0, 0.0], 0.25)

    likelihood_product = prod(
        PreserveTypeProd(Distribution),
        first_message,
        second_message,
    )
    posterior = prod(
        PreserveTypeProd(Distribution),
        likelihood_product,
        prior,
    )

    @test likelihood_product isa MvNormalWeightedMeanPrecision
    @test weightedmean(likelihood_product) == [4.0, 7.0]
    @test Matrix(BayesBase.invcov(likelihood_product)) == [
        8.5 -3.0
        -3.0 4.0
    ]
    @test posterior isa MvNormalWeightedMeanPrecision
    @test weightedmean(posterior) == [4.0, 7.0]
    @test Matrix(BayesBase.invcov(posterior)) == [
        8.75 -3.0
        -3.0 4.25
    ]
end

@testset "Low-rank structured softdot" begin
    low_rank_message = @call_rule SoftDot(:x, Marginalisation) (
        m_y = NormalMeanVariance(2.0, 3.0),
        q_θ = PointMass([1.0, 2.0]),
        q_γ = PointMass(4.0),
        meta = LowRankMeta(),
    )

    dense_message = @call_rule SoftDot(:x, Marginalisation) (
        m_y = NormalMeanVariance(2.0, 3.0),
        q_θ = PointMass([1.0, 2.0]),
        q_γ = PointMass(4.0),
    )

    @test low_rank_message isa LowRankNormalWeightedMeanPrecision
    @test weightedmean(low_rank_message) ≈ weightedmean(dense_message)
    @test Matrix(BayesBase.invcov(low_rank_message)) ≈ Matrix(BayesBase.invcov(dense_message))
end

@testset "Low-rank structured softdot gamma" begin
    q_y_x = MvNormalWeightedMeanPrecision(
        [1.0, 0.5, -0.25],
        [
            3.0 0.2 0.1
            0.2 2.0 0.3
            0.1 0.3 1.5
        ],
    )
    q_θ = PointMass([0.75, -1.25])

    low_rank_message = @call_rule SoftDot(:γ, Marginalisation) (
        q_y_x = q_y_x,
        q_θ = q_θ,
        meta = LowRankMeta(),
    )
    dense_message = @call_rule SoftDot(:γ, Marginalisation) (
        q_y_x = q_y_x,
        q_θ = q_θ,
    )

    @test low_rank_message isa GammaShapeRate
    @test BayesBase.shape(low_rank_message) ≈ BayesBase.shape(dense_message)
    @test BayesBase.rate(low_rank_message) ≈ BayesBase.rate(dense_message)
end

@testset "Low-rank mean-field softdot gamma with point-mass theta" begin
    q_y = NormalMeanVariance(2.0, 3.0)
    q_θ = PointMass([0.75, -1.25])
    q_x = MvNormalMeanCovariance(
        [1.0, -2.0],
        [
            3.0 0.2
            0.2 2.0
        ],
    )

    low_rank_message = @call_rule SoftDot(:γ, Marginalisation) (
        q_y = q_y,
        q_θ = q_θ,
        q_x = q_x,
        meta = LowRankMeta(),
    )
    dense_message = @call_rule SoftDot(:γ, Marginalisation) (
        q_y = q_y,
        q_θ = q_θ,
        q_x = q_x,
    )

    @test low_rank_message isa GammaShapeRate
    @test BayesBase.shape(low_rank_message) ≈ BayesBase.shape(dense_message)
    @test BayesBase.rate(low_rank_message) ≈ BayesBase.rate(dense_message)
end

@testset "Low-rank structured softdot y prediction" begin
    low_rank_message = @call_rule SoftDot(:y, Marginalisation) (
        q_θ = PointMass([1.0, 2.0]),
        m_x = MvNormalMeanCovariance([3.0, 4.0], Matrix{Float64}(I, 2, 2)),
        q_γ = PointMass(5.0),
        meta = LowRankMeta(),
    )

    dense_message = @call_rule SoftDot(:y, Marginalisation) (
        q_θ = PointMass([1.0, 2.0]),
        m_x = MvNormalMeanCovariance([3.0, 4.0], Matrix{Float64}(I, 2, 2)),
        q_γ = PointMass(5.0),
    )

    @test low_rank_message isa NormalMeanVariance
    low_rank_mean, low_rank_cov = mean_cov(low_rank_message)
    dense_mean, dense_cov = mean_cov(dense_message)
    @test low_rank_mean ≈ dense_mean
    @test low_rank_cov ≈ dense_cov
end

@testset "Low-rank structured softdot average energy" begin
    q_y_x = MvNormalMeanCovariance(
        [1.0, 3.0, 4.0],
        Matrix{Float64}(I, 3, 3),
    )
    q_θ = PointMass([1.0, 2.0])
    q_γ = PointMass(5.0)
    marginals = (
        Marginal(q_y_x, false, false),
        Marginal(q_θ, false, false),
        Marginal(q_γ, false, false),
    )

    low_rank_score = score(
        AverageEnergy(),
        SoftDot,
        Val{(:y_x, :θ, :γ)}(),
        marginals,
        LowRankMeta(),
    )
    dense_score = score(
        AverageEnergy(),
        SoftDot,
        Val{(:y_x, :θ, :γ)}(),
        marginals,
        nothing,
    )

    @test isfinite(low_rank_score)
    @test low_rank_score ≈ dense_score
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
