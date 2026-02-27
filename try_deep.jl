#!/usr/bin/env julia

using ProbabilisticEnsembling
using RxInfer
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Statistics

# Fixed, non-learned parameters for the XOR capability demo.
const V_SPLIT = [-14.0, 0.0, 7.0]   # top router: h ~ +7 for x1=0, -7 for x1=1
const W_LEFT = [0.0, 10.0, 0.0]      # lower expert 1: predicts x2
const W_RIGHT = [0.0, -10.0, 10.0]    # lower expert 2: predicts 1 - x2
const TAU_SOFTDOT = 200.0

@model function deep_model_xor_genuine(n_obs, features, y, predictors)
    local h, right_switch, left_switch, z, kappa, γ

    for j = 1:n_obs
        # Level 1: top routing score.
        h[j, 1] ~ softdot(features[j], V_SPLIT, TAU_SOFTDOT)

        # Level 2: two opposite softdot routers driven by h.
        right_switch[j, 1] ~ softdot(h[j, 1], 1.0, TAU_SOFTDOT)
        left_switch[j, 1] ~ softdot(h[j, 1], -1.0, TAU_SOFTDOT)

        # Active branch has switch near -7 -> smaller residual -> larger inferred precision.
        kappa[j, 1, 1] ~ GammaShapeRate(1.0, 1.0)
        kappa[j, 2, 1] ~ GammaShapeRate(1.0, 1.0)
        right_switch[j, 1] ~ Log(kappa[j, 1, 1])
        left_switch[j, 1] ~ Log(kappa[j, 2, 1])

        # Two lower experts.
        z[j, 1, 1] ~ softdot(features[j], W_LEFT, TAU_SOFTDOT)
        z[j, 2, 1] ~ softdot(features[j], W_RIGHT, TAU_SOFTDOT)

        # Product of normals: inferred kappa decides which expert dominates.
        m[j, 1] ~ NormalMeanPrecision(z[j, 1, 1], kappa[j, 1, 1])
        m[j, 1] ~ NormalMeanPrecision(z[j, 2, 1], kappa[j, 2, 1])
        γ[j, 1] ~ GammaShapeRate(1.0, 1.0)
        m[j, 1] ~ Log(γ[j, 1])
        y[j] ~ NormalMeanPrecision(predictors[1, j], γ[j, 1])
    end

    for j = 1:n_obs
        # Level 1: top routing score.
        h[j, 2] ~ softdot(features[j], -V_SPLIT, TAU_SOFTDOT)

        # Level 2: two opposite softdot routers driven by h.
        right_switch[j, 2] ~ softdot(h[j, 2], 1.0, TAU_SOFTDOT)
        left_switch[j, 2] ~ softdot(h[j, 2], -1.0, TAU_SOFTDOT)

        # Active branch has switch near -7 -> smaller residual -> larger inferred precision.
        kappa[j, 1, 2] ~ GammaShapeRate(1.0, 1.0)
        kappa[j, 2, 2] ~ GammaShapeRate(1.0, 1.0)
        right_switch[j, 2] ~ Log(kappa[j, 1, 2])
        left_switch[j, 2] ~ Log(kappa[j, 2, 2])

        # Two lower experts.
        z[j, 1, 2] ~ softdot(features[j], W_LEFT, TAU_SOFTDOT)
        z[j, 2, 2] ~ softdot(features[j], W_RIGHT, TAU_SOFTDOT)

        # Product of normals: inferred kappa decides which expert dominates.
        m[j, 2] ~ NormalMeanPrecision(z[j, 1, 2], kappa[j, 1, 2])
        m[j, 2] ~ NormalMeanPrecision(z[j, 2, 2], kappa[j, 2, 2])
        γ[j, 2] ~ GammaShapeRate(1.0, 1.0)
        m[j, 2] ~ Log(γ[j, 2])
        y[j] ~ NormalMeanPrecision(predictors[2, j], γ[j, 2])
    end
end

@constraints function deep_constraints()
    q(h, right_switch, left_switch, z, kappa, m, γ) = q(h)q(right_switch)q(left_switch)q(z)q(kappa)q(m, γ)
    q(h)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(right_switch)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(left_switch)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(z)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(kappa)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(m)::ProjectedTo(
        NormalMeanVariance,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
    q(γ)::ProjectedTo(
        Gamma,
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )
end

@initialization function deep_init()
    q(h) = NormalMeanVariance(0.0, 1.0)
    q(right_switch) = NormalMeanVariance(0.0, 1.0)
    q(left_switch) = NormalMeanVariance(0.0, 1.0)
    q(z) = NormalMeanVariance(0.5, 1.0)
    q(kappa) = GammaShapeScale(2.0, 1.0)
    q(m) = NormalMeanVariance(0.5, 1.0)
    q(γ) = GammaShapeScale(2.0, 1.0)
end

function xor_dataset()
    features = [
        Float64[0.0, 0.0, 1.0],
        Float64[0.0, 1.0, 1.0],
        Float64[1.0, 0.0, 1.0],
        Float64[1.0, 1.0, 1.0],
    ]
    targets = [0 0 0 0; 1 1 1 1]
    return features, targets
end

function run_xor_demo()
    features, targets = xor_dataset()
    predictors = [0 0 0 0; 1 1 1 1]

    result = infer(
        model = deep_model_xor_genuine(n_obs = length(features)),
        data = (
            features = features,
            y = fill(missing, length(features)),
            predictors=predictors
        ),
        constraints = deep_constraints(),
        initialization = deep_init(),
        iterations = 20,
        free_energy = false,
        showprogress = false,
    )

    y_post = result.predictions[:y][end]
    @show map(mean, y_post)
    @show map(var, y_post)
    # y_mean = mean.(y_post)
    # y_class = Int.(y_mean .>= 0.5)
    # target_class = Int.(targets)
    # acc = mean(y_class .== target_class)

    # kappa_mean = mean.(result.posteriors[:kappa][end])
    # right_switch_mean = mean.(result.posteriors[:right_switch][end])
    # left_switch_mean = mean.(result.posteriors[:left_switch][end])
    # z_mean = mean.(result.posteriors[:z][end])

    # println("XOR capability check (no learning, kappa inferred):")
    # println("right_switch mean = ", round.(right_switch_mean, digits = 4))
    # println("left_switch mean  = ", round.(left_switch_mean, digits = 4))
    # println("kappa mean matrix:")
    # println(round.(kappa_mean, digits = 4))
    # println("z mean matrix:")
    # println(round.(z_mean, digits = 4))
    # for j = 1:length(features)
    #     x1 = Int(features[j][1])
    #     x2 = Int(features[j][2])
    #     println(
    #         "  x=[$x1,$x2] target=$(target_class[j]) pred_mean=$(round(y_mean[j], digits = 4)) class=$(y_class[j])",
    #     )
    # end
    # println("Classification accuracy: ", round(acc, digits = 4))
end
