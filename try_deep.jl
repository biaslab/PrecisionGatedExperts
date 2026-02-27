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

@model function deep_model_xor_genuine(n_obs, n_forecasters, features, y, predictors)
    local h, right_switch, left_switch, z, kappa, γ
    for i in 1:n_forecasters
        for j = 1:n_obs
            # Level 1: top routing score.
            h[j, i] ~ softdot(features[j], V_SPLIT, TAU_SOFTDOT)

            # Level 2: two opposite softdot routers driven by h.
            right_switch[j, i] ~ softdot(h[j, i], 1.0, TAU_SOFTDOT)
            left_switch[j, i] ~ softdot(h[j, i], -1.0, TAU_SOFTDOT)

            # Active branch has switch near -7 -> smaller residual -> larger inferred precision.
            kappa[j, 1, i] ~ GammaShapeRate(1.0, 1.0)
            kappa[j, 2, i] ~ GammaShapeRate(1.0, 1.0)
            right_switch[j, i] ~ Log(kappa[j, 1, i])
            left_switch[j, i] ~ Log(kappa[j, 2, i])

            # Two lower experts.
            z[j, 1, i] ~ softdot(features[j], W_LEFT, TAU_SOFTDOT)
            z[j, 2, i] ~ softdot(features[j], W_RIGHT, TAU_SOFTDOT)

            # Product of normals: inferred kappa decides which expert dominates.
            m[j, i] ~ NormalMeanPrecision(z[j, 1, i], kappa[j, 1, i])
            m[j, i] ~ NormalMeanPrecision(z[j, 2, i], kappa[j, 2, i])
            γ[j, i] ~ GammaShapeRate(1.0, 1.0)
            m[j, i] ~ Log(γ[j, i])
            y[j] ~ NormalMeanPrecision(predictors[i, j], γ[j, i])
        end
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
    targets = Int64[0, 1, 1, 0]
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
    y_mean = mean.(y_post)
    y_var = var.(y_post)
    y_class = Int.(y_mean .>= 0.5)
    acc_direct = mean(y_class .== targets)
    acc_inverted = mean((1 .- y_class) .== targets)

    h_mean = map(mean, result.posteriors[:h][end])
    right_switch_mean = map(mean, result.posteriors[:right_switch][end])
    left_switch_mean = map(mean, result.posteriors[:left_switch][end])
    kappa_mean = map(mean, result.posteriors[:kappa][end])
    z_mean = map(mean, result.posteriors[:z][end])
    m_mean = map(mean, result.posteriors[:m][end])
    gamma_mean = map(mean, result.posteriors[:γ][end])

    println("XOR capability check (no learning, kappa inferred):")
    for j = 1:length(features)
        x1 = Int(features[j][1])
        x2 = Int(features[j][2])
        println("x=[$x1,$x2] target=$(targets[j]) pred=$(round(y_mean[j], digits = 4)) var=$(round(y_var[j], digits = 4)) class=$(y_class[j])")
        println("  h=$(round.(h_mean[j, :], digits = 4))")
        println("  right_switch=$(round.(right_switch_mean[j, :], digits = 4))")
        println("  left_switch=$(round.(left_switch_mean[j, :], digits = 4))")
        println("  kappa[:,:,branch1/2]=$(round.(kappa_mean[j, :, :], digits = 4))")
        println("  z[:,:,branch1/2]=$(round.(z_mean[j, :, :], digits = 4))")
        println("  m=$(round.(m_mean[j, :], digits = 4))")
        println("  gamma=$(round.(gamma_mean[j, :], digits = 4))")
    end
    println("Classification accuracy (direct): ", round(acc_direct, digits = 4))
end
