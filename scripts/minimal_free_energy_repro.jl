#!/usr/bin/env julia

using RxInfer
using ExponentialFamily
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ClosedFormStrategy
using Distributions
using Random
using Statistics
using LinearAlgebra
using Plots
using ProbabilisticEnsembling

@model function dynamic_repro_model(n_forecasters, n_obs, features, predictions, y, w_priors, τ_priors, β_priors)
    local w, z, γ, τ, β

    for i in 1:n_forecasters
        w[i] ~ w_priors[i]
        τ[i] ~ τ_priors[i]
        β[i] ~ β_priors[i]
    end

    for j in 1:n_obs
        for i in 1:n_forecasters
            z[i, j] ~ softdot(features[j], w[i], τ[i])
            γ[i, j] ~ GammaShapeRate(1.0, β[i])
            z[i, j] ~ Log(γ[i, j])
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i, j])
        end
    end
end

@constraints function dynamic_repro_constraints()
    q(w, z, γ, τ, β) = q(w)q(z, γ)q(τ)q(β)
    q(z) :: ProjectedTo(NormalMeanVariance, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
    q(γ) :: ProjectedTo(Gamma, parameters=ProjectionParameters(strategy=ClosedFormStrategy()))
end

@initialization function dynamic_repro_init(n_features)
    q(w) = MvNormalMeanPrecision(zeros(n_features), diagm(ones(n_features)))
    q(z) = NormalMeanVariance(0.0, 1.0)
    q(γ) = GammaShapeScale(1.0, 1.0)
    q(τ) = GammaShapeScale(1.0, 1.0)
    q(β) = GammaShapeRate(1.0, 1.0)
end

@model function static_repro_model(n_forecasters, predictions, y, priors)
    local γ
    for i in 1:n_forecasters
        γ[i] ~ priors[i]
    end
    for i in 1:n_forecasters
        for j in eachindex(y)
            y[j] ~ NormalMeanPrecision(predictions[i, j], γ[i])
        end
    end
end

function create_features(x)
    return [Float64[1.0, xi] for xi in x]
end

function main()
    Random.seed!(7)

    n_train = 60
    n_iterations = 80
    n_forecasters = 2

    x = collect(range(0, 2π, length=n_train))
    y_true = sin.(x)
    y = y_true .+ 0.05 .* randn(n_train)

    predictions = zeros(n_forecasters, n_train)
    # Intentionally "cheating" forecaster to trigger aggressive precision behavior.
    predictions[1, :] = y
    predictions[2, :] .= 0.0

    features = create_features(x)
    n_features = length(features[1])

    w_priors = [MvNormalMeanPrecision(zeros(n_features), 0.1 * diagm(ones(n_features))) for _ in 1:n_forecasters]

    function run_dynamic(τ_priors, β_priors)
        return infer(
            model = dynamic_repro_model(
                n_forecasters = n_forecasters,
                n_obs = n_train,
                w_priors = w_priors,
                τ_priors = τ_priors,
                β_priors = β_priors
            ),
            data = (y = y, features = features, predictions = predictions),
            constraints = dynamic_repro_constraints(),
            initialization = dynamic_repro_init(n_features),
            iterations = n_iterations,
            free_energy = true,
            showprogress = false
        )
    end

    τ_priors_base = [GammaShapeScale(1.0, 1e12) for _ in 1:n_forecasters]
    β_priors_base = [GammaShapeRate(1.0, 1.0) for _ in 1:n_forecasters]
    dynamic_base = run_dynamic(τ_priors_base, β_priors_base)

    # Stronger regularization:
    # - smaller τ scale keeps softdot less extreme
    # - larger β prior mean pushes γ = Gamma(shape=1, rate=β) toward smaller values
    τ_priors_reg = [GammaShapeScale(2.0, 0.2) for _ in 1:n_forecasters]
    β_priors_reg = [GammaShapeRate(20.0, 1.0) for _ in 1:n_forecasters]
    dynamic_reg = run_dynamic(τ_priors_reg, β_priors_reg)

    τ_priors_reg2 = [GammaShapeScale(2.0, 0.05) for _ in 1:n_forecasters]
    β_priors_reg2 = [GammaShapeRate(50.0, 1.0) for _ in 1:n_forecasters]
    dynamic_reg2 = run_dynamic(τ_priors_reg2, β_priors_reg2)

    static_priors = [GammaShapeRate(1.0, 1e-12) for _ in 1:n_forecasters]
    static_result = infer(
        model = static_repro_model(n_forecasters = n_forecasters, priors = static_priors),
        data = (y = y, predictions = predictions),
        iterations = n_iterations,
        free_energy = true,
        showprogress = false
    )

    dynamic_fe = dynamic_base.free_energy
    dynamic_fe_reg = dynamic_reg.free_energy
    dynamic_fe_reg2 = dynamic_reg2.free_energy
    static_fe = static_result.free_energy

    println("Dynamic FE first 5: ", round.(dynamic_fe[1:5], digits=3))
    println("Dynamic FE last 5:  ", round.(dynamic_fe[end-4:end], digits=3))
    println("Dynamic(REG) FE first 5: ", round.(dynamic_fe_reg[1:5], digits=3))
    println("Dynamic(REG) FE last 5:  ", round.(dynamic_fe_reg[end-4:end], digits=3))
    println("Dynamic(REG2) FE first 5: ", round.(dynamic_fe_reg2[1:5], digits=3))
    println("Dynamic(REG2) FE last 5:  ", round.(dynamic_fe_reg2[end-4:end], digits=3))
    println("Static FE first 5:  ", round.(static_fe[1:5], digits=3))
    println("Static FE last 5:   ", round.(static_fe[end-4:end], digits=3))

    p = plot(1:length(dynamic_fe), dynamic_fe,
        xlabel = "Iteration", ylabel = "Free Energy",
        title = "Minimal Free Energy Reproduction",
        label = "Dynamic(base)", lw = 2, color = :blue, marker = :circle, markersize = 3
    )
    plot!(p, 1:length(dynamic_fe_reg), dynamic_fe_reg,
        label = "Dynamic(reg)", lw = 2, color = :green, marker = :diamond, markersize = 3
    )
    plot!(p, 1:length(dynamic_fe_reg2), dynamic_fe_reg2,
        label = "Dynamic(reg2)", lw = 2, color = :orange, marker = :utriangle, markersize = 3
    )
    plot!(p, 1:length(static_fe), static_fe,
        label = "Static", lw = 2, color = :red, marker = :square, markersize = 3
    )
    savefig(p, "free_energy_minimal_repro.png")
    println("Saved: free_energy_minimal_repro.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
