using ProbabilisticEnsembling
using ExponentialFamily
using JLD2
using Plots
using BayesBase: cov
using Distributions
using StableRNGs

saved = JLD2.load("final_results/exchange_rate_h192_multivariate_probabilisticensembling.dynamic_7473143013234889183.jld2")
spec_saved = saved["spec"]

prediction_type = ProbabilisticEnsembling._parse_saved_prediction_type(string(spec_saved.prediction_type))
model_type = ProbabilisticEnsembling._parse_saved_model_type(string(spec_saved.model_type))
prediction_iterations = 1

function resolve_saved_alpha(saved)
    if haskey(saved, "raw_spec")
        params = get(saved["raw_spec"], "params", nothing)
        if !isnothing(params)
            priors = get(params, "priors", nothing)
            if !isnothing(priors) && haskey(priors, "α")
                α_cfg = priors["α"]
                if haskey(α_cfg, "value")
                    return Float64(α_cfg["value"])
                end
            end
        end
    end
    return 1.0
end

alpha = resolve_saved_alpha(saved)

spec_for_data = ProbabilisticEnsembling._spec_for_prediction_from_saved(
    saved,
    prediction_iterations,
)
experts = spec_for_data.experts
saved_selected_quantiles = spec_for_data.selected_quantiles

_, y_test_all, _, predictions_test_all, _, features_test_all = ProbabilisticEnsembling.before_rxinfer(spec_for_data);
n_steps = length(y_test_all)

y_test = ProbabilisticEnsembling.prepare_y_test(prediction_type, y_test_all, n_steps);
predictions_test = predictions_test_all;
features_test = features_test_all;

n_forecasters = size(predictions_test, 1);
prediction_array = [missing for _ = 1:n_steps]

train_results, train_ensemble_preds, train_influence = begin
    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved, alpha);
    @info "Prediction start"
    infer_test = ProbabilisticEnsembling.predict_with_model(
        prediction_type, model_type, priors;
        n_forecasters = n_forecasters,
        n_steps = n_steps,
        prediction_array = prediction_array,
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = prediction_iterations,
    )
    
    influence = infer_test.posteriors[:γ][end]
    
    @info [mean(posterior) for posterior in infer_test.posteriors[:γ][end]][1:5]
    
    ensemble_preds = infer_test.predictions[:y][end];

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test);

    ensemble_mean, ensemble_std, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics
    )

    ensemble_metrics, ensemble_preds, influence
end;

zero_init_metrics, zero_init_ensemble_preds = begin
    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved, alpha);
    priors[:w] = [MvNormalMeanScalePrecision(zeros(length(features_test[1])), 1e12) for _ in 1:n_forecasters]
    @info "Prediction start"
    infer_test = ProbabilisticEnsembling.predict_with_model(
        prediction_type, model_type, priors;
        n_forecasters = n_forecasters,
        n_steps = n_steps,
        prediction_array = prediction_array,
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = prediction_iterations,
    )

    @info [mean(posterior) for posterior in infer_test.posteriors[:γ][end]][1:5]
    ensemble_preds = infer_test.predictions[:y][end];

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test);

    ensemble_mean, ensemble_std, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics
    )

    ensemble_metrics, ensemble_preds
end

full_default_metrics, full_default_ensemble_preds = begin
    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved, alpha);
    priors[:w] = [MvNormalMeanScalePrecision(zeros(length(features_test[1])), 1e12) for _ in 1:n_forecasters]
    priors[:τ] = [GammaShapeRate(1, 1) for _ in 1:n_forecasters]
    priors[:ρ] = [GammaShapeRate(1, 1) for _ in 1:n_forecasters]
    @info "Prediction start"
    infer_test = ProbabilisticEnsembling.predict_with_model(
        prediction_type, model_type, priors;
        n_forecasters = n_forecasters,
        n_steps = n_steps,
        prediction_array = prediction_array,
        predictions_test = predictions_test,
        features_test = features_test,
        prediction_iterations = prediction_iterations,
    )

    @info [mean(posterior) for posterior in infer_test.posteriors[:γ][end]][1:5]
    ensemble_preds = infer_test.predictions[:y][end];

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test);

    ensemble_mean, ensemble_std, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics
    )

    ensemble_metrics, ensemble_preds
end

# --- Plot all three predictions with confidence intervals + normalized influence ---
using LinearAlgebra: diag
using BayesBase: cov
using Random

Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test)
dim = 1  # which variable to plot (change to plot a different one)

# Extract marginal mean and std for a given dimension from ensemble predictions
function marginal_mean_std(preds, dim)
    μ = [mean(p)[dim] for p in preds]
    σ = [sqrt(cov(p)[dim, dim]) for p in preds]
    return μ, σ
end

normalized_samples = normalized_influence_dist(StableRNG(42), train_influence, 100)
n_f = size(train_influence, 1)
n_t = size(train_influence, 2)

# Compute mean and credible intervals of normalized influence
norm_influence_mean = dropdims(mean(normalized_samples, dims=1), dims=1)  # (n_f × n_t)
norm_influence_lo = zeros(n_f, n_t)
norm_influence_hi = zeros(n_f, n_t)
for i in 1:n_f, j in 1:n_t
    samples_ij = normalized_samples[:, i, j]
    norm_influence_lo[i, j] = quantile(samples_ij, 0.025)
    norm_influence_hi[i, j] = quantile(samples_ij, 0.975)
end
top_share = vec(maximum(norm_influence_mean; dims=1))

# Build forecaster labels
quantile_labels = ["q$(Int(round(q)))" for q in saved_selected_quantiles]
forecaster_labels = vcat(experts, quantile_labels)
if length(forecaster_labels) < n_f
    for k in (length(forecaster_labels)+1):n_f
        push!(forecaster_labels, "Expert $k")
    end
end
forecaster_labels = forecaster_labels[1:n_f]

# --- Create combined plot ---
begin
    y_true = Y_for_metrics[dim, :]
    t = 1:length(y_true)

    # Top subplot: predictions comparison
    p1 = plot(t, y_true, label="Ground Truth", color=:black, linewidth=2, legend=:topright,
        ylabel="Value", title="Predictions Comparison (dim=$dim)")

    for (name, preds, color) in [
        ("Saved Priors", train_ensemble_preds, :blue),
        ("Zero Init w", zero_init_ensemble_preds, :red),
        ("Full Default", full_default_ensemble_preds, :green),
    ]
        μ, σ = marginal_mean_std(preds, dim)
        plot!(p1, t, μ, label=name, color=color, linewidth=1.5)
        plot!(p1, t, μ .+ 1.96 .* σ, fillrange=μ .- 1.96 .* σ, fillalpha=0.15,
            linealpha=0, label="", color=color)
    end

    # Bottom-left subplot: normalized influence over time
    expert_colors = distinguishable_colors(n_f, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    p2 = plot(title="Gammas",
        ylabel="Influence (normalized)", xlabel="Time Step",
        legend=:outerright, legendfontsize=6)

    for i in 1:n_f
        plot!(p2, 1:n_t, norm_influence_mean[i, :],
            label=forecaster_labels[i], color=expert_colors[i], linewidth=1.5)
        plot!(p2, 1:n_t, norm_influence_hi[i, :],
            fillrange=norm_influence_lo[i, :],
            fillalpha=0.15, linealpha=0, label="", color=expert_colors[i])
    end

    # Bottom-right subplot: top-share (dominance of strongest expert)
    p3 = plot(
        1:n_t,
        top_share,
        title = "TopShare",
        xlabel = "Time Step",
        ylabel = "Top-1 normalized gamma",
        color = :darkblue,
        linewidth = 2,
        ylims = (0.0, 1.0),
        label = "max_i γᵢ / Σγ",
    )

    # Layout:
    # Prediction plot
    # Gammas | TopShare
    l = @layout [a{0.58h}; b c]
    p = plot(p1, p2, p3, layout=l, size=(1300, 900), margin=5Plots.mm)
end

savefig(p, "predictions_comparison.png")
display(p)
