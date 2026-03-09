using ProbabilisticEnsembling
using YAML
using JLD2
using Plots
using Statistics
using Distributions

# ──────────────────────────────────────────────────────────────
# 1. Load trained model from final_results/ (no re-training)
# ──────────────────────────────────────────────────────────────
config_path = length(ARGS) >= 1 ? ARGS[1] : "sessions/noisy_experts/test_ETTh1_96_one_expert.yaml"
@info "Using config: $config_path"
config = YAML.load_file(config_path)
spec = ProbabilisticEnsembling._parse_spec(config)

# Find the saved .jld2 by hash
ds_name = typeof(spec.dataset).parameters[1]
model_name = ProbabilisticEnsembling.model_type_name(spec.model_type)
config_hash = hash(config)
if spec.prediction_type isa ProbabilisticEnsembling.Univariate
    fname = "$(ds_name)_h$(spec.horizon)_$(spec.column)_$(model_name)_$(config_hash).jld2"
else
    fname = "$(ds_name)_h$(spec.horizon)_multivariate_$(model_name)_$(config_hash).jld2"
end
results_path = joinpath("final_results", fname)

if !isfile(results_path)
    @info "No saved model at $results_path — listing available files:"
    for f in readdir("final_results")
        println("  $f")
    end
    error("Run `run_experiment(\"$config_path\")` first to train and save the model.")
end

@info "Loading saved model" path = results_path
saved = JLD2.load(results_path)

# Reconstruct spec for data loading and extract priors
pt, mt = spec.prediction_type, spec.model_type
prediction_priors = ProbabilisticEnsembling.extract_prediction_priors(mt, saved)

_, y_test, _, predictions_test, _, features_test =
    ProbabilisticEnsembling.before_rxinfer(spec)

n_forecasters = size(predictions_test, 1)
n_test = length(y_test)

mean_κ = [round(mean(p); digits=3) for p in prediction_priors[:κ]]
@info "Loaded model" n_forecasters n_test mean_κ

# ──────────────────────────────────────────────────────────────
# 2. Run prediction with max iterations, collect ALL iterations
# ──────────────────────────────────────────────────────────────
max_iters = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5

pred_model = ProbabilisticEnsembling.univariate_noisy_experts_prediction(
    n_forecasters = n_forecasters,
    n_obs = n_test,
    priors = prediction_priors,
)
pred_data = (features = features_test, predictions = predictions_test)
pred_constraints = ProbabilisticEnsembling.univariate_noisy_experts_constraints(prediction_priors, true)
pred_init = ProbabilisticEnsembling.univariate_noisy_experts_init(prediction_priors)

@info "Running prediction with $max_iters iterations..."
using RxInfer
pred_result = infer(;
    model = pred_model,
    data = pred_data,
    constraints = pred_constraints,
    initialization = pred_init,
    iterations = max_iters,
    free_energy = false,
    showprogress = true,
)

# pred_result.posteriors[:y] is a Vector of length max_iters
# each element is a Vector{Distribution} of length n_test

# ──────────────────────────────────────────────────────────────
# 3. Extract mean and std per iteration
# ──────────────────────────────────────────────────────────────
n_iters_to_show = min(max_iters, length(pred_result.posteriors[:y]))

means_per_iter = []
stds_per_iter = []
vars_per_iter = []

for it in 1:n_iters_to_show
    y_dists = pred_result.posteriors[:y][it]
    push!(means_per_iter, map(mean, y_dists))
    push!(stds_per_iter, map(std, y_dists))
    push!(vars_per_iter, map(var, y_dists))
end

# ──────────────────────────────────────────────────────────────
# 3b. Also track γ and pred per iteration
# ──────────────────────────────────────────────────────────────
γ_means_per_iter = []
pred_vars_per_iter = []
for it in 1:n_iters_to_show
    if haskey(pred_result.posteriors, :γ)
        γ_dists = pred_result.posteriors[:γ][it]
        push!(γ_means_per_iter, map(mean, γ_dists))
    end
    if haskey(pred_result.posteriors, :pred)
        pred_dists = pred_result.posteriors[:pred][it]
        push!(pred_vars_per_iter, map(var, pred_dists))
    end
end

# ──────────────────────────────────────────────────────────────
# 4. Summary statistics per iteration
# ──────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("Iteration convergence summary ($n_forecasters experts, n_test=$n_test)")
println("=" ^ 70)
for it in 1:n_iters_to_show
    avg_std = mean(stds_per_iter[it])
    avg_var = mean(vars_per_iter[it])
    med_std = median(stds_per_iter[it])
    mse_it = mean((means_per_iter[it] .- y_test).^2)
    ci_width = 2 * 1.96 * avg_std

    extra = ""
    if !isempty(γ_means_per_iter)
        avg_γ = mean(γ_means_per_iter[it])
        extra *= "  avg_γ=$(round(avg_γ; digits=6))  1/γ=$(round(1/avg_γ; digits=4))"
    end
    if !isempty(pred_vars_per_iter)
        avg_pred_var = mean(pred_vars_per_iter[it])
        extra *= "  avg_var_pred=$(round(avg_pred_var; digits=4))"
    end

    println("  iter $it: avg_var_y=$(round(avg_var; digits=4))  " *
            "ci95=$(round(ci_width; digits=4))  " *
            "mse=$(round(mse_it; digits=4))" * extra)
end

# ──────────────────────────────────────────────────────────────
# 5. Animated plot: predictions + CI over iterations
#    (styled exactly like upper panel of scripts/compare_models.jl)
# ──────────────────────────────────────────────────────────────
t = 1:n_test

# Compute MSE per iteration for display
mse_per_iter = [mean((means_per_iter[it] .- y_test).^2) for it in 1:n_iters_to_show]

κ_str = join(["κ$i=$(mean_κ[i])" for i in 1:n_forecasters], ", ")

# Fixed axis limits across all frames
all_μ = vcat(means_per_iter...)
all_σ = vcat(stds_per_iter...)
y_lo = min(minimum(y_test), minimum(all_μ .- 1.96 .* all_σ))
y_hi = max(maximum(y_test), maximum(all_μ .+ 1.96 .* all_σ))
y_margin = 0.05 * (y_hi - y_lo)
y_lo -= y_margin
y_hi += y_margin

anim = @animate for it in 1:n_iters_to_show
    μ = means_per_iter[it]
    σ = stds_per_iter[it]
    mse_it = round(mse_per_iter[it]; digits=4)
    avg_s = round(mean(σ); digits=3)

    # Ground truth — black line (same as compare_models.jl)
    plot(t, y_test, label="Ground Truth", color=:black, linewidth=2,
        legend=:topright, ylabel="Value", xlabel="Time Step",
        title="Noisy Experts ($n_forecasters experts, $κ_str) — iteration $it / $n_iters_to_show",
        size=(2400, 800), dpi=150, margin=6Plots.mm,
        xlims=(1, n_test), ylims=(y_lo, y_hi))

    # 95% CI band
    plot!(t, μ .+ 1.96 .* σ, fillrange=μ .- 1.96 .* σ,
        fillalpha=0.15, linealpha=0, label="", color=:blue)

    # Mean prediction
    plot!(t, μ, label="Noisy Experts (MSE=$mse_it, avg_σ=$avg_s)",
        color=:blue, linewidth=1.5)
end

gif(anim, "predictive_iteration_convergence.gif", fps=2)
@info "Animation saved to predictive_iteration_convergence.gif"

# ──────────────────────────────────────────────────────────────
# 6. NLL per iteration plot
#    NLL_i = -mean[ log p(y_true | μ_i, σ²_i) ]
#          = 0.5*log(2π) + mean[ 0.5*log(σ²) + 0.5*(y-μ)²/σ² ]
# ──────────────────────────────────────────────────────────────
nll_per_iter = Float64[]
for it in 1:n_iters_to_show
    μ = means_per_iter[it]
    σ² = vars_per_iter[it]
    nll = mean(0.5 .* log.(2π) .+ 0.5 .* log.(σ²) .+ 0.5 .* (y_test .- μ).^2 ./ σ²)
    push!(nll_per_iter, nll)
end

p_nll = plot(1:n_iters_to_show, nll_per_iter,
    xlabel="Prediction iteration", ylabel="NLL",
    title="Negative Log-Likelihood vs iteration ($n_forecasters experts)",
    marker=:circle, markersize=6, linewidth=2, color=:blue,
    legend=false, size=(800, 500), dpi=150, margin=6Plots.mm)

# Annotate each point with its value
for (it, nll) in enumerate(nll_per_iter)
    annotate!(p_nll, it, nll, text("$(round(nll; digits=3))", :top, 9))
end

savefig(p_nll, "predictive_nll_convergence.png")
@info "NLL plot saved to predictive_nll_convergence.png"
println("\nNLL per iteration: ", [round(v; digits=4) for v in nll_per_iter])
