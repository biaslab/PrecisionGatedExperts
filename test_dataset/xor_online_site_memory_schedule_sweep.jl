include("xor_relu_direct_projection_niterations.jl")

const BASELINE_MSE = 0.221875

struct GaussianNaturalTerm
    xi::Vector{Float64}
    precision::Matrix{Float64}
end

function zero_term(n_features)
    return GaussianNaturalTerm(zeros(n_features), zeros(n_features, n_features))
end

function natural_term(dist)
    return GaussianNaturalTerm(
        Vector{Float64}(weightedmean(dist)),
        Matrix{Float64}(invcov(dist)),
    )
end

function term_distribution(term::GaussianNaturalTerm)
    precision = Symmetric(0.5 .* (term.precision .+ term.precision'))
    return MvNormalWeightedMeanPrecision(term.xi, Matrix(precision))
end

add_terms(a::GaussianNaturalTerm, b::GaussianNaturalTerm) = GaussianNaturalTerm(
    a.xi .+ b.xi,
    a.precision .+ b.precision,
)

subtract_terms(a::GaussianNaturalTerm, b::GaussianNaturalTerm) = GaussianNaturalTerm(
    a.xi .- b.xi,
    a.precision .- b.precision,
)

damp_terms(old::GaussianNaturalTerm, new::GaussianNaturalTerm, alpha) = GaussianNaturalTerm(
    (1 - alpha) .* old.xi .+ alpha .* new.xi,
    (1 - alpha) .* old.precision .+ alpha .* new.precision,
)

function terms_from_priors(priors, n_neurons)
    return (
        w_mean = [natural_term(priors[:w_mean][k]) for k in 1:n_neurons],
        w_a = [natural_term(priors[:w_a][k]) for k in 1:n_neurons],
        tau = deepcopy(priors[:tau]),
    )
end

function priors_from_terms(terms, n_neurons)
    return Dict{Symbol,Any}(
        :w_mean => [term_distribution(terms.w_mean[k]) for k in 1:n_neurons],
        :w_a => [term_distribution(terms.w_a[k]) for k in 1:n_neurons],
        :tau => deepcopy(terms.tau),
    )
end

function zero_site(n_neurons, n_features)
    return (
        w_mean = [zero_term(n_features) for _ in 1:n_neurons],
        w_a = [zero_term(n_features) for _ in 1:n_neurons],
    )
end

function subtract_site(global_terms, site_terms, n_neurons)
    return (
        w_mean = [
            subtract_terms(global_terms.w_mean[k], site_terms.w_mean[k])
            for k in 1:n_neurons
        ],
        w_a = [
            subtract_terms(global_terms.w_a[k], site_terms.w_a[k])
            for k in 1:n_neurons
        ],
        tau = deepcopy(global_terms.tau),
    )
end

function add_site(cavity_terms, site_terms, tau, n_neurons)
    return (
        w_mean = [
            add_terms(cavity_terms.w_mean[k], site_terms.w_mean[k])
            for k in 1:n_neurons
        ],
        w_a = [
            add_terms(cavity_terms.w_a[k], site_terms.w_a[k])
            for k in 1:n_neurons
        ],
        tau = deepcopy(tau),
    )
end

function posterior_site_terms(posterior_terms, cavity_terms, n_neurons)
    return (
        w_mean = [
            subtract_terms(posterior_terms.w_mean[k], cavity_terms.w_mean[k])
            for k in 1:n_neurons
        ],
        w_a = [
            subtract_terms(posterior_terms.w_a[k], cavity_terms.w_a[k])
            for k in 1:n_neurons
        ],
    )
end

function damp_site_terms(old_site, new_site, alpha_mean, alpha_gate, n_neurons)
    return (
        w_mean = [
            damp_terms(old_site.w_mean[k], new_site.w_mean[k], alpha_mean)
            for k in 1:n_neurons
        ],
        w_a = [
            damp_terms(old_site.w_a[k], new_site.w_a[k], alpha_gate)
            for k in 1:n_neurons
        ],
    )
end

function make_online_metrics()
    return DataFrame(
        update = Int[],
        epoch = Int[],
        batch = Int[],
        batch_size = Int[],
        free_energy = Float64[],
        test_mse = Float64[],
        invalid_predictions = Int[],
        mean_gate_mass = Float64[],
        min_gate_mass = Float64[],
        min_var_w_mean = Float64[],
        median_var_w_mean = Float64[],
        max_var_w_mean = Float64[],
        min_var_w_a = Float64[],
        median_var_w_a = Float64[],
        max_var_w_a = Float64[],
        max_precision_w_mean = Float64[],
        max_precision_w_a = Float64[],
        mean_w_mean_slope_norm = Float64[],
        max_w_mean_slope_norm = Float64[],
        mean_w_a_slope_norm = Float64[],
        max_w_a_slope_norm = Float64[],
        mean_abs_w_a_bias = Float64[],
        w_a_bias_slope_ratio = Float64[],
        gate_active_fraction = Float64[],
        mean_effective_gate_count = Float64[],
    )
end

function tail_rows(df, n)
    first_row = max(1, nrow(df) - n + 1)
    return df[first_row:end, :]
end

function best_row_after(metrics, first_update)
    rows = metrics[metrics.update .>= first_update, :]
    if nrow(rows) == 0
        return nothing
    end

    return best_metric_row(rows)
end

function scheduled_summary_dataframe()
    return DataFrame(
        name = String[],
        epochs = Int[],
        updates = Int[],
        batch_size = Int[],
        batch_iterations = Int[],
        examples_seen = Int[],
        inner_steps = Int[],
        obs_precision = Float64[],
        alpha_mean = Float64[],
        alpha_gate = Float64[],
        schedule = String[],
        final_mse = Float64[],
        best_mse = Float64[],
        best_update = Union{Missing,Int}[],
        best_after10_mse = Float64[],
        best_after10_update = Union{Missing,Int}[],
        tail20_mean_mse = Float64[],
        tail20_min_mse = Float64[],
        tail20_std_mse = Float64[],
        tail20_below_baseline_fraction = Float64[],
        final_min_var_w_mean = Float64[],
        final_min_var_w_a = Float64[],
        final_median_var_w_a = Float64[],
        final_mean_w_mean_slope_norm = Float64[],
        final_mean_w_a_slope_norm = Float64[],
        final_gate_active_fraction = Float64[],
        final_effective_gate_count = Float64[],
        metrics_path = String[],
        log_path = String[],
    )
end

function print_case_result(row)
    @printf(
        "updates=%d final=%.6f best=%.6f@%s best10=%.6f tail20=%.6f below_tail=%.2f minvar_a=%.3g slope_a=%.4g gates=%.2f\n",
        row.updates,
        row.final_mse,
        row.best_mse,
        string(row.best_update),
        row.best_after10_mse,
        row.tail20_mean_mse,
        row.tail20_below_baseline_fraction,
        row.final_min_var_w_a,
        row.final_mean_w_a_slope_norm,
        row.final_effective_gate_count,
    )
end

function scheduled_alphas(case, epoch)
    if case.schedule == "constant"
        return case.alpha_mean, case.alpha_gate
    elseif case.schedule == "sqrt_epoch"
        scale = 1 / sqrt(epoch)
        return case.alpha_mean * scale, case.alpha_gate * scale
    elseif case.schedule == "inv_epoch"
        scale = 1 / epoch
        return case.alpha_mean * scale, case.alpha_gate * scale
    elseif case.schedule == "drop_after5"
        scale = epoch <= 5 ? 1.0 : 0.1
        return case.alpha_mean * scale, case.alpha_gate * scale
    elseif case.schedule == "freeze_after5"
        scale = epoch <= 5 ? 1.0 : 0.0
        return case.alpha_mean * scale, case.alpha_gate * scale
    end

    error("Unknown schedule: $(case.schedule)")
end

function run_scheduled_site_memory_case(case)
    n_neurons = 16
    n_features = 3
    df = CSV.read("test_dataset/xor_simple_dataset.csv", DataFrame)
    df_train, df_test = split_dataset(df; train_fraction = 0.3, seed = 2027)
    features_train = build_features(df_train)
    features_test = build_features(df_test)
    y_test = df_test.OT

    initial_priors = make_priors(
        n_neurons = n_neurons,
        seed = 42,
        parameterization = "weighted",
        w_mean_prior_precision = 1e-4,
        w_a_prior_precision = 1e-3,
    )

    n_train = nrow(df_train)
    batches = [
        collect(first_idx:min(first_idx + case.batch_size - 1, n_train))
        for first_idx in 1:case.batch_size:n_train
    ]
    site_terms = [zero_site(n_neurons, n_features) for _ in batches]
    global_terms = terms_from_priors(initial_priors, n_neurons)
    global_priors = priors_from_terms(global_terms, n_neurons)
    metrics = make_online_metrics()
    w_mean_history = Any[]
    w_a_history = Any[]
    update = 0

    for epoch in 1:case.epochs
        alpha_mean_epoch, alpha_gate_epoch = scheduled_alphas(case, epoch)
        for (batch_number, batch_indices) in enumerate(batches)
            update += 1
            cavity_terms = subtract_site(global_terms, site_terms[batch_number], n_neurons)
            cavity_priors = priors_from_terms(cavity_terms, n_neurons)
            batch_features = features_train[batch_indices]
            batch_y = Vector(df_train.OT[batch_indices])

            result = infer(
                model = xor_relu_direct_projection_niterations(
                    n_neurons = n_neurons,
                    priors = cavity_priors,
                    obs_precision = case.obs_precision,
                ),
                data = (y = batch_y, features = batch_features),
                constraints = xor_relu_direct_projection_niterations_constraints(1),
                initialization = xor_relu_direct_projection_niterations_init(cavity_priors),
                iterations = case.batch_iterations,
                free_energy = true,
                showprogress = false,
                options = (limit_stack_depth = 100,),
            )

            posterior_priors = posterior_priors_from_result(result)
            posterior_terms = terms_from_priors(posterior_priors, n_neurons)
            new_site = posterior_site_terms(posterior_terms, cavity_terms, n_neurons)
            site_terms[batch_number] = damp_site_terms(
                site_terms[batch_number],
                new_site,
                alpha_mean_epoch,
                alpha_gate_epoch,
                n_neurons,
            )
            global_terms = add_site(
                cavity_terms,
                site_terms[batch_number],
                posterior_priors[:tau],
                n_neurons,
            )
            global_priors = priors_from_terms(global_terms, n_neurons)

            w_means, w_as = weights_from_priors(global_priors, n_neurons)
            push!(w_mean_history, w_means)
            push!(w_a_history, w_as)
            fe = isempty(result.free_energy) ? NaN : Float64(result.free_energy[end])
            append_online_metric!(
                metrics,
                update,
                epoch,
                batch_number,
                length(batch_indices),
                fe,
                features_test,
                y_test,
                w_means,
                w_as,
                global_priors,
            )
        end
    end

    return global_priors, metrics, w_mean_history, w_a_history
end

function quiet_scheduled_site_case(case)
    open(case.log_path, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                final_priors, metrics, _, _ = run_scheduled_site_memory_case(case)
                CSV.write("$(case.output_prefix)_metrics.csv", metrics)
                w_means_final, w_as_final = weights_from_priors(final_priors, 16)
                println("Final weights:")
                for k in 1:16
                    println("  neuron $k: w_mean=$(round.(w_means_final[k]; digits = 4))  w_a=$(round.(w_as_final[k]; digits = 4))")
                end
            end
        end
    end
end

function summarize_scheduled_site_case!(summary, case)
    metrics_path = "$(case.output_prefix)_metrics.csv"
    metrics = CSV.read(metrics_path, DataFrame)
    best_row = best_metric_row(metrics)
    best_after10 = best_row_after(metrics, 10)
    final_row = metrics[end, :]
    tail = tail_rows(metrics, 20)

    push!(summary, (
        name = case.name,
        epochs = case.epochs,
        updates = nrow(metrics),
        batch_size = case.batch_size,
        batch_iterations = case.batch_iterations,
        examples_seen = nrow(metrics) * case.batch_size,
        inner_steps = nrow(metrics) * case.batch_iterations,
        obs_precision = Float64(case.obs_precision),
        alpha_mean = Float64(case.alpha_mean),
        alpha_gate = Float64(case.alpha_gate),
        schedule = case.schedule,
        final_mse = Float64(final_row.test_mse),
        best_mse = isnothing(best_row) ? NaN : Float64(best_row.test_mse),
        best_update = isnothing(best_row) ? missing : Int(best_row.update),
        best_after10_mse = isnothing(best_after10) ? NaN : Float64(best_after10.test_mse),
        best_after10_update = isnothing(best_after10) ? missing : Int(best_after10.update),
        tail20_mean_mse = mean(tail.test_mse),
        tail20_min_mse = minimum(tail.test_mse),
        tail20_std_mse = std(tail.test_mse),
        tail20_below_baseline_fraction = mean(tail.test_mse .< BASELINE_MSE),
        final_min_var_w_mean = Float64(final_row.min_var_w_mean),
        final_min_var_w_a = Float64(final_row.min_var_w_a),
        final_median_var_w_a = Float64(final_row.median_var_w_a),
        final_mean_w_mean_slope_norm = Float64(final_row.mean_w_mean_slope_norm),
        final_mean_w_a_slope_norm = Float64(final_row.mean_w_a_slope_norm),
        final_gate_active_fraction = Float64(final_row.gate_active_fraction),
        final_effective_gate_count = Float64(final_row.mean_effective_gate_count),
        metrics_path = metrics_path,
        log_path = case.log_path,
    ))

    return summary
end

function main()
    output_dir = "docs/bong_online_vmp/results/site_memory_schedule"
    mkpath(output_dir)

    common = (
        epochs = 10,
        batch_size = 24,
        batch_iterations = 5,
        obs_precision = 1e3,
        alpha_mean = 0.5,
        alpha_gate = 0.05,
    )
    schedules = ["sqrt_epoch", "inv_epoch", "drop_after5", "freeze_after5"]

    cases = NamedTuple[]
    for schedule in schedules
        name = "schedule_site_b24_vmp5_amean0p5_agate0p05_$(schedule)_epochs10"
        push!(cases, merge(common, (
            name = name,
            schedule = schedule,
            output_prefix = "$output_dir/$name",
            log_path = "$output_dir/$name.log",
        )))
    end

    summary = scheduled_summary_dataframe()

    println("=" ^ 78)
    println("XOR scheduled site-memory sweep")
    println("  projection_iterations=1")
    println("  fixed batches, epochs=$(common.epochs), batch_size=$(common.batch_size)")
    println("  base alpha_mean=$(common.alpha_mean), alpha_gate=$(common.alpha_gate)")
    println("  schedules=$(schedules)")
    println("=" ^ 78)

    for (i, case) in enumerate(cases)
        print("[$i/$(length(cases))] $(case.name): ")
        flush(stdout)
        quiet_scheduled_site_case(case)
        summarize_scheduled_site_case!(summary, case)
        print_case_result(summary[end, :])
        CSV.write("$output_dir/summary_partial.csv", summary)
    end

    sort!(summary, :tail20_mean_mse)
    summary_path = "$output_dir/summary.csv"
    CSV.write(summary_path, summary)

    println("\nScheduled site-memory summary saved to $summary_path")
    show(summary[:, [
        :schedule,
        :batch_size,
        :batch_iterations,
        :alpha_mean,
        :alpha_gate,
        :final_mse,
        :best_mse,
        :best_update,
        :best_after10_mse,
        :tail20_mean_mse,
        :tail20_below_baseline_fraction,
        :final_mean_w_a_slope_norm,
        :final_effective_gate_count,
    ]], allcols = true, allrows = true)
    println()

    best_case = cases[findfirst(==(summary.name[1]), [case.name for case in cases])]
    visual_prefix = "$output_dir/best_sqrt_site_b24_amean0p5_agate0p05"
    visual_case = merge(best_case, (
        output_prefix = visual_prefix,
        log_path = "$visual_prefix.log",
    ))

    println("\nRe-running best case to save heatmap...")
    final_priors, best_metrics, w_mean_history, w_a_history = run_scheduled_site_memory_case(visual_case)
    CSV.write("$(visual_prefix)_metrics.csv", best_metrics)

    df = CSV.read("test_dataset/xor_simple_dataset.csv", DataFrame)
    _, df_test = split_dataset(df; train_fraction = 0.3, seed = 2027)
    features_test = build_features(df_test)
    visual_config = Dict{Symbol,Any}(
        :output_prefix => visual_prefix,
        :projection_niterations => 1,
        :save_animation => false,
    )
    save_visuals_from_history(
        w_mean_history,
        w_a_history,
        visual_config,
        features_test;
        frame_label = "site update",
    )

    final_row = best_metrics[end, :]
    best_row = best_metric_row(best_metrics)
    @printf("Best visual final MSE = %.6f\n", final_row.test_mse)
    if !isnothing(best_row)
        @printf("Best visual best MSE = %.6f at update %d\n", best_row.test_mse, best_row.update)
    end

    w_means, w_as = weights_from_priors(final_priors, 16)
    geometry = weight_geometry_metrics(w_means, w_as, features_test)
    @printf("Best visual mean gate slope norm = %.6f\n", geometry.mean_w_a_slope_norm)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
