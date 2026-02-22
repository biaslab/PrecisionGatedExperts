using ProbabilisticEnsembling
using ExponentialFamily
using JLD2

saved = JLD2.load("/Users/ruiite/projects/prob_ensem_mle/probabilistic_ensemble_forecasting/final_results/ETTh2_h336_OT_probabilisticensembling.static_11370337655097017902.jld2")
spec_saved = saved["spec"]

prediction_type = ProbabilisticEnsembling._parse_saved_prediction_type(string(spec_saved.prediction_type))
model_type = ProbabilisticEnsembling._parse_saved_model_type(string(spec_saved.model_type))
column = isnothing(spec_saved.column) ? nothing : String(spec_saved.column)
dataset = ProbabilisticEnsembling._dataset_val(spec_saved.dataset)
dataset_path = String(spec_saved.dataset_path)
experts = String.(spec_saved.experts)
prediction_iterations = 1

spec_for_data = ProbabilisticEnsembling.ExperimentSpecifier(
    prediction_type,
    model_type,
    column,
    Int(spec_saved.horizon),
    dataset,
    dataset_path,
    experts,
    [10.0, 90.0],
    2,
    Dict{Symbol,Any}(),
    1,
    prediction_iterations,
    false,
    nothing,
    nothing,
)

_, y_test_all, _, predictions_test_all, _, features_test_all = ProbabilisticEnsembling.before_rxinfer(spec_for_data);
n_steps = length(y_test_all)

y_test = ProbabilisticEnsembling.prepare_y_test(prediction_type, y_test_all, n_steps);
predictions_test = predictions_test_all;
features_test = features_test_all;

n_forecasters = size(predictions_test, 1);
prediction_array = [missing for _ = 1:n_steps]

train_results, train_preds = begin
    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved, 1.0);
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
    #@info [mean(posterior) for posterior in infer_test.posteriors[:γ][end]][1:5]
    ensemble_preds = infer_test.predictions[:y][end];

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test);

    ensemble_mean, ensemble_std, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics
    )

    ensemble_metrics, ensemble_preds
end;

zero_init = begin
    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved, 1.0);
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

    #@info [mean(posterior) for posterior in infer_test.posteriors[:γ][end]][1:5]
    ensemble_preds = infer_test.predictions[:y][end];

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test);

    ensemble_mean, ensemble_std, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics
    )
end

full_default = begin
    priors = ProbabilisticEnsembling.extract_prediction_priors(model_type, saved, 1.0);
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

    #@info [mean(posterior) for posterior in infer_test.posteriors[:γ][end]][1:5]
    ensemble_preds = infer_test.predictions[:y][end];

    Y_for_metrics = ProbabilisticEnsembling.prepare_y_for_metrics(prediction_type, y_test);

    ensemble_mean, ensemble_std, ensemble_metrics = ProbabilisticEnsembling.compute_ensemble_metrics(
        spec_for_data.prediction_type,
        ensemble_preds,
        Y_for_metrics
    )
end


@info train_results
@info zero_init.ensemble_metrics
@info full_default.ensemble_metrics
