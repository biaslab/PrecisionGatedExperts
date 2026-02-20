using ProbabilisticEnsembling
using YAML
using RxInfer

spec = ProbabilisticEnsembling._parse_spec(YAML.load_file("sessions/hierarchical/hierarchical_exchange_rate_96.yaml"))
(y_val, y_test, predictions_val, predictions_test, features_val, features_test) = ProbabilisticEnsembling.before_rxinfer(spec)

@show y_val[1]
@show predictions_val[1, 1]
@show features_val[1]

y_sampled_data = ProbabilisticEnsembling.SubsampledData(y_val, 1)
predictons_val_data = ProbabilisticEnsembling.SubsampledData(predictions_val, 1)
features_val_data = ProbabilisticEnsembling.SubsampledData(features_val, 1)

sample_y = RxInfer.get_data(y_sampled_data);
sample_pred = RxInfer.get_data(predictons_val_data);
sample_features = RxInfer.get_data(features_val_data);

@show y_sampled_data.rng
@show y_sampled_data.rng
@show y_sampled_data.rng == predictons_val_data.rng 
@show predictons_val_data.sampled_ids == y_sampled_data.sampled_ids
@show all(predictions_val[:, y_sampled_data.sampled_ids[1]] .≈ sample_pred)