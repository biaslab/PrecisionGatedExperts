using ProbabilisticEnsembling

const NOIS_SESS = [
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_exchange_rate_336.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh2_96.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_exchange_rate_720.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh2_192.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_exchange_rate_192.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh1_336.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh1_96.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh2_336.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh2_720.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_exchange_rate_96.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh1_192.yaml",
    "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_ETTh1_720.yaml",
    # "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_electricity_192.yaml",
    # "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_electricity_336.yaml",
    # "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_electricity_720.yaml",
    # "/Users/ruiite/projects/new_prob/probabilistic_ensemble_forecasting/sessions/noisy_experts/vae/noisy_experts_electricity_96.yaml",
]

const SESSIONS = [
    # "sessions/neural_ensemble/neural_ensemble_ETTh1_96.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh1_192.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh1_336.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh1_720.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh2_96.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh2_192.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh2_336.yaml",
    # "sessions/neural_ensemble/neural_ensemble_ETTh2_720.yaml",
    # "sessions/neural_ensemble/neural_ensemble_exchange_rate_96.yaml",
    # "sessions/neural_ensemble/neural_ensemble_exchange_rate_192.yaml",
    # "sessions/neural_ensemble/neural_ensemble_exchange_rate_336.yaml",
    # "sessions/neural_ensemble/neural_ensemble_exchange_rate_720.yaml",
    # "sessions/neural_ensemble/neural_ensemble_electricity_96.yaml",
    # "sessions/neural_ensemble/neural_ensemble_electricity_192.yaml",
    # "sessions/neural_ensemble/neural_ensemble_electricity_336.yaml",
    # "sessions/neural_ensemble/neural_ensemble_electricity_720.yaml",
    "sessions/neural_ensemble/neural_ensemble_traffic_96.yaml",
    "sessions/neural_ensemble/neural_ensemble_traffic_192.yaml",
    "sessions/neural_ensemble/neural_ensemble_traffic_336.yaml",
    "sessions/neural_ensemble/neural_ensemble_traffic_720.yaml",]

# for session in NOIS_SESS
#     @info "noise"
#     run_experiment(session)
# end

for session in SESSIONS
    @info "Training" session
    run_neural_ensemble_experiment(session)
    @info "Done" session
end

# After training, copy results to paper/results/neural_ensemble/ with correct names:
#   ETTh1_h{H}_OT_neural_ensemble.jld2
#   exchange_rate_h{H}_multivariate_neural_ensemble.jld2
@info "All training complete. Copy results from saved_neural_ensemble_models/ to paper/results/neural_ensemble/"
