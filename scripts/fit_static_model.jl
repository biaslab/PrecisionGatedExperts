using ProbabilisticEnsembling
using JLD2

for file in readdir("sessions/static")
    example = joinpath("sessions/static", file)
    ProbabilisticEnsembling.run_experiment(example)
end