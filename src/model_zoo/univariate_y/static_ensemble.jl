export univariate_ensemble_precision_model

@model function univariate_ensemble_precision_model(n_forecasters, X, y, priors)
    local γ

    for i = 1:n_forecasters
        γ[i] ~ priors[:γ][i]
    end

    for i = 1:n_forecasters
        for j = 1:length(y)
            y[j] ~ NormalMeanPrecision(X[i, j], γ[i])
        end
    end
end
