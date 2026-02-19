export multivariate_ensemble_precision_model

@model function multivariate_ensemble_precision_model(n_forecasters, X, y, priors)
    local γ

    for i = 1:n_forecasters
        γ[i] ~ priors[:γ][i]
    end

    for i = 1:n_forecasters
        for j = 1:length(y)
            y[j] ~ MvNormalMeanScalePrecision(X[i, j], γ[i])
        end
    end
end
