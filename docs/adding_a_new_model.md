# Adding a New Model Type

This guide walks through adding a new RxInfer-based model type to the ensemble framework. The system uses Julia's multiple dispatch — you register a type and implement a handful of hooks; the generic pipeline handles everything else.

## Architecture Overview

```
model_types.jl          abstract type ModelType, _parse_model_type
shared_pipeline.jl      Univariate / Multivariate prediction types, metrics
model_specifier.jl      Generic pipeline (run_experiment, predict_with_model) + hook defaults
<name>/                 Per-model directory: type + univariate/multivariate @model + hooks
```

The generic pipeline in `model_specifier.jl` calls hook functions dispatched on `(prediction_type, model_type)`. You only override what differs from the defaults.

## Step-by-Step

### 1. Create the directory structure

```
src/model_zoo/<name>/
    model_type.jl    # struct, dispatch, priors
    univariate.jl    # univariate @model (and optionally @constraints, @initialization)
    multivariate.jl  # multivariate @model variant (optional)
    pipeline.jl      # hook overrides
```

### 2. Define the model type (`model_type.jl`)

```julia
struct MyModel <: ModelType end

# Register for Val-dispatch (used by YAML config parsing)
create_based_on_symbol(::Val{:mymodel}) = MyModel()

# Human-readable name (used in logging and file names)
model_type_name(::MyModel) = "mymodel"

# Parse YAML priors config into a Dict{Symbol,Any}
function parse_priors(::MyModel, cfg::Dict, n_forecasters::Int)
    priors = Dict{Symbol,Any}()
    # Example: per-forecaster gamma priors
    γ_cfg = cfg["γ"]
    priors[:γ] = [GammaShapeRate(γ_cfg["shape"], γ_cfg["rate"]) for _ in 1:n_forecasters]
    return priors
end

# Extract priors from a saved JLD2 for re-prediction.
# IMPORTANT: reconstruct distribution types explicitly — JLD2 may
# deserialize them as ReconstructedStatic instead of proper types.
function extract_prediction_priors(::MyModel, saved)
    γ_raw = saved["γ_posteriors"]
    γ = map(d -> GammaShapeRate(d.a, d.b), γ_raw)
    return Dict{Symbol,Any}(:γ => γ)
end
```

### 3. Define the RxInfer model (`<name>.jl`)

```julia
export univariate_mymodel

@model function univariate_mymodel(n_forecasters, X, y, priors)
    γ = randomvar(n_forecasters)
    for i in 1:n_forecasters
        γ[i] ~ priors[:γ][i]
    end
    # ... your generative model ...
    for t in eachindex(y)
        y[t] ~ NormalMeanPrecision(dot(X[:, t], γ), 1.0)
    end
end
```

If you need variational constraints or initialization:

```julia
@constraints function mymodel_constraints(priors, prediction)
    # ...
end

@initialization function mymodel_init(priors)
    # ...
end
```

For multivariate support, create `multivariate.jl` in the same directory with a separate `@model` function.

### 4. Implement pipeline hooks (`pipeline.jl`)

The generic pipeline calls these hooks. Override only what you need:

```julia
# REQUIRED — no default exists:

# Build the RxInfer model. Dispatch on (prediction_type, model_type).
build_rxinfer_model(::Univariate, ::MyModel, nf, n_obs, p) =
    univariate_mymodel(n_forecasters = nf, priors = p)
# Add multivariate variant if needed:
# build_rxinfer_model(::Multivariate, ::MyModel, nf, n_obs, p) = ...

# Assemble the named tuple of results to save.
# `posteriors` = training posteriors, `test_posteriors` = test-time posteriors.
function model_results(::Any, ::MyModel, posteriors, test_posteriors)
    γ_posteriors = posteriors[:γ]
    weights = mean.(γ_posteriors)
    weights ./= sum(weights)
    return (γ_posteriors = γ_posteriors, weights = weights)
end
```

```julia
# OPTIONAL — defaults exist in model_specifier.jl:

# How to pack data for infer(). Default: (y=y, features=features, predictions=predictions)
# Static overrides this to (y=y, X=predictions) because it ignores features.
build_training_data(::Any, ::MyModel, y, features, predictions) = (y = y, X = predictions)

# Constraints and initialization (default: nothing → unconstrained mean-field).
build_rxinfer_constraints(::Univariate, ::MyModel, priors, prediction) =
    mymodel_constraints(priors, prediction)
build_rxinfer_init(::Univariate, ::MyModel, priors) =
    mymodel_init(priors)

# Which posterior keys to extract after training. Default: (:γ,)
training_posterior_keys(::MyModel) = (:γ, :τ)

# Which keys become priors for prediction. Default: all training keys except :γ
prediction_prior_keys(::MyModel) = (:τ,)
```

### 5. Register includes (`ProbabilisticEnsembling.jl`)

Add your includes **after** the shared infrastructure and **in this order**:

```julia
# mymodel
include("model_zoo/mymodel/model_type.jl")
include("model_zoo/mymodel/univariate.jl")
# include("model_zoo/mymodel/multivariate.jl")  # if applicable
include("model_zoo/mymodel/pipeline.jl")
```

Order matters: `model_type.jl` first (defines the struct), then `@model` files (use the struct), then `pipeline.jl` (dispatches on the struct).

### 6. Add a YAML config

```yaml
params:
  prediction_type: univariate
  model_type: mymodel
  dataset: ETTh1
  dataset_path: /path/to/ETTh1.csv
  column: OT
  horizon: 96
  experts:
    - /path/to/expert1.jld2
  quantiles: [0.1, 0.9]
  priors:
    γ:
      shape: 1.0
      rate: 1.0
  inference_iterations: 10
  prediction_iterations: 3
```

Run with:
```bash
julia --project -e 'using ProbabilisticEnsembling; run_experiment("config.yaml")'
```

## Hook Reference

| Hook | Signature | Default | Required? |
|------|-----------|---------|-----------|
| `build_rxinfer_model` | `(pt, mt, n_forecasters, n_obs, priors)` | none | Yes |
| `model_results` | `(pt, mt, posteriors, test_posteriors)` | none | Yes |
| `build_training_data` | `(pt, mt, y, features, predictions)` | `(y=y, features=features, predictions=predictions)` | No |
| `build_rxinfer_constraints` | `(pt, mt, priors, prediction)` | `nothing` | No |
| `build_rxinfer_init` | `(pt, mt, priors)` | `nothing` | No |
| `training_posterior_keys` | `(mt)` | `(:γ,)` | No |
| `prediction_prior_keys` | `(mt)` | all training keys except `:γ` | No |
| `extract_prediction_priors` | `(mt, saved)` | none (defined in `model_type.jl`) | Yes |
| `parse_priors` | `(mt, cfg, n_forecasters)` | none (defined in `model_type.jl`) | Yes |

## Existing Models as Reference

- **Static** (`model_zoo/static/`) — simplest model. No constraints, no initialization, single `:γ` posterior. Good starting point.
- **Dynamic** (`model_zoo/dynamic/`) — state-space model with constraints and initialization. Shows how to override all optional hooks.
- **Noisy Experts** (`model_zoo/noisy_experts/`) — univariate-only model with noisy expert observations.
