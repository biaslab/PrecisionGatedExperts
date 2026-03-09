# Dynamic Diagonal Model — Design Rationale

## Motivation

The existing `Dynamic` model maintains a full-rank MvNormal posterior over the
weight vector `w ∈ ℝⁿ` (n = number of forecasters). Each observation generates a
rank-1 `LowRankNormalWeightedMeanPrecision` message that is folded into a dense
`MvNormalWeightedMeanPrecision` via an in-place BLAS.syr! rank-1 update. This
means the posterior precision `Λ` is a dense n × n matrix that is allocated once
per inference iteration and updated T times (T = number of observations).

For moderate n this is fine, but the dense precision carries O(n²) storage and
every downstream operation (mean recovery, KL computation, message products)
touches the full matrix. The `dynamic_diagonal` model replaces the dense
posterior with a **diagonal-precision approximation** that stays O(n) throughout,
never allocating an n × n matrix.

## Mathematical Foundation

### The projection step

Given a multivariate normal q with full precision Λ and mean μ, the
KL-optimal projection onto the diagonal-precision family minimizes

    KL[q_diag ‖ q_full]

over all q_diag = N(m, diag(d)⁻¹). The closed-form solution is
(Bishop, 2006, §10.1.2):

- **mean**: m = μ  (unchanged)
- **precision diagonal**: d = diag(Λ)  (drop off-diagonal entries)

In the weighted-mean-precision parametrization (η, Λ) where η = Λμ, one
**cannot** simply take diag(η). The weighted mean η_i = Σ_j Λ_{ij} μ_j bakes in
off-diagonal couplings, so after extracting d = diag(Λ) one must recompute
η_new = d ⊙ μ, which requires recovering μ = Λ⁻¹η first.

### Sequential diagonal updates (Assumed Density Filtering)

Rather than building the full posterior and projecting once, we process
observations one at a time and project to diagonal after each rank-1 update.
This is **Assumed Density Filtering** (ADF), the single-pass limit of
Expectation Propagation (Minka, 2001).

State: diagonal precision vector d ∈ ℝⁿ and mean μ ∈ ℝⁿ.

For observation i with feature vector x_i ∈ ℝⁿ, precision scalar γ_i, and
weighted-mean contribution ξ_i = γ_i x_i y_i:

```
# 1. Form natural parameters of (diagonal + rank-1) posterior
η = d ⊙ μ + ξ_i                           # O(n)

# 2. Recover exact mean via Sherman–Morrison on D + γ_i x_i x_iᵀ
u  = x_i ./ d                              # D⁻¹ x,  O(n)
s  = 1/γ_i + dot(x_i, u)                  # scalar,  O(n)
z  = η ./ d                                # D⁻¹ η,  O(n)
μ  = z - u * (dot(x_i, z) / s)            # SM step, O(n)

# 3. Update diagonal precision
d  = d + γ_i * (x_i .⊙ x_i)              # diag(Λ + γ xxᵀ), O(n)
```

Total for T observations: **O(Tn) time, O(n) memory**. No n × n matrix is ever
formed.

### Why the precision diagonal is exact

The diagonal of a sum equals the sum of diagonals:

    diag(D₀ + Σ_i γ_i x_i x_iᵀ) = d₀ + Σ_i γ_i (x_i ⊙ x_i)

This holds regardless of processing order — no information is lost in the
precision diagonal. The approximation enters only through the mean, where each
intermediate projection discards the off-diagonal information needed for the
exact matrix solve.

### Relation to Expectation Propagation

ADF is a single forward pass. If accuracy of the mean matters, one can iterate
the pass multiple times to converge to the EP fixed point (Minka, 2001). In
practice, for ensemble weight learning with n ≈ 5–20, the ADF approximation is
already good because:

1. The prior is isotropic (MvNormalMeanScalePrecision), so the initial precision
   is already diagonal.
2. Expert predictions are often mildly correlated, keeping off-diagonal
   precision entries small relative to the diagonal.
3. The softdot likelihood generates rank-1 updates, so each individual update
   introduces limited off-diagonal structure.

## Implementation Strategy

### Accumulator: `MvNormalWeightedMeanPrecision` with `Diagonal` precision

No new distribution type is needed. Julia's `Diagonal{T,Vector{T}}` from
`LinearAlgebra` is already O(n) storage, and `MvNormalWeightedMeanPrecision` is
parametric on its matrix type:

```julia
MvNormalWeightedMeanPrecision{T, Vector{T}, Diagonal{T, Vector{T}}}
```

This reuses all existing BayesBase/ExponentialFamily infrastructure (mean, cov,
logdetcov, entropy, etc.) — `Diagonal` already dispatches efficiently for all of
these. The accumulator is just a `MvNormalWeightedMeanPrecision` whose precision
happens to be `Diagonal` instead of `Matrix`.

### Dispatch via prior type — no new meta or message types

The key simplification: no new meta type or message type is needed. The existing
`LowRankMeta` and `LowRankNormalWeightedMeanPrecision` (LR) messages are reused
unchanged. The dispatch to the diagonal path is controlled entirely by the
**prior type** of `w`:

- `Dynamic` model: w prior is `MvNormalMeanScalePrecision(zeros(n), scale)`
  → first product `MvNormalMeanScalePrecision × LR` creates dense `Matrix` accumulator
- `DynamicDiagonal` model: w prior is `MvNormalWeightedMeanPrecision(zeros(n), Diagonal(fill(scale, n)))`
  → first product `MvNormalWeightedMeanPrecision{..., Diagonal} × LR` fires the diagonal rule

The existing dense product rule dispatches on `M<:Matrix`:

```julia
function BayesBase.prod(..., left::MvNormalWeightedMeanPrecision{T,V,M}, right::LR) where {T,V,M<:Matrix}
```

The new diagonal rule dispatches on `M<:Diagonal`:

```julia
function BayesBase.prod(..., left::MvNormalWeightedMeanPrecision{T,V,M}, right::LR) where {T,V,M<:Diagonal}
```

No ambiguity, no new types — just one new product rule.

### Product rule

| Left | Right | Result | Cost |
|------|-------|--------|------|
| `MvNormalWeightedMeanPrecision{..., Diagonal}` | `LR` | same (in-place) | O(n) — Sherman–Morrison projection |

Each LR message updates the accumulator in-place:

1. Accumulate weighted mean: `xi .+= right.xi`
2. Recover the current mean via Sherman–Morrison (O(n) since Λ is Diagonal)
3. Update diagonal precision in-place: `d .+= scale * (u .⊙ u)`
4. Recompute weighted mean in-place: `xi .= d_new .⊙ μ`

### Model structure

The `@model` function is structurally identical to the existing `Dynamic` model.
The same `LowRankMeta()` is used on softdot nodes. The only difference is that
`w` priors are `MvNormalWeightedMeanPrecision{..., Diagonal}`:

```
w[i] ~ prior                           # MvNormalWeightedMeanPrecision{..., Diagonal}
τ[i] ~ GammaShapeRate(...)
β[i] ~ GammaShapeRate(...)
z[i,j] ~ softdot(features[j], w[i], τ[i])   where {meta = LowRankMeta()}
γ[i,j] ~ GammaShapeRate(1.0, β[i])
z[i,j] ~ Log(γ[i,j])
y[j]   ~ NormalMeanPrecision(predictions[i,j], γ[i,j])
```

## Complexity Comparison

| Operation | Dynamic (dense) | Dynamic Diagonal |
|-----------|----------------|-----------------|
| Posterior storage | O(n²) per w[i] | O(n) per w[i] |
| Per-observation update | O(n²) BLAS.syr! | O(n) elementwise |
| Mean recovery | O(n³) or O(n²) solve | O(n) Sherman–Morrison |
| Total for T obs | O(Tn²) | O(Tn) |
| KL divergence | O(n³) det + solve | O(n) diagonal det + div |

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §10.1.
  Springer. — KL-optimal diagonal projection of multivariate normals.

- Minka, T. P. (2001). *Expectation Propagation for Approximate Bayesian
  Inference*. UAI. — ADF as single-pass EP; sequential moment matching with
  projection onto exponential families.

- Opper, M. (1998). *A Bayesian Approach to On-line Learning*. In On-line
  Learning in Neural Networks, Cambridge University Press. — Assumed density
  filtering for sequential Bayesian updates with tractable projections.

- Seeger, M. (2005). *Expectation Propagation for Exponential Families*. Tech
  report. — Connections between ADF, EP, and variational message passing with
  constrained posterior families.

- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
  Johns Hopkins University Press. — Sherman–Morrison–Woodbury identity for
  efficient rank-1 updates of matrix inverses.
