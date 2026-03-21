# Interactive ELBO Visualizations: Gamma vs LogNormal

Three interactive HTML visualizations exploring when and how well Gamma and LogNormal distributions can approximate each other under variational inference. All computations are closed-form (no sampling), running entirely in the browser.

## Context

In non-conjugate factor graphs, a variable `γ > 0` may receive messages of different exponential-family types — e.g., a **LogNormal** message from a latent variable `z` and a **Gamma** message from observations. The variational posterior `q(γ)` must be chosen from one family. These visualizations answer: **which family, and how good is the approximation?**

---

## 1. LogNormal Approximating Gamma — `elbo_visualization.html`

**Question:** How well can `Q = LogNormal(μ, σ²)` approximate a `Gamma(α, β)` target?

### Key Math

The ELBO decomposes into terms requiring only `E_Q[x]` and `E_Q[ln x]`, both closed-form under LogNormal:

```
E_Q[log Gamma(x; α, β)] = (α−1)μ − β·exp(μ + σ²/2) + α·ln β − ln Γ(α)
H[Q]                    = μ + ½ ln(2πe σ²)
```

At optimal `Q`: `σ² = 1/α`, `μ = ln(α/β) − 1/(2α)`.

### Key Result

**Approximation quality depends only on α** (the Gamma shape), not on β.

```
KL*(α) = ln Γ(α) − (α − ½) ln α + α − ½ ln(2π)    [Stirling remainder]
```

| α range | KL (nats) | Quality |
|---------|-----------|---------|
| α ≥ 5  | < 0.02    | Excellent — both distributions are nearly Gaussian |
| α ≈ 2  | ~0.04     | Good |
| α < 1  | > 0.08    | Poor — Gamma is J-shaped, LogNormal always has a mode > 0 |

### Controls

- **α, β** sliders: set the Gamma target parameters
- **μ, σ** sliders: set the LogNormal approximation (manual mode)
- **Auto-optimize**: finds the optimal `(μ*, σ*)` analytically
- **Click on ELBO heatmap**: manually set `(μ, σ)` to explore the landscape

### Panels

1. **PDF Comparison** — overlay of Gamma target (red) and LogNormal approximation (blue)
2. **ELBO Landscape** — heatmap in `(μ, σ)` space with optimal marked
3. **KL vs α** — the Stirling remainder curve; shows quality depends only on shape

---

## 2. Gamma Approximating LogNormal — `elbo_reverse.html`

**Question:** How well can `Q = Gamma(α, β)` approximate a `LogNormal(μ, σ²)` target?

### Key Math

Requires **digamma** `ψ(a)` and **trigamma** `ψ₁(a)` for the moments of `ln x` under a Gamma:

```
E_Q[ln x]    = ψ(a) − ln b
Var_Q[ln x]  = ψ₁(a)
E_Q[x]       = a / b
```

The ELBO is:

```
ELBO = (a − a·ψ(a) + ln Γ(a)) − ln σ − ½ ln(2π) − ψ₁(a)/(2σ²) − (ψ(a) − ln b − μ)²/(2σ²)
```

At optimal: `β* = exp(ψ(α*) − μ)`, and `α*` satisfies `1 − α·ψ₁(α) − ψ₂(α)/(2σ²) = 0`.

### Key Result

**Approximation quality depends only on σ** (the LogNormal log-std), not on μ.

```
α* ≈ 1/σ²    (log-scale precision becomes Gamma shape)
```

| σ range | KL (nats) | Quality |
|---------|-----------|---------|
| σ ≤ 0.3 | < 0.01   | Excellent — LogNormal is concentrated |
| σ ≈ 0.6 | ~0.03    | Good |
| σ > 1.0 | > 0.08   | Poor — LogNormal's heavy tail exceeds Gamma's exponential tail |

### Controls

- **μ, σ** sliders: set the LogNormal target parameters
- **α, β** sliders: set the Gamma approximation (manual mode)
- **Auto-optimize**: finds optimal `(α*, β*)` numerically
- **Click on ELBO heatmap**: manually set `(α, β)`

### Panels

1. **PDF Comparison** — LogNormal target (red) vs Gamma approximation (blue)
2. **ELBO Landscape** — heatmap in `(α, β)` space
3. **KL vs σ** — quality curve; shows dependence only on log-std

---

## 3. Phase Diagram: Which Constraint to Use — `elbo_phase_diagram.html`

**Question:** Given a variable receiving both a `Gamma(α, β)` message and a `LogNormal(μ, σ)` message, should `q` be Gamma or LogNormal?

### Setup

At depth 1 in the factor graph, γ receives two messages:
- **From z** (prior): LogNormal message parameterized by σ
- **From observations** (likelihood): Gamma message parameterized by α

If `q = Gamma`: it matches the Gamma message for free (conjugate) but pays cost **f(σ)** for the LogNormal message.

If `q = LogNormal`: it matches the LogNormal message for free but pays cost **g(α)** for the Gamma message.

### Key Result — Decision Boundary

```
Prefer Gamma q    when  f(σ) < g(α)     i.e.  ασ² < 1
Prefer LogNormal q when  g(α) < f(σ)     i.e.  ασ² > 1
```

**Boundary: ασ² = 1**

Derivation from the asymptotic costs:
```
g(α) ≈ 1/(12α)    for large α     (Stirling error)
f(σ) ≈ σ²/12      for small σ

g(α) = f(σ)  ⟹  1/(12α) = σ²/12  ⟹  ασ² = 1
```

### Intuition

| Region | Why |
|--------|-----|
| **ασ² < 1** (blue) | σ is small → LogNormal is concentrated → Gamma can capture it. α is small → Gamma is J-shaped → LogNormal can't capture it. Use **Gamma q**. |
| **ασ² > 1** (pink) | σ is large → LogNormal has heavy tail → Gamma's exponential tail can't match. α is large → Gamma is bell-shaped → LogNormal handles it fine. Use **LogNormal q**. |

### Controls

- **Click on the phase diagram**: select an `(α, σ)` point to explore
- **Hover**: see real-time preference at any point

### Panels

1. **Phase Diagram** — 2D heatmap with α (x-axis) vs σ (y-axis), blue = Gamma preferred, pink = LogNormal preferred, white curve = exact boundary, dashed = ασ² = 1 asymptote
2. **Density Comparison** — at the clicked point: true posterior (gray, product of both messages) vs best Gamma q (blue) vs best LogNormal q (pink)
3. **KL Bar Chart** — side-by-side comparison of f(σ) and g(α) at clicked point

---

## Summary Table

| Visualization | Target | Approx | Quality controlled by | Clean regime |
|---|---|---|---|---|
| 1. `elbo_visualization` | Gamma(α,β) | LogNormal | **α** (shape) | α ≥ 5 |
| 2. `elbo_reverse` | LogNormal(μ,σ) | Gamma | **σ** (log-std) | σ ≤ 0.3 |
| 3. `elbo_phase_diagram` | Both messages | Choose family | **ασ²** (product) | boundary at 1 |

### The Unifying Insight

Both approximations become exact when the target is Gaussian-like:
- Gamma becomes Gaussian for large α
- LogNormal becomes Gaussian (on log-scale) for small σ

The phase boundary **ασ² = 1** is where both families are equally (in)capable. Below it, the Gamma message is "harder" (J-shaped) so use Gamma q to match it exactly. Above it, the LogNormal message is "harder" (heavy-tailed) so use LogNormal q to match it exactly.

---

## Technical Notes

- All special functions (logΓ, ψ, ψ₁) are implemented via recurrence + asymptotic series
- Optimizations in the phase diagram use 2D grid search with refinement (~25,000 ELBO evaluations per click, sub-10ms)
- No external dependencies — pure HTML/CSS/JS, runs in any modern browser
- Dark theme (GitHub-style), responsive layout
