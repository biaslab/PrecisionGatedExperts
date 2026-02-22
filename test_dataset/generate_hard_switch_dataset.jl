#!/usr/bin/env julia

using Random
using CSV
using DataFrames

"""
Create a synthetic univariate dataset with explicit regime switches and simple predictors.

Output columns:
  - t: time index
  - regime: 1 or 2
  - OT: target
  - pred_a: expert A (good in regime 1, bad in regime 2)
  - pred_b: expert B (good in regime 2, bad in regime 1)
  - pred_avg: simple average baseline of A/B
"""
function main()
    rng = MersenneTwister(42)
    n = 1600
    block = 200

    t = collect(1:n)
    regime = [isodd(fld(i - 1, block) + 1) ? 1 : 2 for i in t]

    # Two latent patterns; the active one changes by regime.
    s1 = [sin(2π * i / 40) + 0.15 * sin(2π * i / 9) for i in t]
    s2 = [0.9 * cos(2π * i / 55) - 0.1 * sin(2π * i / 7) for i in t]

    noise = 0.06 .* randn(rng, n)
    y = [regime[i] == 1 ? s1[i] + noise[i] : s2[i] + noise[i] for i in t]

    # Expert A tracks s1 well, but is biased in regime 2.
    pred_a = [
        regime[i] == 1 ?
        s1[i] + 0.05 * randn(rng) :
        (0.25 * s1[i] + 0.45 + 0.12 * randn(rng)) for i in t
    ]

    # Expert B tracks s2 well, but is biased in regime 1.
    pred_b = [
        regime[i] == 2 ?
        s2[i] + 0.05 * randn(rng) :
        (0.2 * s2[i] - 0.4 + 0.12 * randn(rng)) for i in t
    ]

    df = DataFrame(;
        t = t,
        regime = regime,
        OT = y,
        pred_a = pred_a,
        pred_b = pred_b,
    )

    out = joinpath(@__DIR__, "hard_switch_univariate.csv")
    CSV.write(out, df)
    println("Wrote: ", out)
end

main()
