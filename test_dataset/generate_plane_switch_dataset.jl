#!/usr/bin/env julia

using DataFrames
using CSV
using StableRNGs
using Plots
using Statistics

# ── 1. Linear boundary (plane switch) ────────────────────────────────────────

function generate_plane_switch_data(
    rng;
    n::Int = 1600,
    line::NTuple{3,Float64} = (1.0, -0.7, 0.1),
    regime_noise::Float64 = 0.35,
    obs_noise::Float64 = 0.08,
)
    a, b, c = line
    x1 = rand(rng, n) .* 4 .- 2
    x2 = rand(rng, n) .* 4 .- 2

    signed = a .* x1 .+ b .* x2 .+ c
    signed_noisy = signed .+ regime_noise .* randn(rng, n)
    regime = ifelse.(signed_noisy .< 0.0, 1, 2)

    f1 = 1.2 .* x1 .- 0.9 .* x2 .+ 0.35 .* sin.(1.8 .* x1)
    f2 = -0.8 .* x1 .+ 1.1 .* x2 .+ 0.30 .* cos.(1.4 .* x2)
    y = [regime[i] == 1 ? f1[i] : f2[i] for i in 1:n] .+ obs_noise .* randn(rng, n)

    pred_a = [
        regime[i] == 1 ?
        (f1[i] + 0.06 * randn(rng)) :
        (0.35 * f1[i] + 0.55 + 0.16 * randn(rng)) for i in 1:n
    ]

    pred_b = [
        regime[i] == 2 ?
        (f2[i] + 0.06 * randn(rng)) :
        (0.35 * f2[i] - 0.55 + 0.16 * randn(rng)) for i in 1:n
    ]

    dist = signed ./ sqrt(a^2 + b^2)

    return DataFrame(;
        x1 = x1,
        x2 = x2,
        dist_to_line = dist,
        regime = regime,
        OT = y,
        pred_a = pred_a,
        pred_b = pred_b,
    )
end

# ── 2. XOR boundary (nonlinear quadrant-based regimes) ───────────────────────

function generate_xor_data(
    rng;
    n::Int = 1600,
    obs_noise::Float64 = 0.08,
    regime_noise::Float64 = 0.20,
)
    x1 = rand(rng, n) .* 4 .- 2
    x2 = rand(rng, n) .* 4 .- 2

    # XOR: regime 1 when sign(x1) == sign(x2), regime 2 otherwise
    signed = x1 .* x2
    signed_noisy = signed .+ regime_noise .* randn(rng, n)
    regime = ifelse.(signed_noisy .> 0.0, 1, 2)

    f1 = 1.0 .* x1 .- 0.8 .* x2 .+ 0.3 .* sin.(2.0 .* x1 .* x2)
    f2 = -0.7 .* x1 .+ 1.2 .* x2 .+ 0.25 .* cos.(1.5 .* (x1 .+ x2))
    y = [regime[i] == 1 ? f1[i] : f2[i] for i in 1:n] .+ obs_noise .* randn(rng, n)

    pred_a = [
        regime[i] == 1 ?
        (f1[i] + 0.06 * randn(rng)) :
        (0.3 * f1[i] + 0.5 + 0.16 * randn(rng)) for i in 1:n
    ]

    pred_b = [
        regime[i] == 2 ?
        (f2[i] + 0.06 * randn(rng)) :
        (0.3 * f2[i] - 0.5 + 0.16 * randn(rng)) for i in 1:n
    ]

    return DataFrame(;
        x1 = x1,
        x2 = x2,
        dist_to_line = signed,
        regime = regime,
        OT = y,
        pred_a = pred_a,
        pred_b = pred_b,
    )
end

# ── 3. XOR-simple target (target itself has XOR structure) ───────────────────

function generate_xor_simple_data(
    rng;
    n::Int = 1600,
    obs_noise::Float64 = 0.08,
    regime_noise::Float64 = 0.20,
)
    x1 = rand(rng, n) .* 4 .- 2
    x2 = rand(rng, n) .* 4 .- 2

    # Keep the same regime logic as old XOR dataset:
    # regime 1 when signs match, regime 2 otherwise (with noisy boundary).
    signed = x1 .* x2
    signed_noisy = signed .+ regime_noise .* randn(rng, n)
    regime = ifelse.(signed_noisy .> 0.0, 1, 2)

    # XOR-like target in [0, 1]:
    #   regime 1 -> around 0
    #   regime 2 -> around 1
    y_base = ifelse.(regime .== 1, 0.0, 1.0)
    y = clamp.(y_base .+ obs_noise .* randn(rng, n), 0.0, 1.0)

    # Experts follow the same regime preferences as before:
    # pred_a good in regime 1, pred_b good in regime 2.
    pred_a = Vector{Float64}(undef, n)
    pred_b = Vector{Float64}(undef, n)
    for i in 1:n
        if regime[i] == 1
            pred_a[i] = clamp(0.06 * randn(rng), 0.0, 1.0)
            pred_b[i] = clamp(0.78 + 0.16 * randn(rng), 0.0, 1.0)
        else
            pred_a[i] = clamp(0.22 + 0.16 * randn(rng), 0.0, 1.0)
            pred_b[i] = clamp(1.0 + 0.06 * randn(rng), 0.0, 1.0)
        end
    end

    return DataFrame(;
        x1 = x1,
        x2 = x2,
        dist_to_line = signed,
        regime = regime,
        OT = y,
        pred_a = pred_a,
        pred_b = pred_b,
    )
end

# ── Generic heatmap (works for any dataset) ──────────────────────────────────

function plot_regime_heatmap(df; title_str = "", boundary_fn = nothing)
    mse_a = (df.pred_a .- df.OT) .^ 2
    mse_b = (df.pred_b .- df.OT) .^ 2

    # Positive => A is better (blue), Negative => B is better (red)
    advantage = mse_b .- mse_a

    # Compress range with signed cube-root for saturated colors even near boundary
    adv_compressed = sign.(advantage) .* abs.(advantage) .^ (1 / 3)
    clim = maximum(abs.(adv_compressed))

    margin = 0.15
    xl = (minimum(df.x1) - margin, maximum(df.x1) + margin)
    yl = (minimum(df.x2) - margin, maximum(df.x2) + margin)

    p = scatter(
        df.x1,
        df.x2;
        marker_z = adv_compressed,
        color = :RdBu,
        clims = (-clim, clim),
        markersize = 3,
        alpha = 0.9,
        markerstrokewidth = 0,
        xlabel = "x₁",
        ylabel = "x₂",
        xlims = xl,
        ylims = yl,
        colorbar_title = "sign·∛|ΔM|",
        title = title_str,
        label = "",
        legend = :topright,
    )

    if boundary_fn !== nothing
        xs = range(xl[1], xl[2]; length = 300)
        boundary_fn(p, xs)
    end

    return p
end

function plot_xor_simple_quality(df; title_prefix = "XOR-simple")
    margin = 0.15
    xl = (minimum(df.x1) - margin, maximum(df.x1) + margin)
    yl = (minimum(df.x2) - margin, maximum(df.x2) + margin)

    p_target = scatter(
        df.x1, df.x2;
        marker_z = df.OT,
        color = :viridis,
        clims = (0.0, 1.0),
        markersize = 3.2,
        alpha = 0.9,
        markerstrokewidth = 0,
        xlabel = "x₁",
        ylabel = "x₂",
        xlims = xl,
        ylims = yl,
        colorbar_title = "OT",
        title = "$(title_prefix): target",
        label = "",
    )
    plot!(p_target, range(xl[1], xl[2]; length = 300), zero.(range(xl[1], xl[2]; length = 300));
        linewidth = 2.2, color = :black, linestyle = :dash, label = "")
    vline!(p_target, [0.0];
        linewidth = 2.2, color = :black, linestyle = :dash, label = "x₁=0,x₂=0")

    p_regime = scatter(
        df.x1, df.x2;
        marker_z = df.regime,
        color = cgrad([:navy, :gold]),
        clims = (1.0, 2.0),
        markersize = 3.2,
        alpha = 0.9,
        markerstrokewidth = 0,
        xlabel = "x₁",
        ylabel = "x₂",
        xlims = xl,
        ylims = yl,
        colorbar_title = "regime",
        title = "$(title_prefix): regime",
        label = "",
    )
    plot!(p_regime, range(xl[1], xl[2]; length = 300), zero.(range(xl[1], xl[2]; length = 300));
        linewidth = 2.2, color = :black, linestyle = :dash, label = "")
    vline!(p_regime, [0.0];
        linewidth = 2.2, color = :black, linestyle = :dash, label = "x₁=0,x₂=0")

    p_adv = plot_regime_heatmap(df;
        title_str = "$(title_prefix): expert advantage",
        boundary_fn = (p, xs) -> begin
            plot!(p, xs, zero.(xs); linewidth = 2.2, color = :black, linestyle = :dash, label = "")
            vline!(p, [0.0]; linewidth = 2.2, color = :black, linestyle = :dash, label = "boundary")
        end,
    )

    return plot(p_target, p_regime, p_adv; layout = (1, 3), size = (1700, 500), margin = 5Plots.mm)
end

function plot_xor_quality(df; title_prefix = "XOR")
    margin = 0.15
    xl = (minimum(df.x1) - margin, maximum(df.x1) + margin)
    yl = (minimum(df.x2) - margin, maximum(df.x2) + margin)

    tmin, tmax = extrema(df.OT)
    if tmax <= tmin
        tmin, tmax = tmin - 1e-6, tmax + 1e-6
    end

    p_target = scatter(
        df.x1, df.x2;
        marker_z = df.OT,
        color = :viridis,
        clims = (tmin, tmax),
        markersize = 3.2,
        alpha = 0.9,
        markerstrokewidth = 0,
        xlabel = "x₁",
        ylabel = "x₂",
        xlims = xl,
        ylims = yl,
        colorbar_title = "OT",
        title = "$(title_prefix): target",
        label = "",
    )
    xs = range(xl[1], xl[2]; length = 300)
    plot!(p_target, xs, zero.(xs);
        linewidth = 2.2, color = :black, linestyle = :dash, label = "")
    vline!(p_target, [0.0];
        linewidth = 2.2, color = :black, linestyle = :dash, label = "x₁=0,x₂=0")

    p_regime = scatter(
        df.x1, df.x2;
        marker_z = df.regime,
        color = cgrad([:navy, :gold]),
        clims = (1.0, 2.0),
        markersize = 3.2,
        alpha = 0.9,
        markerstrokewidth = 0,
        xlabel = "x₁",
        ylabel = "x₂",
        xlims = xl,
        ylims = yl,
        colorbar_title = "regime",
        title = "$(title_prefix): regime",
        label = "",
    )
    plot!(p_regime, xs, zero.(xs);
        linewidth = 2.2, color = :black, linestyle = :dash, label = "")
    vline!(p_regime, [0.0];
        linewidth = 2.2, color = :black, linestyle = :dash, label = "x₁=0,x₂=0")

    p_adv = plot_regime_heatmap(df;
        title_str = "$(title_prefix): expert advantage",
        boundary_fn = (p, xs) -> begin
            plot!(p, xs, zero.(xs); linewidth = 2.2, color = :black, linestyle = :dash, label = "")
            vline!(p, [0.0]; linewidth = 2.2, color = :black, linestyle = :dash, label = "boundary")
        end,
    )

    return plot(p_target, p_regime, p_adv; layout = (1, 3), size = (1700, 500), margin = 5Plots.mm)
end

# ── Main ─────────────────────────────────────────────────────────────────────

function main()
    rng = StableRNG(2026)
    mkpath("test_dataset/viz")

    # 1. Linear
    line = (1.0, -0.7, 0.1)
    df_linear = generate_plane_switch_data(rng; n = 1600, line = line)
    CSV.write("test_dataset/plane_switch_dataset.csv", df_linear)
    @info "Linear dataset saved" n = nrow(df_linear)

    a, b, c = line
    p1 = plot_regime_heatmap(df_linear;
        title_str = "Linear boundary",
        boundary_fn = (p, xs) -> plot!(p, xs, (-a .* xs .- c) ./ b;
            linewidth = 2.5, color = :black, linestyle = :dash, label = "boundary"),
    )

    # 2. XOR
    df_xor = generate_xor_data(rng; n = 1600)
    CSV.write("test_dataset/xor_dataset.csv", df_xor)
    @info "XOR dataset saved" n = nrow(df_xor)

    p2 = plot_regime_heatmap(df_xor;
        title_str = "XOR boundary",
        boundary_fn = (p, xs) -> begin
            plot!(p, xs, zero.(xs); linewidth = 2.5, color = :black, linestyle = :dash, label = "")
            vline!(p, [0.0]; linewidth = 2.5, color = :black, linestyle = :dash, label = "boundary")
        end,
    )
    p_xor_quality = plot_xor_quality(df_xor)
    savefig(p_xor_quality, "test_dataset/viz/xor_quality.png")
    @info "XOR quality plot saved" path = "test_dataset/viz/xor_quality.png"

    # 3. XOR-simple (target itself is XOR-like)
    df_xor_simple = generate_xor_simple_data(rng; n = 1600)
    CSV.write("test_dataset/xor_simple_dataset.csv", df_xor_simple)
    @info "XOR-simple dataset saved" n = nrow(df_xor_simple) mean_OT = mean(df_xor_simple.OT)

    p3 = plot_regime_heatmap(df_xor_simple;
        title_str = "XOR-simple boundary",
        boundary_fn = (p, xs) -> begin
            plot!(p, xs, zero.(xs); linewidth = 2.5, color = :black, linestyle = :dash, label = "")
            vline!(p, [0.0]; linewidth = 2.5, color = :black, linestyle = :dash, label = "boundary")
        end,
    )

    p_xor_simple_quality = plot_xor_simple_quality(df_xor_simple)
    savefig(p_xor_simple_quality, "test_dataset/viz/xor_simple_quality.png")
    @info "XOR-simple quality plot saved" path = "test_dataset/viz/xor_simple_quality.png"

    # Combined plot
    p_all = plot(p1, p2, p3; layout = (1, 3), size = (1800, 550), margin = 5Plots.mm)
    savefig(p_all, "test_dataset/viz/regime_heatmaps_all.png")
    @info "Combined heatmap saved" path = "test_dataset/viz/regime_heatmaps_all.png"
end

main()
