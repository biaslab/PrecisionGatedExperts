import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

data = []
with open("/Users/mykola/Downloads/hierarchical_hyperparam_grid.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            "dataset": row["dataset"],
            "horizon": int(row["horizon"]),
            "w_scale": float(row["w_scale"]),
            "tau_rate": float(row["tau_rate"]),
            "subsample_size": int(row["subsample_size"]),
            "mae": float(row["mae"]),
        })

datasets = sorted(set(d["dataset"] for d in data))
horizons = sorted(set(d["horizon"] for d in data))

fig, axes = plt.subplots(len(datasets), len(horizons), figsize=(16, 4 * len(datasets)),
                         sharex=True, squeeze=False)

for i, ds in enumerate(datasets):
    for j, h in enumerate(horizons):
        ax = axes[i][j]
        subset = [d for d in data if d["dataset"] == ds and d["horizon"] == h]

        for d in subset:
            label = f"w={d['w_scale']}, τ={d['tau_rate']}"
            ax.plot(d["subsample_size"], d["mae"], 'o', markersize=8, alpha=0.7)

        ax.set_title(f"{ds}  h={h}", fontsize=11)
        ax.set_xticks([10, 20])
        ax.set_xlabel("subsample_size")
        if j == 0:
            ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)

# Build a shared legend from unique (w_scale, tau_rate) combos
configs = sorted(set((d["w_scale"], d["tau_rate"]) for d in data))
colors = plt.cm.tab10.colors
markers = ['o', 's', '^', 'D']
# Re-plot with consistent colors/markers
for ax_row in axes:
    for ax in ax_row:
        ax.clear()

for i, ds in enumerate(datasets):
    for j, h in enumerate(horizons):
        ax = axes[i][j]
        subset = [d for d in data if d["dataset"] == ds and d["horizon"] == h]
        for d in subset:
            cfg_idx = configs.index((d["w_scale"], d["tau_rate"]))
            ax.plot(d["subsample_size"], d["mae"],
                    marker=markers[cfg_idx], color=colors[cfg_idx],
                    markersize=9, alpha=0.8, linestyle='none')

        # Connect same config across subsample sizes
        for ci, (ws, tr) in enumerate(configs):
            pts = sorted([(d["subsample_size"], d["mae"]) for d in subset
                          if d["w_scale"] == ws and d["tau_rate"] == tr])
            if len(pts) == 2:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        linestyle='--', color=colors[ci], alpha=0.4, linewidth=1)

        ax.set_title(f"{ds}  h={h}", fontsize=11)
        ax.set_xticks([10, 20])
        ax.set_xlabel("subsample_size")
        if j == 0:
            ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)

legend_handles = [plt.Line2D([0], [0], marker=markers[ci], color=colors[ci],
                              linestyle='none', markersize=9,
                              label=f"w_scale={ws}, tau_rate={tr}")
                  for ci, (ws, tr) in enumerate(configs)]
fig.legend(handles=legend_handles, loc='upper center', ncol=len(configs),
           bbox_to_anchor=(0.5, 1.02), fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/Users/mykola/repos/probabilistic_ensemble_forecasting/viz/mae_vs_subsample_size.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved to viz/mae_vs_subsample_size.png")
