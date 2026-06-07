import numpy as np
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────────
models  = ["TimesNet", "ModernTCN", "TranAD", "AnomTr.", "GTA"]
colors  = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA3377"]

raw = {
    "F1 score":               [83.87, 83.97, 89.61, 94.77, 88.04],
    "Train Mem. Usage":     [331.784, 423.286, 2.774, 2006.818, 841.788],
    "Inference Mem. Usage": [219.81, 171.93, 1.608, 1533.734, 409.748],
    "Latency":                [340.4, 58.6, 107.8, 138.7, 880.3],
    "Throughput":             [395.1, 3321.6, 1256.8, 461.4, 146.6],
    "kMACs":                  [1509074, 15553, 1013, 516087, 31357],
}

lower_is_better = {
    "F1 score":               False,
    "Train Mem. Usage":     True,
    "Inference Mem. Usage": True,
    "Latency":                True,
    "Throughput":             False,
    "kMACs":                  True,
}


fixed_ranges = {
    "F1 score":               (83.87,       94.77),
    "Train Mem. Usage":     (2.774,    2006.818),
    "Inference Mem. Usage": (1.608,    1533.734),
    "Latency":                (58.6,       880.3),
    "Throughput":             (146.6,     3321.6),
    "kMACs":                  (1013,    1509074),
}

# Fixed ranges: (min, max) — domain-motivated, not data-driven
#fixed_ranges = {
#    "F1 (%)":         (0,       100),
#    "Train Mem (MB)": (0,    2006.818),
#    "Infer Mem (MB)": (0,    1533.734),
#    "Latency (ms)":   (0,       880.3),
#    "kMACs":          (0,    1509074),
#}

metrics   = list(raw.keys())
n_metrics = len(metrics)
n_models  = len(models)

# ── Normalise by fixed range, invert where lower is better ────────────────────
norm = np.zeros((n_models, n_metrics))
for j, metric in enumerate(metrics):
    vals = np.array(raw[metric], dtype=float)
    mn, mx = fixed_ranges[metric]
    scaled = (vals - mn) / (mx - mn)
    if lower_is_better[metric]:
        scaled = 1 - scaled
    norm[:, j] = scaled

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(n_metrics)

for i, (model, color) in enumerate(zip(models, colors)):
    y = norm[i]
    ax.plot(x, y, color=color, linewidth=2.2, marker="o",
            markersize=7, label=model, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{m}\n"
     for m in metrics],
    fontsize=10
)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["worst\n(0)", "0.25", "0.50", "0.75", "best\n(1.0)"],
                   fontsize=9)
ax.set_ylabel("Normalised score", fontsize=10)

for xi in x:
    ax.axvline(xi, color="grey", linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28),
          ncol=5, fontsize=10, frameon=False)

ax.set_title("Model Comparison: Accuracy and Edge Deployment Efficiency",
             fontsize=12, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("parallel_coords.png", bbox_inches="tight", dpi=300)
print("Saved parallel_coords.png and parallel_coords.pdf")