import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
models = ["TimesNet", "ModernTCN", "TranAD", "AnomTr.", "GTA"]

avg_latency_server = {
    "TimesNet":  106.6,
    "ModernTCN":   9.7,
    "TranAD":     10.9,
    "AnomTr.":    33.7,
    "GTA":        64.0,
}

avg_latency_edge = {
    "TimesNet":  340.4,
    "ModernTCN":  58.6,
    "TranAD":    107.8,
    "AnomTr.":   138.7,
    "GTA":       880.3,
}

slowdown = {m: avg_latency_edge[m] / avg_latency_server[m] for m in models}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(models))
bars = ax.bar(x, [slowdown[m] for m in models], color="#4C72B0",
              width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

# Annotate each bar with the exact slowdown value
for bar, m in zip(bars, models):
    val = slowdown[m]
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{val:.1f}\u00d7",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold",
    )

# Reference line at 1x (no slowdown)
ax.axhline(1, color="black", linewidth=0.9, linestyle=":", zorder=4)
ax.text(len(models) - 0.45, 1.15, "no slowdown", fontsize=8.5,
        color="black", va="bottom")

# Axes labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel("Latency slowdown factor (edge / server)", fontsize=11)
ax.set_ylim(0, max(slowdown.values()) * 1.18)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f\u00d7"))

ax.set_title("Edge vs. Server Inference Latency Slowdown", fontsize=12,
             fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig("latency_slowdown.png",
            bbox_inches="tight", dpi=300)
print("Saved latency_slowdown.png and latency_slowdown.pdf")