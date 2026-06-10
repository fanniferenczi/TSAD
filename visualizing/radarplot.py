import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------
# Data
# -----------------------------------------------
models = ["TimesNet", "ModernTCN", "TranAD", "AnomTr.", "GTA"]

data = {
    #                f1      au-pr   train_mem  inf_mem  latency  throughput  
    "TimesNet":  [83.87,  18.044,  331.784,  219.81,   340.0,   395.0],
    "ModernTCN": [83.97,  17.65,   423.286,  171.93,    59.0,  3327.0],
    "TranAD":    [89.61,  39.322,    2.774,    1.608,   108.0,  1257.0],
    "AnomTr.":   [94.77,  13.542, 2006.818, 1533.734,  139.0,   461.0],
    "GTA":       [88.04,  16.672,  841.788,  409.748,  880.0,   147.0],
}

labels = ["F1 (%)", "AU-PR (%)", "Train Mem (MB)\nInverted scale", "Inf Mem (MB)\nInverted scale", "Latency (ms)\nInverted scale", "Throughput (windows/s)"]
N = len(labels)

# -----------------------------------------------
# Per-axis scale definitions
# axis_min, axis_max, log_scale, invert
# For inverted axes: higher raw value → lower normalised score (worse)
# -----------------------------------------------
axes_cfg = [
    # (min,    max,     log,   invert, top_pad)  top_pad keeps best-case values off the outer ring
    (0,       100,     False, False,  0.00),   # F1
    (0,        40,     False, False,  0.00),   # AU-PR
    (0,      2100,     False, True,   0.03),   # Train mem  (TranAD≈0 would otherwise touch ring)
    (0,      1700,     False, True,   0.03),   # Inf mem    (same)
    (0,      1100,     False, True,   0.00),   # Latency
    (0,      3400,     False, False,  0.00),   # Throughput
]

# Per-axis offset for the outermost ring label (r=1.00).
# Inner rings always use 0.04. Increase a value here if the label sits on the ring.
outer_offsets = [
    0.1,   # F1
    0.04,   # AU-PR
    0.04,   # Train mem
    0.1,   # Inf mem
    0.1,   # Latency
    0.1,   # Throughput
]

# -----------------------------------------------
# Normalise each value to [0, 1] on its axis
# 1 = best (outermost ring), 0 = worst (centre)
# -----------------------------------------------
def normalise(value, cfg):
    vmin, vmax, log, invert, top_pad = cfg
    if log:
        v    = np.log10(max(value, vmin))
        vlo  = np.log10(vmin)
        vhi  = np.log10(vmax)
        norm = (v - vlo) / (vhi - vlo)
    else:
        norm = (value - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    result = (1 - norm) if invert else norm
    return np.clip(result, 0, 1 - top_pad)

normed = {}
for m in models:
    normed[m] = [normalise(data[m][i], axes_cfg[i]) for i in range(N)]

# -----------------------------------------------
# Colorblind-friendly palette (Wong 2011)
# -----------------------------------------------
colors = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA3377"]

# -----------------------------------------------
# Build radar
# -----------------------------------------------
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels([])
ax.set_ylim(0, 1)
ax.yaxis.grid(True, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
ax.xaxis.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.3)

# Plot each model
for i, m in enumerate(models):
    vals = normed[m] + normed[m][:1]
    ax.plot(angles, vals, color=colors[i], linewidth=2, label=m)
    ax.fill(angles, vals, color=colors[i], alpha=0.07)

# -----------------------------------------------
# Axis labels (no scale hints in label — shown as spoke ticks instead)
# -----------------------------------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=11)

# Per-axis padding (points) between the outermost ring and the axis label.
label_pads = [
    30,   # F1
    15,   # AU-PR
    20,   # Train mem
    50,   # Inf mem
    20,   # Latency
    20,   # Throughput
]

for tick, pad in zip(ax.xaxis.get_major_ticks(), label_pads):
    tick.set_pad(pad)

# -----------------------------------------------
# Tick annotations on every spoke
# Show 4 evenly-spaced values at ring positions 0.25 / 0.50 / 0.75 / 1.00
# For inverted axes reverse-normalise so displayed value decreases outward
# -----------------------------------------------
ring_levels = [0.25, 0.50, 0.75, 1.00]

def ring_to_actual(r, cfg):
    vmin, vmax, log, invert, _ = cfg
    # r is the normalised ring position (0–1, 1 = outermost = best)
    # convert back to raw value
    norm = (1 - r) if invert else r
    if log:
        vlo = np.log10(max(vmin, 1e-9))
        vhi = np.log10(vmax)
        return 10 ** (vlo + norm * (vhi - vlo))
    else:
        return vmin + norm * (vmax - vmin)

# Format helper: integers for large values, 1 decimal for small
def fmt(v):
    if v >= 100:
        return f"{v:.0f}"
    elif v >= 10:
        return f"{v:.1f}"
    else:
        return f"{v:.2f}"

for axis_idx in range(N):
    angle = angles[axis_idx]
    cfg   = axes_cfg[axis_idx]
    for r in ring_levels:
        actual = ring_to_actual(r, cfg)
        offset = outer_offsets[axis_idx] if r == 1.00 else 0.04
        ax.text(
            angle, r + offset, fmt(actual),
            ha="center", va="bottom", fontsize=7.5, color="grey"
        )

ax.set_title("Model Comparison for Edge Deployment",
             fontsize=13, fontweight="bold", pad=40)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.15),
    fontsize=10,
    framealpha=0.7,
)

plt.tight_layout()
plt.savefig("radar_chart.png", bbox_inches="tight", dpi=300)
plt.show()