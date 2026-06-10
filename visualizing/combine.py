import matplotlib.pyplot as plt
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
server = {
    "TimesNet": {"infer_mem": 219.81, "throughput": 1606.7},
    "ModernTCN": {"infer_mem": 171.93 , "throughput": 14047.1},
    "TranAD": {"infer_mem": 1.61, "throughput": 11545.2},
    "AnomTr.": {"infer_mem": 3123.55, "throughput": 3797.0},
    "GTA": {"infer_mem": 409.75, "throughput": 2009.8},
}

edge = {
    "TimesNet": {"infer_mem": 219.81, "throughput": 395.1},
    "ModernTCN": {"infer_mem": 171.93, "throughput": 3321.6},
    "TranAD": {"infer_mem": 1.61, "throughput": 1256.8},
    "AnomTr.": {"infer_mem": 1533.73, "throughput": 461.4},
    "GTA": {"infer_mem": 409.75, "throughput": 146.6},
}

# ── Base model colors (same as your previous plots) ─────────────────────────
model_colors = {
    "TimesNet": "#4C72B0",
    "ModernTCN": "#DD8452",
    "TranAD": "#55A868",
    "AnomTr.": "#C44E52",
    "GTA": "#8172B2",
}

# ── helper: lighten color ────────────────────────────────────────────────────
def lighten_color(color, amount=0.55):
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(color))
    white = np.array([1, 1, 1])
    return mc.to_hex(c + (white - c) * amount)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 12

for model in server.keys():
    s = server[model]
    e = edge[model]

    base_color = model_colors[model]
    edge_color = lighten_color(base_color, amount=0.6)

    # server point (dark / original color)
    ax.scatter(s["infer_mem"], s["throughput"],
               color=base_color, s=110, zorder=3)

    # edge point (lighter version of same color)
    ax.scatter(e["infer_mem"], e["throughput"],
               color=edge_color, s=110, zorder=3)

    # arrow server → edge
    ax.annotate(
        "",
        xy=(e["infer_mem"], e["throughput"]),
        xytext=(s["infer_mem"], s["throughput"]),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray", alpha=0.7)
    )

    # label on edge point
    offset = (6, 6)
    if model in ["ModernTCN", "GTA"]:
        offset = (10, -10)

    ax.annotate(
        model,
        (e["infer_mem"], e["throughput"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE
    )

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_title("Server → Edge Device Transition", fontsize=TITLE_SIZE)
ax.set_xlabel("Peak inference memory (MB)", fontsize=LABEL_SIZE)
ax.set_ylabel("Throughput (samples/s)", fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3)

# legend (manual)
ax.scatter([], [], color="black", label="Server (original color)", s=80)
ax.scatter([], [], color="gray", label="Edge Device (lightened color)", s=80)
ax.legend()

plt.tight_layout()
plt.savefig('server_to_edge_transition.png', bbox_inches='tight', dpi=300)
plt.show()