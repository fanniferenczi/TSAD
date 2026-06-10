import matplotlib.pyplot as plt
import numpy as np

# Average values from Table 3.7 (Server)
server = {
    "TimesNet": {"latency": 106.6, "throughput": 1606.7},
    "ModernTCN": {"latency": 9.7,  "throughput": 14047.1},
    "TranAD":    {"latency": 10.9, "throughput": 11545.2},
    "AnomTr.":   {"latency": 33.7, "throughput": 3797.0},
    "GTA":       {"latency": 64.0, "throughput": 2009.8},
}

# Average values from Table 3.8 (Edge)
edge = {
    "TimesNet": {"latency": 340.4, "throughput": 395.1},
    "ModernTCN": {"latency": 58.6, "throughput": 3321.6},
    "TranAD":    {"latency": 107.8, "throughput": 1256.8},
    "AnomTr.":   {"latency": 138.7, "throughput": 461.4},
    "GTA":       {"latency": 880.3, "throughput": 146.6},
}

# Colorblind-friendly palette (Wong 2011)
COLOR_SERVER = "#0077BB"
COLOR_EDGE   = "#EE7733"

TITLE_SIZE  = 15
LABEL_SIZE  = 13
TICK_SIZE   = 11
LEGEND_SIZE = 11

# -----------------------------------------------
# Figure 1: Latency — horizontal grouped bar chart
# Sorted by edge latency ascending; reversed so best (smallest) is at top
# -----------------------------------------------
models_lat = sorted(server.keys(), key=lambda m: edge[m]["latency"], reverse=True)
y_lat  = np.arange(len(models_lat))
height = 0.35

latency_server = [server[m]["latency"] for m in models_lat]
latency_edge   = [edge[m]["latency"]   for m in models_lat]

fig1, ax1 = plt.subplots(figsize=(9, 4))

ax1.barh(y_lat + height / 2, latency_server, height, label="Server", color=COLOR_SERVER)
ax1.barh(y_lat - height / 2, latency_edge,   height, label="Edge",   color=COLOR_EDGE)

# Annotate edge bars with the latency increase factor (edge / server)
for i, m in enumerate(models_lat):
    factor = edge[m]["latency"] / server[m]["latency"]
    ax1.text(
        latency_edge[i] + 8, y_lat[i] - height / 2,
        f"×{factor:.1f}",
        va="center", ha="left", fontsize=TICK_SIZE, color=COLOR_EDGE, fontweight="bold"
    )

# Extend x-axis slightly to make room for labels
ax1.set_xlim(right=ax1.get_xlim()[1] * 1.15)

ax1.set_yticks(y_lat)
ax1.set_yticklabels(models_lat, fontsize=TICK_SIZE)
ax1.set_xlabel("Latency (ms)  ←", fontsize=LABEL_SIZE)
ax1.set_title("Latency (avg. across datasets)", fontsize=TITLE_SIZE)
ax1.tick_params(axis='x', labelsize=TICK_SIZE)
ax1.legend(fontsize=LEGEND_SIZE)
ax1.grid(True, axis='x', alpha=0.3)
ax1.set_axisbelow(True)

fig1.tight_layout()
fig1.savefig('latency.png', bbox_inches='tight', dpi=300)

# -----------------------------------------------
# Figure 2: Throughput — vertical grouped bar chart
# -----------------------------------------------
models = sorted(server.keys(), key=lambda m: edge[m]["throughput"], reverse=True)
x     = np.arange(len(models))
width = 0.35

throughput_server = [server[m]["throughput"] for m in models]
throughput_edge   = [edge[m]["throughput"]   for m in models]

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.bar(x - width / 2, throughput_server, width, label="Server", color=COLOR_SERVER)
ax2.bar(x + width / 2, throughput_edge,   width, label="Edge",   color=COLOR_EDGE)

# Annotate edge bars with how many times worse edge is vs server (server / edge)
for i, m in enumerate(models):
    factor = throughput_server[i] / throughput_edge[i]
    ax2.text(
        x[i] + width / 2, throughput_edge[i] + 80,
        f"÷{factor:.1f}",
        ha="center", va="bottom", fontsize=TICK_SIZE, color=COLOR_EDGE, fontweight="bold"
    )

# Extend y-axis slightly to make room for labels
ax2.set_ylim(top=ax2.get_ylim()[1] * 1.12)

ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=TICK_SIZE)
ax2.set_ylabel("Throughput (samples/s) →", fontsize=LABEL_SIZE)
ax2.set_title("Throughput (avg. across datasets)", fontsize=TITLE_SIZE)
ax2.tick_params(axis='y', labelsize=TICK_SIZE)
ax2.legend(fontsize=LEGEND_SIZE)
ax2.grid(True, axis='y', alpha=0.3)
ax2.set_axisbelow(True)

fig2.tight_layout()
fig2.savefig('throughput.png', bbox_inches='tight', dpi=300)

plt.show()