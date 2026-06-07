import matplotlib.pyplot as plt

# Average values from Table 3.7 (Server)
server = {
    "TimesNet": {"latency": 106.6, "throughput": 1606.7},
    "ModernTCN": {"latency": 9.7, "throughput": 14047.1},
    "TranAD": {"latency": 10.9, "throughput": 11545.2},
    "AnomTr.": {"latency": 33.7, "throughput": 3797.0},
    "GTA": {"latency": 64.0, "throughput": 2009.8},
}

# Average values from Table 3.8 (Edge)
edge = {
    "TimesNet": {"latency": 340.4, "throughput": 395.1},
    "ModernTCN": {"latency": 58.6, "throughput": 3321.6},
    "TranAD": {"latency": 107.8, "throughput": 1256.8},
    "AnomTr.": {"latency": 138.7, "throughput": 461.4},
    "GTA": {"latency": 880.3, "throughput": 146.6},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 12

# -------------------
# Server subplot
# -------------------
ax = axes[0]

for model, values in server.items():

    ax.scatter(values["latency"], values["throughput"], s=100)

    # Default position
    offset = (5, 5)

    # Custom label positions
    if model == "ModernTCN":
        offset = (10, -10)  # right and below
    elif model == "TimesNet":
        offset = (-45, 10)     # left of point

    ax.annotate(
    model,
    (values["latency"], values["throughput"]),
    xytext=offset,
    textcoords="offset points",
    fontsize=ANNOTATION_SIZE
)

ax.set_title("Server", fontsize=TITLE_SIZE)
ax.set_xlabel("Latency (ms) ←", fontsize=LABEL_SIZE)
ax.set_ylabel("Throughput (samples/s) →", fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3)

# -------------------
# Edge subplot
# -------------------
ax = axes[1]


for model, values in edge.items():

    ax.scatter(values["latency"], values["throughput"], s=100)

    # Default position
    offset = (5, 5)

    # Custom label positions
    if model == "ModernTCN":
        offset = (10, -10)  # right and below
    elif model == "GTA":
        offset = (-20, 10)    # move label to left of point

    ax.annotate(
        model,
        (values["latency"], values["throughput"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_SIZE
    )

ax.set_title("Edge Device", fontsize=TITLE_SIZE)
ax.set_xlabel("Latency (ms) ←", fontsize=LABEL_SIZE)
ax.set_ylabel("Throughput (samples/s) →", fontsize=LABEL_SIZE)
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.grid(True, alpha=0.3)

plt.suptitle("Latency vs Throughput (Average Across Datasets)", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig('latency_throughput.png', bbox_inches='tight', dpi=300)
plt.show()