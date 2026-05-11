"""
Radar Chart — Edge Deployment Suitability (Memory & Efficiency)
Master thesis: Time-Series Anomaly Detection

Axes (all normalized min-max, 0 = most efficient, 1 = least efficient):
  - Infer Peak Memory (MB)
  - Parameter Count
  - Weights (MB)
  - Activation Overhead (MB)
  - MACs (avg across datasets, streaming-normalized where applicable)
  - Latency (ms, avg across datasets)

Note: closer to center = more edge-friendly on that dimension.
MACs: streaming-normalized for TimesNet & ModernTCN; fvcore total for
      TranAD, Anomaly Transformer, GTA (streaming incompatible).
Latency averaged over SMD, MSL, SMAP, SWaT (PSM not available).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Raw averages across datasets ───────────────────────────────────────────────
MODELS = ['TimesNet', 'ModernTCN', 'TranAD', 'Anomaly\nTransformer', 'GTA']
MODELS_LEGEND = ['TimesNet', 'ModernTCN', 'TranAD', 'Anomaly Transformer', 'GTA']

AXES = [
    'Infer Peak\nMemory (MB)',
    'Parameter\nCount',
    'Weights\n(MB)',
    'Activation\nOverhead (MB)',
    'MACs',
    'Latency\n(ms)',
]

# Rows = models, columns = axes (same order as AXES)
RAW = np.array([
    # Infer peak  Params      Weights   Overhead    MACs          Latency
    [219.81,      4930957,    19.74,    310.71,     15090745,     119.712],  # TimesNet
    [171.93,       592827,     2.37,    420.87,       155533,       8.999],  # ModernTCN
    [  1.61,       145769,     1.17,      0.95,       974333,      11.588],  # TranAD
    [3123.55,     4829861,    18.43,   4296.02,    512317440,      33.519],  # Anomaly Transformer
    [ 825.86,      802577,     6.12,    409.75,     30238914,      65.023],  # GTA
])

# ── Min-max normalise (0 = best/most efficient, 1 = worst) ────────────────────
mins = RAW.min(axis=0)
maxs = RAW.max(axis=0)
NORM = (RAW - mins) / (maxs - mins)

# ── Radar setup ───────────────────────────────────────────────────────────────
N = len(AXES)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

MODEL_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
    values = NORM[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, label=MODELS_LEGEND[i])
    ax.fill(angles, values, color=color, alpha=0.08)

# ── Gridlines & labels ────────────────────────────────────────────────────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels(AXES, fontsize=10.5)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7, color='grey')
ax.set_ylim(0, 1)

# Annotate centre and edge
ax.text(0, 0, 'best', ha='center', va='center', fontsize=7, color='grey')

ax.set_title(
    'Edge Deployment Suitability\n(normalised, lower = more efficient)',
    fontsize=13, pad=20
)

ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.15),
          fontsize=9, title='Model', title_fontsize=10)

plt.tight_layout()
plt.savefig('radar_memory.png', dpi=150, bbox_inches='tight')
plt.show()