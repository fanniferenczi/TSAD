"""
AU-PR Boxplots — grouped by dataset and by model
Master thesis: Time-Series Anomaly Detection

Data source: results.xlsx (Sheet2)
Models: TimesNet, ModernTCN, TranAD, Anomaly Transformer, GTA
Datasets: SMD, MSL, SMAP, SWaT, PSM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ['TimesNet', 'ModernTCN', 'TranAD', 'Anomaly Transformer', 'GTA']
DATASETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']

# Rows = models, Columns = datasets
AUPR = np.array([
    # SMD    MSL    SMAP   SWaT   PSM
    [14.90, 14.85, 11.86, 10.81, 37.80],  # TimesNet
    [14.20, 14.80, 11.36,  9.60, 38.29],  # ModernTCN
    [50.19, 16.59, 10.16, 73.66, 46.01],  # TranAD
    [ 4.45, 10.42, 12.90, 11.10, 28.84],  # Anomaly Transformer
    [15.82, 11.62, 12.33,  8.62, 34.97],  # GTA
])

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_COLORS  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
MODEL_MARKERS = ['o', 's', '^', 'D', 'P']
BOX_COLORS    = ['#aec6cf', '#ffb347', '#b5ead7', '#ff9aa2', '#c9c0d3']

DATASET_MARKERS = ['o', 's', '^', 'D', 'P']
DATASET_COLORS  = ['#333333', '#e05c00', '#1a7a4a', '#9b1c1c', '#4a2c8a']

# ── Figure 1: Boxplot grouped by dataset ──────────────────────────────────────
data_by_dataset = [AUPR[:, j] for j in range(len(DATASETS))]

fig1, ax1 = plt.subplots(figsize=(10, 6))

bp = ax1.boxplot(data_by_dataset, patch_artist=True, widths=0.5, showfliers=False,
                 medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], BOX_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for j in range(len(DATASETS)):
    for i in range(len(MODELS)):
        ax1.scatter(j + 1, AUPR[i, j], color=MODEL_COLORS[i],
                    marker=MODEL_MARKERS[i], s=70, zorder=5)

handles = [mlines.Line2D([], [], color=c, marker=m, linestyle='None',
           markersize=8, label=name)
           for c, m, name in zip(MODEL_COLORS, MODEL_MARKERS, MODELS)]
ax1.legend(handles=handles, title='Model', fontsize=9, title_fontsize=10, loc='upper right')

ax1.set_xticks(range(1, len(DATASETS) + 1))
ax1.set_xticklabels(DATASETS, fontsize=12)
ax1.set_ylabel('AUPR (%)', fontsize=12)
ax1.set_title('AUPR Distribution per Dataset', fontsize=13)
ax1.set_ylim(0, 100)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)

plt.tight_layout()
fig1.savefig('aupr_boxplot_by_dataset.png', dpi=150, bbox_inches='tight')

# ── Figure 2: Boxplot grouped by model ────────────────────────────────────────
data_by_model = [AUPR[i, :] for i in range(len(MODELS))]

fig2, ax2 = plt.subplots(figsize=(10, 6))

bp2 = ax2.boxplot(data_by_model, patch_artist=True, widths=0.5, showfliers=False,
                  medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp2['boxes'], MODEL_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

MANUAL_JITTER = {
    (0, 0): 0.08,   # TimesNet, SMD → shift right
    (1, 0): 0.08,   # ModernTCN, SMD → shift right
    (3, 1): 0.08,   # Anomaly Transformer, MSL → shift right
    (4, 1): 0.08,   # GTA, MSL → shift right
}

for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        x_offset = MANUAL_JITTER.get((i, j), 0.0)
        ax2.scatter(i + 1 + x_offset, AUPR[i, j], color=DATASET_COLORS[j],
                    marker=DATASET_MARKERS[j], s=70, zorder=5,
                    edgecolors='black', linewidths=0.5)

handles2 = [mlines.Line2D([], [], color=c, marker=m, linestyle='None',
            markersize=8, label=name)
            for c, m, name in zip(DATASET_COLORS, DATASET_MARKERS, DATASETS)]
ax2.legend(handles=handles2, title='Dataset', fontsize=9, title_fontsize=10, loc='upper right')

ax2.set_xticks(range(1, len(MODELS) + 1))
ax2.set_xticklabels(MODELS, fontsize=11, rotation=10, ha='right')
ax2.set_ylabel('AUPR (%)', fontsize=12)
ax2.set_title('AUPR Distribution per Model', fontsize=13)
ax2.set_ylim(0, 100)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
fig2.savefig('aupr_boxplot_by_model.png', dpi=150, bbox_inches='tight')

plt.show()