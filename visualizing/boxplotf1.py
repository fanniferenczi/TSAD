"""
F1 Score Boxplots — grouped by dataset, one per model
Master thesis: Time-Series Anomaly Detection

Data source: results.xlsx (Sheet2)
Evaluation protocol: Point-Adjust (PA) F1
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
F1 = np.array([
    # SMD    MSL    SMAP   SWaT   PSM
    [83.80, 78.45, 68.47, 92.78, 95.85],  # TimesNet
    [83.28, 81.45, 67.46, 91.19, 96.48],  # ModernTCN
    [95.01, 94.95, 89.15, 81.43, 87.50],  # TranAD
    [92.68, 94.91, 96.41, 92.56, 97.31],  # Anomaly Transformer
    [90.67, 87.81, 76.00, 89.31, 96.41],  # GTA
])

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_COLORS  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
MODEL_MARKERS = ['o', 's', '^', 'D', 'P']
BOX_COLORS    = ['#aec6cf', '#ffb347', '#b5ead7', '#ff9aa2', '#c9c0d3']

# ── Figure 1: Boxplot grouped by dataset ──────────────────────────────────────
data_by_dataset = [F1[:, j] for j in range(len(DATASETS))]

fig1, ax1 = plt.subplots(figsize=(10, 6))

bp = ax1.boxplot(data_by_dataset, patch_artist=True, widths=0.5, showfliers=False,
                 medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], BOX_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for j in range(len(DATASETS)):
    for i in range(len(MODELS)):
        ax1.scatter(j + 1, F1[i, j], color=MODEL_COLORS[i],
                    marker=MODEL_MARKERS[i], s=70, zorder=5)

handles = [mlines.Line2D([], [], color=c, marker=m, linestyle='None',
           markersize=8, label=name)
           for c, m, name in zip(MODEL_COLORS, MODEL_MARKERS, MODELS)]
ax1.legend(handles=handles, title='Model', fontsize=9, title_fontsize=10, loc='lower right')

ax1.set_xticks(range(1, len(DATASETS) + 1))
ax1.set_xticklabels(DATASETS, fontsize=12)
ax1.set_ylabel('F1 Score (%)', fontsize=12)
ax1.set_title('F1 Score Distribution per Dataset', fontsize=13)
ax1.set_ylim(55, 102)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)

plt.tight_layout()
fig1.savefig('f1_boxplot_by_dataset.png', dpi=150, bbox_inches='tight')

# ── Figure 2: Boxplot grouped by model ────────────────────────────────────────
# Each box = F1 distribution across the 5 datasets for one model
data_by_model = [F1[i, :] for i in range(len(MODELS))]

DATASET_MARKERS = ['o', 's', '^', 'D', 'P']
DATASET_COLORS  = ['#333333', '#e05c00', '#1a7a4a', '#9b1c1c', '#4a2c8a']

fig2, ax2 = plt.subplots(figsize=(10, 6))

bp2 = ax2.boxplot(data_by_model, patch_artist=True, widths=0.5, showfliers=False,
                  medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp2['boxes'], MODEL_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

MANUAL_JITTER = {
    (2, 0): 0.08,   # TranAD, SMD → shift right
    (3, 0): 0.08,   # Anomaly Transformer, SMD → shift right
}

for i in range(len(MODELS)):
    for j in range(len(DATASETS)):
        x_offset = MANUAL_JITTER.get((i, j), 0.0)
        ax2.scatter(i + 1 + x_offset, F1[i, j], color=DATASET_COLORS[j],
                    marker=DATASET_MARKERS[j], s=70, zorder=5,
                    edgecolors='black', linewidths=0.5)

handles2 = [mlines.Line2D([], [], color=c, marker=m, linestyle='None',
            markersize=8, label=name)
            for c, m, name in zip(DATASET_COLORS, DATASET_MARKERS, DATASETS)]
ax2.legend(handles=handles2, title='Dataset', fontsize=9, title_fontsize=10, loc='lower right')

ax2.set_xticks(range(1, len(MODELS) + 1))
ax2.set_xticklabels(MODELS, fontsize=11, rotation=10, ha='right')
ax2.set_ylabel('F1 Score (%)', fontsize=12)
ax2.set_title('F1 Score Distribution per Model', fontsize=13)
ax2.set_ylim(55, 102)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
fig2.savefig('f1_boxplot_by_model.png', dpi=150, bbox_inches='tight')

plt.show()