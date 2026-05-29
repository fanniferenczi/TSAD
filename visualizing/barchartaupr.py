import numpy as np
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ['TimesNet', 'ModernTCN', 'TranAD', 'Anomaly Transformer', 'GTA']
DATASETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']

AUPR = np.array([
    # SMD    MSL    SMAP   SWaT   PSM
    [14.90, 14.85, 11.86, 10.81, 37.80],  # TimesNet
    [14.20, 14.80, 11.36,  9.60, 38.29],  # ModernTCN
    [50.19, 16.59, 10.16, 73.66, 46.01],  # TranAD
    [ 4.45, 10.42, 12.90, 11.10, 28.84],  # Anomaly Transformer
    [15.82, 11.62, 12.33,  8.62, 34.97],  # GTA
])

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_COLORS   = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
DATASET_COLORS = ['#333333', '#e05c00', '#1a7a4a', '#9b1c1c', '#4a2c8a']

n_models   = len(MODELS)
n_datasets = len(DATASETS)
bar_width  = 0.15

# ── Figure 1: Grouped bar chart by dataset ────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(n_datasets)
for i, model in enumerate(MODELS):
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax1.bar(x + offset, AUPR[i, :], width=bar_width,
            color=MODEL_COLORS[i], label=model,
            edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(DATASETS, fontsize=12)
ax1.set_ylabel('AUPR (%)', fontsize=12)
ax1.set_title('AUPR per Dataset', fontsize=13)
ax1.set_ylim(0, 100)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)
ax1.legend(title='Model', fontsize=9, title_fontsize=10, loc='upper right')

plt.tight_layout()
fig1.savefig('aupr_barchart_by_dataset.png', dpi=150, bbox_inches='tight')

# ── Figure 2: Grouped bar chart by model ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 6))

x = np.arange(n_models)
for j, dataset in enumerate(DATASETS):
    offset = (j - n_datasets / 2 + 0.5) * bar_width
    ax2.bar(x + offset, AUPR[:, j], width=bar_width,
            color=DATASET_COLORS[j], label=dataset,
            edgecolor='black', linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(MODELS, fontsize=11, rotation=10, ha='right')
ax2.set_ylabel('AUPR (%)', fontsize=12)
ax2.set_title('AUPR per Model', fontsize=13)
ax2.set_ylim(0, 100)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)
ax2.legend(title='Dataset', fontsize=9, title_fontsize=10, loc='upper right')

plt.tight_layout()
fig2.savefig('aupr_barchart_by_model.png', dpi=150, bbox_inches='tight')

plt.show()