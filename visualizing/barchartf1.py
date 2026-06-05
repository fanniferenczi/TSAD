import numpy as np
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = ['TimesNet', 'ModernTCN', 'TranAD', 'Anomaly Transformer', 'GTA']
DATASETS = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']

F1 = np.array([
    # SMD    MSL    SMAP   SWaT   PSM
    [83.80, 78.45, 68.47, 92.78, 95.85],  # TimesNet
    [83.28, 81.45, 67.46, 91.19, 96.48],  # ModernTCN
    [95.01, 94.95, 89.15, 81.43, 87.50],  # TranAD
    [92.68, 94.91, 96.41, 92.56, 97.31],  # Anomaly Transformer
    [90.67, 87.81, 76.00, 89.31, 96.41],  # GTA
])

# ── Style ─────────────────────────────────────────────────────────────────────
MODEL_COLORS   = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
DATASET_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

n_models   = len(MODELS)
n_datasets = len(DATASETS)
bar_width  = 0.15

# ── Figure 1: Grouped bar chart by dataset ────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(n_datasets)
for i, model in enumerate(MODELS):
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax1.bar(x + offset, F1[i, :], width=bar_width,
            color=MODEL_COLORS[i], label=model,
            edgecolor='black', linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(DATASETS, fontsize=13)
ax1.set_ylabel('F1 Score (%)', fontsize=13)
ax1.set_title('F1 Score per Dataset (Point-Adjust Evaluation)', fontsize=16, fontweight="bold")
ax1.set_ylim(0, 102)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)
ax1.legend(title='Model', fontsize=9, title_fontsize=10, loc='lower right')

plt.tight_layout()
fig1.savefig('f1_barchart_by_dataset.png', dpi=150, bbox_inches='tight')

# ── Figure 2: Grouped bar chart by model ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(12, 6))

x = np.arange(n_models)
for j, dataset in enumerate(DATASETS):
    offset = (j - n_datasets / 2 + 0.5) * bar_width
    ax2.bar(x + offset, F1[:, j], width=bar_width,
            color=DATASET_COLORS[j], label=dataset,
            edgecolor='black', linewidth=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels(MODELS, fontsize=13, rotation=10, ha='right')
ax2.set_ylabel('F1 Score (%)', fontsize=13)
ax2.set_title('F1 Score per Model (Point-Adjust Evaluation)', fontsize=16, fontweight="bold")
ax2.set_ylim(0, 102)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5)
ax2.set_axisbelow(True)
ax2.legend(title='Dataset', fontsize=9, title_fontsize=10, loc='lower right')

plt.tight_layout()
fig2.savefig('f1_barchart_by_model.png', dpi=150, bbox_inches='tight')

plt.show()