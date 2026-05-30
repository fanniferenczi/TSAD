import matplotlib.pyplot as plt
import numpy as np

# ── Data (W_alloc per dataset, directly from tables) ─────────────────────────
models    = ['TimesNet', 'ModernTCN', 'TranAD', 'Anom. Tr.', 'GTA']
datasets  = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']

# Allocated weight memory (MB) — rows: models, cols: datasets
w_alloc = np.array([
    [30.44, 30.46,  7.70, 30.45,  7.70],  # TimesNet
    [ 2.65,  4.32,  0.59,  3.90,  0.59],  # ModernTCN
    [ 1.03,  2.10,  0.46,  1.81,  0.46],  # TranAD
    [28.77, 28.90, 28.67, 28.87, 28.67],  # Anom. Tr.
    [15.88, 16.26, 15.66, 16.16, 15.66],  # GTA
])

# ── Colors per dataset ────────────────────────────────────────────────────────
dataset_colors = {
    'SMD':  '#4C72B0',
    'MSL':  '#DD8452',
    'SMAP': '#55A868',
    'SWaT': '#C44E52',
    'PSM':  '#8172B2',
}

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

y_positions = np.arange(len(models))

for i, (model, row) in enumerate(zip(models, w_alloc)):
    # Range line: min to max
    ax.hlines(i, row.min(), row.max(),
              colors='lightgray', linewidth=2.5, zorder=1)

    # One dot per dataset
    for j, (val, dataset) in enumerate(zip(row, datasets)):
        ax.scatter(val, i,
                   color=dataset_colors[dataset],
                   s=80, zorder=3,
                   label=dataset if i == 0 else '')

    # Annotate min and max values
    ax.annotate(f'{row.min():.2f}',
                xy=(row.min(), i),
                xytext=(-6, 0), textcoords='offset points',
                ha='right', va='center', fontsize=8, color='gray')
    ax.annotate(f'{row.max():.2f}',
                xy=(row.max(), i),
                xytext=(6, 0), textcoords='offset points',
                ha='left', va='center', fontsize=8, color='gray')

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_yticks(y_positions)
ax.set_yticklabels(models, fontsize=12)
ax.set_xlabel('Allocated Weight Memory (MB)', fontsize=12)
ax.set_xlim(-2, 35)
ax.set_title('Allocated Weight Memory Variability Across Datasets',
             fontsize=13, pad=14)
ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
ax.invert_yaxis()

# ── Legend for datasets ───────────────────────────────────────────────────────
ax.legend(title='Dataset', loc='lower right',
          fontsize=9, title_fontsize=10, framealpha=0.9)


plt.tight_layout()
plt.savefig('memory_alloc_variability.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: memory_alloc_variability.pdf / .png')