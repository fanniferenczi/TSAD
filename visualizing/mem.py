import matplotlib.pyplot as plt
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
models    = ['TimesNet', 'ModernTCN', 'TranAD', 'Anom. Tr.', 'GTA']
datasets  = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']

w_alloc = np.array([
    [30.44, 30.46,  7.70, 30.45,  7.70],
    [ 2.65,  4.32,  0.59,  3.90,  0.59],
    [ 1.03,  2.10,  0.46,  1.81,  0.46],
    [28.77, 28.90, 28.67, 28.87, 28.67],
    [15.88, 16.26, 15.66, 16.16, 15.66],
])

dataset_colors = {
    'SMD':  '#4C72B0',
    'MSL':  '#DD8452',
    'SMAP': '#55A868',
    'SWaT': '#C44E52',
    'PSM':  '#8172B2',
}

fig, ax = plt.subplots(figsize=(10, 5))

y_positions = np.arange(len(models))

# ── Horizontal jitter strength ───────────────────────────────────────────────
np.random.seed(42)
jitter_strength = 0.15  # increase if overlaps are still too strong

for i, (model, row) in enumerate(zip(models, w_alloc)):
    ax.hlines(i, row.min(), row.max(),
              colors='lightgray', linewidth=2.5, zorder=1)

    # deterministic offsets so colors stay consistent per dataset
    base_offsets = np.linspace(-jitter_strength, jitter_strength, len(datasets))

    for j, (val, dataset) in enumerate(zip(row, datasets)):
        x_jitter = val + base_offsets[j]

        ax.scatter(x_jitter, i,
                   color=dataset_colors[dataset],
                   s=80,
                   alpha=0.8,
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=3,
                   label=dataset if i == 0 else '')

    ax.annotate(f'{row.min():.2f}',
                xy=(row.min(), i),
                xytext=(-6, 0), textcoords='offset points',
                ha='right', va='center', fontsize=8, color='gray')

    ax.annotate(f'{row.max():.2f}',
                xy=(row.max(), i),
                xytext=(6, 0), textcoords='offset points',
                ha='left', va='center', fontsize=8, color='gray')

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_yticks(y_positions)
ax.set_yticklabels(models, fontsize=12)
ax.set_xlabel('Allocated Weight Memory (MB)', fontsize=12)
ax.set_xlim(-2, 35)
ax.set_title('Allocated Weight Memory Variability Across Datasets',
             fontsize=13, pad=14)

ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
ax.invert_yaxis()

ax.legend(title='Dataset', loc='lower right',
          fontsize=9, title_fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('memory_alloc_variability.png', bbox_inches='tight', dpi=300)
plt.show()

print('Saved: memory_alloc_variability.png')