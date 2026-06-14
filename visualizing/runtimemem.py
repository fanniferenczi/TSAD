import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
models = ['TimesNet', 'ModernTCN', 'TranAD', 'AnomTr.', 'GTA']
datasets = ['SMD', 'MSL', 'SMAP', 'SWaT', 'PSM']

train = np.array([
    [408.97,  410.13,  215.64,  411.52,  212.66],
    [470.28,  669.29,  176.75,  623.31,  176.80],
    [  2.47,    4.81,    1.20,    4.19,    1.20],
    [4228.20, 4351.70, 4346.71, 4350.73, 4346.66],
    [777.10, 1178.96,  616.12, 1019.17,  617.59],
])

infer = np.array([
    [295.39,  296.63,  104.74,  296.63,  105.66],
    [188.19,  264.98,   79.37,  247.19,   79.92],
    [  1.45,    2.72,    0.74,    2.39,    0.74],
    [3027.15, 3151.52, 3144.19, 3150.69, 3144.19],
    [314.77, 603.16,  303.65, 522.51,  304.65],
])

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

n_models      = len(models)
n_datasets    = len(datasets)
bar_width     = 0.13
group_gap     = 0.3
cluster_width = n_models * bar_width
group_centers = np.arange(n_datasets) * (cluster_width + group_gap)

def fmt(v):
    return f'{v:.1f}' if v < 10 else f'{v:,.0f}'

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 8))

for i, (model, color) in enumerate(zip(models, colors)):
    offsets = group_centers + i * bar_width - cluster_width / 2 + bar_width / 2

    # Training bar — faded, full width
    train_bars = ax.bar(offsets, train[i], width=bar_width,
                        color=color, alpha=0.25)

    # Inference bar — solid, slightly narrower
    infer_bars = ax.bar(offsets, infer[i], width=bar_width * 0.65,
                        color=color, alpha=0.9)

    for j in range(n_datasets):
        train_val = train[i, j]
        infer_val = infer[i, j]

        # Training value on top of training bar
        ax.text(offsets[j], train_val,
                fmt(train_val),
                ha='center', va='bottom',
                fontsize=8.5, color=color, alpha=0.6,
                transform=ax.transData)

        # Inference value on top of inference bar
        ax.text(offsets[j], infer_val,
                fmt(infer_val),
                ha='center', va='bottom',
                fontsize=8.5, color=color, alpha=1.0,      # up from 6
                fontweight='bold',
                transform=ax.transData)

# ── Log scale y-axis ──────────────────────────────────────────────────────────
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda val, _: f'{val:,.0f}' if val >= 1 else f'{val:.1f}'
))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1,
                                              numticks=20))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_ylabel('Memory (MB, log scale)', fontsize=15)
ax.set_ylim(0.5, 12000)

# ── Reference lines ───────────────────────────────────────────────────────────
for ref_mb, label in [(1, '1 MB'), (10, '10 MB'),
                       (100, '100 MB'), (1000, '1,000 MB'), (4000, '4,000 MB')]:
    ax.axhline(ref_mb, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.text(group_centers[-1] + cluster_width / 2 + 0.05,
            ref_mb, label,
            va='center', fontsize=11, color='grey', alpha=0.7)

# ── x-axis ────────────────────────────────────────────────────────────────────
ax.set_xticks(group_centers)
ax.set_xticklabels(datasets, fontsize=13)
ax.set_xlabel('Dataset', fontsize=15)
ax.set_xlim(group_centers[0] - cluster_width,
            group_centers[-1] + cluster_width + 0.4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.2)

# ── Legend ────────────────────────────────────────────────────────────────────
model_handles = [mpatches.Patch(color=c, alpha=0.9, label=m)
                 for m, c in zip(models, colors)]
train_patch = mpatches.Patch(color='grey', alpha=0.25,
                              label='Peak train. mem.')
infer_patch = mpatches.Patch(color='grey', alpha=0.9,
                              label='Peak infer. mem.')

ax.legend(handles=model_handles + [train_patch, infer_patch],
          loc='upper right', fontsize=10, frameon=False, ncol=2,
          bbox_to_anchor=(1.0, 1.07))

ax.set_title('Peak Training vs. Inference Memory',
             fontsize=19, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('runtime_memory_logscale.png', bbox_inches='tight', dpi=300)
plt.show()