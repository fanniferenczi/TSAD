import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('anomaly_segments.csv')

colors = {
    'MSL':  '#E07B39',
    'SMAP': '#4CAF50',
    'SMD':  '#5B8DB8',
    'PSM':  '#7B68EE',
    'SWaT': '#B22222'
}

datasets = list(colors.keys())
data_per_dataset = [df[df['dataset'] == d]['length'].values for d in datasets]

# Two subplots sharing x-axis, different y ranges
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7),
                                      sharex=True,
                                      gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.05)

for ax in [ax_top, ax_bot]:
    bp = ax.boxplot(data_per_dataset,
                    patch_artist=True,
                    notch=False,
                    showfliers=True,
                    medianprops=dict(color='black', linewidth=2))

    for patch, dataset in zip(bp['boxes'], datasets):
        patch.set_facecolor(colors[dataset])
        patch.set_alpha(0.85)

    for flier, dataset in zip(bp['fliers'], datasets):
        flier.set(marker='o', markerfacecolor=colors[dataset],
                  markeredgecolor='white', markersize=4, alpha=0.5)

    ax.set_xticks(range(1, len(datasets) + 1))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Set y ranges: top shows the high outliers, bottom shows the bulk
ax_top.set_ylim(5000, df['length'].max() + 500)
ax_bot.set_ylim(0, 5000)

# Hide the inner spines at the cut
ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(bottom=False)

# Draw the cut marks
d = 0.01
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Labels
ax_bot.set_xticklabels(datasets, fontsize=11)
ax_bot.set_ylabel('Segment Length', fontsize=12)
ax_top.set_ylabel('Segment Length', fontsize=12)
fig.suptitle('Anomaly Segment Length Distribution per Dataset', fontsize=13)

# Summary table
col_labels = datasets
row_labels = ['n', 'median', 'mean', 'max']
table_data = [
    [str(len(d)) for d in data_per_dataset],
    [f'{np.median(d):.0f}' for d in data_per_dataset],
    [f'{np.mean(d):.0f}' for d in data_per_dataset],
    [str(d.max()) for d in data_per_dataset],
]

table = ax_bot.table(cellText=table_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='bottom',
                     bbox=[0, -0.42, 1, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)

plt.subplots_adjust(bottom=0.28)
plt.savefig('segment_length_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()