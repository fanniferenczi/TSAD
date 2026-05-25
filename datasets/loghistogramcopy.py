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

# True logarithmic bins: each bin spans one order of magnitude
# Automatically go up to the max value in the data
max_val = df['length'].max()
max_power = int(np.ceil(np.log10(max_val + 1)))

bin_edges = [10**i for i in range(max_power + 1)]
bin_labels = []
for i in range(len(bin_edges) - 1):
    lo = bin_edges[i]
    hi = bin_edges[i + 1]
    if lo == 1:
        bin_labels.append(f'1-9')
    else:
        bin_labels.append(f'{lo:,}-{hi-1:,}')

n_bins = len(bin_labels)
x = np.arange(n_bins)
width = 0.15
datasets = list(colors.keys())

fig, ax = plt.subplots(figsize=(12, 5))

for i, (dataset, color) in enumerate(colors.items()):
    data = df[df['dataset'] == dataset]['length'].values
    counts, _ = np.histogram(data, bins=bin_edges)
    offset = (i - len(datasets) / 2) * width + width / 2
    ax.bar(x + offset, counts, width=width,
           color=color, edgecolor='white',
           linewidth=0.5, alpha=0.85,
           label=f'{dataset} (n={len(data)}, median={np.median(data):.0f})')

ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=10)
ax.set_xlabel('Segment Length', fontsize=12)
ax.set_ylabel('Count (log scale)', fontsize=12)
ax.set_title('Anomaly Segment Length Distribution per Dataset', fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis='y', which='both', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('segment_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()