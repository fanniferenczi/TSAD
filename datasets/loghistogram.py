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

# Define meaningful bins with readable labels
# Length 1 is its own bin, rest are log-spaced but with clean labels
bin_edges =  [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, float('inf')]
bin_labels = ['1', '2-4', '5-9', '10-24', '25-49', '50-99', 
              '100-249', '250-499', '500-999', '1k-5k', '>5k']

n_bins = len(bin_labels)
x = np.arange(n_bins)
width = 0.15  # width of each bar
datasets = list(colors.keys())

fig, ax = plt.subplots(figsize=(14, 5))

for i, (dataset, color) in enumerate(colors.items()):
    data = df[df['dataset'] == dataset]['length'].values
    counts, _ = np.histogram(data, bins=bin_edges)
    offset = (i - len(datasets) / 2) * width + width / 2
    bars = ax.bar(x + offset, counts, width=width, 
                  color=color, edgecolor='white',
                  linewidth=0.5, alpha=0.85,
                  label=f'{dataset} (n={len(data)}, median={np.median(data):.0f})')

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=10)
ax.set_xlabel('Segment Length', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Anomaly Segment Length Distribution per Dataset', fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('segment_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()