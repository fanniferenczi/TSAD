import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data (average across datasets, directly from Table) ──────────────────────
models = ['TimesNet', 'ModernTCN', 'TranAD', 'Anom. Tr.', 'GTA']

# W_ex (exact weight memory, MB) — already accounts for fp32/fp64
weight_exact_mb = [19.74, 2.37, 1.17, 18.43, 6.12]

# Peak inference memory (MB)
# TranAD: CPU RAM, others: GPU VRAM
infer_mb = [219.81, 171.93, 1.61, 3123.55, 825.86]

# ── Layout ────────────────────────────────────────────────────────────────────
x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y-axis: exact weight memory (first bar, left)
bars1 = ax1.bar(x - width / 2, weight_exact_mb, width,
                label='Exact weight memory (MB)',
                color='#DD8452', alpha=0.85, zorder=3)

# Right y-axis: peak inference memory (second bar, right)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, infer_mb, width,
                label='Peak inference memory (MB)',
                color='#4C72B0', alpha=0.85, zorder=3)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.set_ylabel('Exact Weight Memory (MB) — log scale', fontsize=12)
ax2.set_ylabel('Peak Inference Memory (MB) — log scale', fontsize=12)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_title('Exact Weight Memory vs. Peak Inference Memory\n'
              '(average across datasets)',
              fontsize=13, pad=14)

ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f'{val:g}'))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f'{val:g}'))

ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# ── Value labels on bars ──────────────────────────────────────────────────────
def label_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

label_bars(ax1, bars1)
label_bars(ax2, bars2)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [bars1, bars2]
labels_legend = ['Exact weight memory (MB)', 'Peak inference memory (MB)']
ax1.legend(handles, labels_legend, loc='upper left', fontsize=10, framealpha=0.9)


plt.tight_layout()
plt.savefig('memory_inference_vs_weights.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: memory_inference_vs_weights.png')