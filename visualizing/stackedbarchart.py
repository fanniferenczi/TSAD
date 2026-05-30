import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data (average across datasets, directly from Table) ──────────────────────
models = ['TimesNet', 'ModernTCN', 'TranAD†', 'Anom. Tr.', 'GTA']

# Exact weight memory (MB) — base of each bar
weight_exact_mb = [19.74, 2.37, 1.17, 18.43, 6.12]

# Overhead (MB) — stacked on top
overhead_mb = [310.71, 420.87, 0.95, 4296.02, 409.75]

# Peak training memory (MB) — for verification labels on top
train_peak_mb = [331.78, 423.29, 2.77, 4324.80, 841.79]

# Overhead ratio (overhead / weight_exact) — for annotation
overhead_ratio = [o / w for o, w in zip(overhead_mb, weight_exact_mb)]

# ── Layout ────────────────────────────────────────────────────────────────────
x = np.arange(len(models))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 7))

# Base stack: exact weight memory
bars_weights = ax.bar(x, weight_exact_mb, width,
                      label='Exact weight memory',
                      color='#4C72B0', alpha=0.85, zorder=3)

# Top stack: overhead
bars_overhead = ax.bar(x, overhead_mb, width,
                       bottom=weight_exact_mb,
                       label='Overhead (activations, gradients, buffers)',
                       color='#DD8452', alpha=0.85, zorder=3)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_yscale('log')
ax.set_ylabel('Memory (MB) — log scale', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_title('Peak Training Memory Breakdown: Weights vs. Overhead\n'
             '(average across datasets)',
             fontsize=13, pad=14)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f'{val:g}'))
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# ── Total peak training memory label on top of each bar ──────────────────────
for i, (total, ratio) in enumerate(zip(train_peak_mb, overhead_ratio)):
    ax.annotate(f'{total:.2f} MB',
                xy=(i, total),
                xytext=(0, 6), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Overhead ratio annotation just below the top label
    ax.annotate(f'OH: {ratio:.0f}×',
                xy=(i, total),
                xytext=(0, -14), textcoords='offset points',
                ha='center', va='top', fontsize=8, color='#DD8452')

# ── Weight memory label inside base bar ──────────────────────────────────────
for i, w in enumerate(weight_exact_mb):
    # Only annotate if bar is tall enough to be readable
    if w > 0.5:
        ax.annotate(f'{w:.2f}',
                    xy=(i, w / 2),
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

# ── Legend ────────────────────────────────────────────────────────────────────
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# ── Footnote ─────────────────────────────────────────────────────────────────
fig.text(0.01, -0.03,
         '† TranAD executes on CPU; all memory measured in RAM.\n'
         '  TranAD and GTA use float64 (×8 B/param); all others use float32 (×4 B/param).\n'
         '  OH = overhead ratio (overhead ÷ exact weight memory).',
         fontsize=8, color='gray', ha='left')

plt.tight_layout()
plt.savefig('memory_breakdown_stacked.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: memory_breakdown_stacked.pdf / .png')