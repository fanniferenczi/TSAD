import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

dataset = 'SWaT'

models = ['TranAD', 'TimesNet', 'ModernTCN', 'AnomalyTransformer', 'GTA']

# Colorblind-friendly palette (Wong, 2011)
colors = {
    'TranAD':             '#0072B2',  # blue
    'TimesNet':           '#E69F00',  # orange
    'ModernTCN':          '#009E73',  # green
    'AnomalyTransformer': '#D55E00',  # vermillion
    'GTA':                '#CC79A7',  # pink
}

fig, ax = plt.subplots(figsize=(8, 6))

no_skill = None  # will be set from first successfully loaded file

for model in models:
    try:
        scores = np.load(f'scores_{model}_{dataset}_loss.npy')
        labels = np.load(f'scores_{model}_{dataset}_labels.npy').astype(int)
    except FileNotFoundError:
        print(f"  Skipping {model} — file not found")
        continue

    if no_skill is None:
        no_skill = labels.sum() / len(labels)

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    ax.plot(recall, precision,
            color=colors[model],
            linewidth=1.5,
            label=f'{model} (AP = {ap:.3f})')

    print(f"{model:25s}  AP = {ap:.4f}  anomaly rate = {labels.sum()/len(labels):.4f}")

# No-skill baseline — use the shared anomaly rate
# All models are evaluated on the same test set so no_skill is identical across models
if no_skill is not None:
    ax.axhline(y=no_skill, color='grey', linestyle='--', linewidth=0.8,
               label=f'No-skill baseline ({no_skill:.3f})')

ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title(f'Precision-Recall Curves — {dataset}', fontsize=12)
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'pr_curves_all_{dataset}.png', dpi=150, bbox_inches='tight')
plt.show()