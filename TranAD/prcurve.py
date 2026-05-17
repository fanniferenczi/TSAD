import numpy as np
import matplotlib.pyplot as plt

dataset = 'SWaT'
model   = 'TranAD'

scores = np.load(f'scores_{model}_{dataset}_loss.npy')
labels = np.load(f'scores_{model}_{dataset}_labels.npy').astype(int)

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(labels, scores)
ap = average_precision_score(labels, scores)

no_skill = labels.sum() / len(labels)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(recall, precision, color='steelblue', linewidth=1.5,
        label=f'PR curve (AP = {ap:.3f})')

ax.axhline(y=no_skill, color='grey', linestyle='--', linewidth=0.8,
           label=f'No-skill baseline ({no_skill:.3f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall Curve — {model} on {dataset}')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'pr_curve_{model}_{dataset}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSummary — {model} on {dataset}")
print(f"  Average Precision (AP):   {ap:.4f}")
print(f"  Anomaly rate:             {no_skill:.4f}")