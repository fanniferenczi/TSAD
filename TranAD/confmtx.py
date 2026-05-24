import numpy as np
import matplotlib.pyplot as plt

dataset = 'SWaT'
model   = 'TranAD'

folder = f'test_results/anomaly_detection_{dataset}_{model}'

preds  = np.load(f'{folder}/predictions_raw.npy').astype(int)
labels = np.load(f'{folder}/ground_truth.npy').astype(int)

# ── Counts ─────────────────────────────────────────────────────────────────────
TP = int(((preds == 1) & (labels == 1)).sum())
TN = int(((preds == 0) & (labels == 0)).sum())
FP = int(((preds == 1) & (labels == 0)).sum())
FN = int(((preds == 0) & (labels == 1)).sum())

cm = np.array([[TP, FN],
               [FP, TN]])

total = cm.sum()
cm_pct = cm / total * 100

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)

for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)',
                ha='center', va='center', fontsize=12,
                color='white' if cm[i, j] > cm.max() / 2 else 'black')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Anomaly', 'Predicted Normal'], fontsize=10)
ax.set_yticklabels(['Actual Anomaly', 'Actual Normal'], fontsize=10)
ax.set_title(f'Confusion Matrix — {model} on {dataset}', fontsize=12)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nSummary — {model} on {dataset}")
print(f"  TP: {TP:,}  FP: {FP:,}  TN: {TN:,}  FN: {FN:,}")
print(f"  Precision:  {precision:.4f}")
print(f"  Recall:     {recall:.4f}")
print(f"  F1:         {f1:.4f}")

plt.tight_layout()
plt.savefig(f'confusion_matrix_{model}_{dataset}.png', dpi=150, bbox_inches='tight')
plt.show()