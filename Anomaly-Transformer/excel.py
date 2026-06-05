"""
AnomalyTransformer SMAP — Segment-Level Detection Report
=========================================================
Output: smap_anomalytransformer_segments.xlsx
  columns: segment, global_start, global_end, length, detected, pts_flagged
"""

import numpy as np
import pandas as pd

# ── USER CONFIG ───────────────────────────────────────────────────────────────

LABEL_FILE  = "/home/fzf/dev/TSAD/datasets/SMAP/SMAP_test_label.npy"
PRED_FILE   = "/home/fzf/dev/TSAD/Anomaly-Transformer/scores_AnomalyTransformer_SMAP_predictions_raw.npy"
OUTPUT_FILE = "smap_anomalytransformer_segments.xlsx"

# ── LOAD ──────────────────────────────────────────────────────────────────────

labels = np.load(LABEL_FILE).astype(int)
n_ts   = len(labels)

# AnomalyTransformer saves per-timestep predictions, 17 points short — pad tail
raw  = np.load(PRED_FILE).astype(int)
pred = np.pad(raw, (0, n_ts - len(raw)))

print(f"Labels : {n_ts:,} timesteps, {labels.sum():,} anomalous")
print(f"Pred   : {pred.sum():,} flagged")

# ── EXTRACT SEGMENTS & EVALUATE ──────────────────────────────────────────────

rows    = []
seg_idx = 0
in_seg  = False

for i, v in enumerate(labels):
    if v == 1 and not in_seg:
        start, in_seg = i, True
    elif v == 0 and in_seg:
        end   = i - 1
        n_det = int(pred[start:end+1].sum())
        seg_idx += 1
        rows.append({
            'segment':      seg_idx,
            'global_start': start,
            'global_end':   end,
            'length':       end - start + 1,
            'detected':     n_det > 0,
            'pts_flagged':  n_det,
        })
        in_seg = False

if in_seg:
    end   = n_ts - 1
    n_det = int(pred[start:end+1].sum())
    seg_idx += 1
    rows.append({
        'segment':      seg_idx,
        'global_start': start,
        'global_end':   end,
        'length':       end - start + 1,
        'detected':     n_det > 0,
        'pts_flagged':  n_det,
    })

# ── SAVE ─────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}  ({len(df)} segments)")
print(f"Detected : {df['detected'].sum()}/{len(df)}  ({100*df['detected'].mean():.1f}%)")
print(f"Missed   : {(~df['detected']).sum()}/{len(df)}")