import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── USER CONFIG ───────────────────────────────────────────────────────────────

DATA_FILE      = "/home/fzf/dev/TSAD/datasets/SMAP/SMAP_test.npy"
LABEL_FILE     = "/home/fzf/dev/TSAD/datasets/SMAP/SMAP_test_label.npy"
PRED_FILE      = "/home/fzf/dev/TSAD/TimesNet/Time-Series-Library/test_results/anomaly_detection_SMAP_TimesNet_SMAP_ftM_sl100_ll48_pl0_dm32_nh8_el3_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/predictions_raw.npy"
SCORES_FILE    = "/home/fzf/dev/TSAD/TimesNet/Time-Series-Library/test_results/anomaly_detection_SMAP_TimesNet_SMAP_ftM_sl100_ll48_pl0_dm32_nh8_el3_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/anomaly_scores.npy"
THRESHOLD_FILE = "/home/fzf/dev/TSAD/TimesNet/Time-Series-Library/test_results/anomaly_detection_SMAP_TimesNet_SMAP_ftM_sl100_ll48_pl0_dm32_nh8_el3_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/threshold.npy"
RECON_FILE     = "/home/fzf/dev/TSAD/TimesNet/Time-Series-Library/test_results/anomaly_detection_SMAP_TimesNet_SMAP_ftM_sl100_ll48_pl0_dm32_nh8_el3_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_test_0/test_output.npy"
OUTPUT_FILE    = "smap_timesnet_signal_recon.png"

G_START = 371000
G_END   = 393000

WONG = dict(blue="#0072B2", vermillion="#D55E00", sky_blue="#56B4E9",
            orange="#E69F00", green="#009E73", black="#000000")

# ── LOAD ──────────────────────────────────────────────────────────────────────

data      = np.load(DATA_FILE)
labels    = np.load(LABEL_FILE).astype(int)
threshold = float(np.load(THRESHOLD_FILE)[0])
recon     = np.load(RECON_FILE).astype(np.float32)
n_ts      = len(labels)

def load_window_array(path, n_ts, agg='or'):
    raw = np.load(path)
    if len(raw) == n_ts:
        return raw
    seq_len   = next(L for L in range(1, 1000) if (n_ts - L + 1) * L == len(raw))
    n_windows = n_ts - seq_len + 1
    raw_2d    = raw.reshape(n_windows, seq_len)
    out       = np.zeros(n_ts, dtype=raw.dtype)
    if agg == 'or':
        raw_2d = raw_2d.astype(np.int64)
        for j in range(seq_len):
            out[j:j+n_windows] |= raw_2d[:, j]
    else:
        for j in range(seq_len):
            np.maximum(out[j:j+n_windows], raw_2d[:, j], out=out[j:j+n_windows])
    return out

pred   = load_window_array(PRED_FILE,   n_ts, agg='or').astype(int)
scores = load_window_array(SCORES_FILE, n_ts, agg='max').astype(float)

# ── SLICE WINDOW ──────────────────────────────────────────────────────────────

t      = np.arange(G_START, G_END + 1)
signal = data[G_START:G_END+1, 0].astype(float)
rec    = recon[G_START:G_END+1, 0]
gt     = labels[G_START:G_END+1]
pr     = pred[G_START:G_END+1]
sc     = scores[G_START:G_END+1]

# ── FIGURE ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 7))
gs  = GridSpec(2, 1, figure=fig, hspace=0.08)

fig.suptitle(
    f"SMAP — TimesNet  |  Global timesteps {G_START:,}–{G_END:,}\n",
    fontsize=16, fontweight="bold"
)

# Panel 1: signal vs reconstruction + TimesNet detections
ax2 = fig.add_subplot(gs[0])
ax2.plot(t, signal, color=WONG["black"], lw=0.9, zorder=4, label="Input signal")
ax2.plot(t, rec,    color=WONG["green"], lw=0.9, zorder=3, alpha=0.85,
         label="Reconstruction", ls="--")
ax2.fill_between(t, signal.min(), signal.max(), where=gt.astype(bool),
                 color=WONG["vermillion"], alpha=0.20, zorder=2, step="post")
ax2.fill_between(t, signal.min(), signal.max(), where=pr.astype(bool),
                 color=WONG["sky_blue"], alpha=0.35, zorder=3, step="post")
ax2.set_xlim(G_START, G_END)
ax2.set_ylabel("Value", fontsize=12)
ax2.set_title("Signal vs TimesNet reconstruction", fontsize=12, pad=4)
ax2.tick_params(labelbottom=False, labelsize=8)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(loc="upper right", bbox_to_anchor=(1.10, 0.945), fontsize=10, framealpha=0.8)

# Panel 2: reconstruction error + threshold + TimesNet detections
ax3 = fig.add_subplot(gs[1], sharex=ax2)
ax3.plot(t, sc, color=WONG["orange"], lw=0.8, zorder=3, label="Reconstruction error")
ax3.axhline(threshold, color=WONG["vermillion"], lw=1.2, ls="--", zorder=4,
            label=f"Threshold ({threshold:.4f})")
ax3.fill_between(t, 0, sc, where=(sc > threshold),
                 color=WONG["vermillion"], alpha=0.30, zorder=2)
ax3.fill_between(t, 0, sc.max(), where=gt.astype(bool),
                 color=WONG["vermillion"], alpha=0.10, zorder=1, step="post")
ax3.fill_between(t, 0, sc.max(), where=pr.astype(bool),
                 color=WONG["sky_blue"], alpha=0.25, zorder=2, step="post")
ax3.set_xlim(G_START, G_END)
ax3.set_ylabel("Error", fontsize=12)
ax3.set_xlabel("Global timestep", fontsize=12)
ax3.set_title("Reconstruction error vs threshold", fontsize=12, pad=4)
ax3.tick_params(labelsize=8)
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.10, 0.95), framealpha=0.8)

# Shared legend
handles = [
    mpatches.Patch(color=WONG["vermillion"], alpha=0.4, label="Ground truth anomaly"),
    mpatches.Patch(color=WONG["sky_blue"],   alpha=0.5, label="TimesNet reported anomaly"),
]
fig.legend(handles=handles, loc="upper right", fontsize=10,
           framealpha=0.85, bbox_to_anchor=(0.99, 0.99))

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved: {OUTPUT_FILE}")
