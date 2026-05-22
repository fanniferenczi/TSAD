# ── Inference speed for ModernTCN on ALL datasets (CUDA Events) ───────────
import torch
import argparse
import numpy as np
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, './')
from exp.exp_anomaly_detection import Exp_Anomaly_Detection

DATASETS = {
    'SMD':  dict(seq_len=100, label_len=0, pred_len=0, enc_in=38, c_out=38,
                 ffn_ratio=1, patch_size=8, patch_stride=4,
                 num_blocks=[1], large_size=[51], small_size=[5], dims=[64],
                 dropout=0.1, head_dropout=0.0,
                 use_multi_scale=False, small_kernel_merged=False),

    'MSL':  dict(seq_len=100, label_len=0, pred_len=0, enc_in=55, c_out=55,
                 ffn_ratio=1, patch_size=8, patch_stride=4,
                 num_blocks=[1], large_size=[51], small_size=[5], dims=[64],
                 dropout=0.1, head_dropout=0.0,
                 use_multi_scale=False, small_kernel_merged=False),

    'SMAP': dict(seq_len=100, label_len=0, pred_len=0, enc_in=25, c_out=25,
                 ffn_ratio=1, patch_size=8, patch_stride=4,
                 num_blocks=[1], large_size=[51], small_size=[5], dims=[32],
                 dropout=0.1, head_dropout=0.0,
                 use_multi_scale=False, small_kernel_merged=False),

    'SWaT': dict(seq_len=100, label_len=0, pred_len=0, enc_in=51, c_out=51,
                 ffn_ratio=1, patch_size=8, patch_stride=4,
                 num_blocks=[1], large_size=[51], small_size=[5], dims=[64],
                 dropout=0.1, head_dropout=0.0,
                 use_multi_scale=False, small_kernel_merged=False),

    'PSM':  dict(seq_len=100, label_len=0, pred_len=0, enc_in=25, c_out=25,
                 ffn_ratio=1, patch_size=8, patch_stride=4,
                 num_blocks=[1], large_size=[51], small_size=[5], dims=[32],
                 dropout=0.1, head_dropout=0.0,
                 use_multi_scale=False, small_kernel_merged=False),
}

# ── Fixed args shared across all datasets ───────────────────────────────
BASE_ARGS = dict(
    task_name="anomaly_detection",
    is_training=0,
    model="ModernTCN",
    features="M", target="OT", freq="h", embed="timeF",
    checkpoints="./checkpoints/",
    # run.py defaults used when scripts omit values
    seq_len=96, label_len=48, pred_len=96,
    stem_ratio=6, downsample_ratio=2,
    ffn_ratio=2, patch_size=16, patch_stride=8,
    num_blocks=[1, 1, 1, 1],
    large_size=[31, 29, 27, 13],
    small_size=[5, 5, 5, 5],
    dims=[256, 256, 256, 256],
    dw_dims=[256, 256, 256, 256],
    small_kernel_merged=False,
    use_multi_scale=True,
    call_structural_reparam=False,
    # misc flags used by model wrapper
    revin=1, affine=0, subtract_last=0,
    decomposition=0, kernel_size=25, individual=0,
    dropout=0.1, head_dropout=0.0, fc_dropout=0.05,
    # optimization (only used for training; still needed for Exp to exist)
    batch_size=128, train_epochs=1, patience=10, learning_rate=0.0005,
    pct_start=0.3, lradj="type1", loss="mse", use_amp=False,
    num_workers=10, itr=1, des="Exp",
    # GPU
    use_gpu=True, gpu=0, use_multi_gpu=False, devices="0,1,2,3",
)

if 'DATASETS' not in globals() or 'BASE_ARGS' not in globals():
    raise RuntimeError("Run the Memory Usage cell first (it defines DATASETS and BASE_ARGS).")

device = globals().get('device', torch.device('cuda:0'))
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available.')
print(f"Using device: {torch.cuda.get_device_name(0)}")

N_WARMUP = 10
N_RUNS   = 100
BATCH_SIZES = [128, 1]

# ── Measure each dataset ──────────────────────────────────────────────
speed_results = {}

for dataset_name, ds_args in DATASETS.items():
    speed_results[dataset_name] = {}

    for batch_size in BATCH_SIZES:
        print(f"Measuring {dataset_name} (batch={batch_size})...", end=" ", flush=True)
        try:
            args_dict = {**BASE_ARGS, **ds_args}
            args_dict['dw_dims'] = list(args_dict['dims'])
            args_dict['batch_size'] = batch_size
            args = argparse.Namespace(**args_dict)

            torch.cuda.empty_cache()
            exp   = Exp_Anomaly_Detection(args)
            model = exp.model
            model.eval()

            dummy = torch.randn(batch_size, args.seq_len, args.enc_in, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    _ = model(dummy, None, None, None)

            # Measure
            times_ms = []
            with torch.no_grad():
                for _ in range(N_RUNS):
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start.record()
                    _ = model(dummy, None, None, None)
                    end.record()
                    torch.cuda.synchronize()
                    times_ms.append(start.elapsed_time(end))

            times_ms = np.asarray(times_ms, dtype=np.float64)

            speed_results[dataset_name][batch_size] = {
                'latency_mean_ms': float(times_ms.mean()),
                'latency_std_ms':  float(times_ms.std(ddof=0)),
                'latency_min_ms':  float(times_ms.min()),
                'latency_max_ms':  float(times_ms.max()),
                'throughput':      float(batch_size / (times_ms.mean() / 1000.0)),
                's_per_iter':      float(times_ms.mean() / 1000.0),
                'seq_len':         int(args.seq_len),
                'enc_in':          int(args.enc_in),
                'batch_size':      int(batch_size),
                # per-sample latency: total latency divided by batch size
                'latency_per_sample_ms': float(times_ms.mean() / batch_size),
            }
            print(f"done ({times_ms.mean():.2f} ms)")

            del model, exp, dummy
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR: {e}")
            speed_results[dataset_name][batch_size] = None

# ── Print summary table ────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")

for batch_size in BATCH_SIZES:
    lines.append("")
    lines.append("=" * 108)
    lines.append(f"INFERENCE SPEED SUMMARY — ModernTCN Anomaly Detection (batch_size={batch_size}, N={N_RUNS} runs)")
    lines.append("=" * 108)
    lines.append(f"{'Dataset':<8} {'seq_len':<9} {'enc_in':<8} {'B':<4} "
                 f"{'Latency mean±std (ms)':<26} {'Per-sample (ms)':<18} "
                 f"{'Throughput (samp/s)':<22} {'s/iter'}")
    lines.append("-" * 108)

    for name, batch_results in speed_results.items():
        r = batch_results.get(batch_size)
        if r is None:
            lines.append(f"{name:<8} ERROR")
        else:
            lat = f"{r['latency_mean_ms']:.3f} ± {r['latency_std_ms']:.3f}"
            lines.append(f"{name:<8} {r['seq_len']:<9} {r['enc_in']:<8} {r['batch_size']:<4} "
                         f"{lat:<26} {r['latency_per_sample_ms']:>14.3f}   "
                         f"{r['throughput']:>18,.1f}     {r['s_per_iter']:.4f}")

output_text = "\n".join(lines)
print(output_text)

with open("inference_speed.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to inference_speed.txt")