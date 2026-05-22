# ── Memory usage for ModernTCN Anomaly Detection (batch_size=128) ─────────
import os, sys, argparse
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import torch.nn as nn

# Assumes you're in ModernTCN-detection (the notebook already does %cd there)
sys.path.insert(0, "./")

# IMPORTANT: set this BEFORE the first torch.cuda call.
# This mirrors the TimesNet notebook: exposing physical GPU "3" as cuda:0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change if needed

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Run this cell on a GPU machine or switch to a CUDA-enabled PyTorch install.")

# ── Force CUDA initialization ───────────────────────────────────────────
_ = torch.zeros(1).cuda()
device = torch.device("cuda:0")
print(f"Using device: {torch.cuda.get_device_name(0)}")


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

# ── Measure memory for each dataset ─────────────────────────────────────
results = {}

for dataset_name, ds_args in DATASETS.items():
    print(f"Measuring {dataset_name}...")
    try:
        # Build args
        args_dict = {**BASE_ARGS, **ds_args}
        # Keep dw_dims consistent with dims when overridden
        args_dict["dw_dims"] = list(args_dict["dims"])
        args = argparse.Namespace(**args_dict)

        # Clear GPU memory from previous iteration
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        # Load model on GPU via the repo's experiment wrapper
        exp = Exp_Anomaly_Detection(args)
        model = exp.model
        weight_mem = (torch.cuda.memory_allocated(device) - baseline) / 1e6

        # Dummy input matching dataset dimensions
        dummy = torch.randn(args.batch_size, args.seq_len, args.enc_in, device=device)
        criterion = nn.MSELoss()

        # Peak training memory (forward + backward)
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        output = model(dummy, None, None, None)
        loss = criterion(output, dummy)
        loss.backward()
        peak_train = torch.cuda.max_memory_allocated(device) / 1e6

        # Peak inference memory (no gradients)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy, None, None, None)
        peak_infer = torch.cuda.max_memory_allocated(device) / 1e6

        total_params = sum(p.numel() for p in model.parameters())

        results[dataset_name] = {
            "params":       total_params,
            "weight_exact": total_params * 4 / 1e6,  # params × 4 bytes → MB
            "weight_mb":    weight_mem,              # allocated after model init (includes buffers)
            "train_mb":     peak_train,
            "infer_mb":     peak_infer,
            "overhead_mb":  peak_train - weight_mem,
        }

        # Clean up before next dataset
        del model, exp, dummy, output, loss
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        results[dataset_name] = None

# ── Print summary table ────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 95)
lines.append("MEMORY SUMMARY — ModernTCN Anomaly Detection (batch_size=128)")
lines.append("=" * 95)
lines.append(f"{'Dataset':<8} {'Params':<12} {'Weights exact':<15} {'Weights GPU':<13} "
             f"{'Train peak':<13} {'Infer peak':<13} {'Overhead':<10}")
lines.append(f"{'':>20} {'(params×4B)':>15} {'(allocated)':>13} "
             f"{'(MB)':>13} {'(MB)':>13} {'(MB)':>10}")
lines.append("-" * 95)
for name, r in results.items():
    if r is None:
        lines.append(f"{name:<8} ERROR")
    else:
        lines.append(f"{name:<8} {r['params']:>10,}   {r['weight_exact']:>11.2f} MB  "
                     f"{r['weight_mb']:>9.2f} MB  "
                     f"{r['train_mb']:>9.2f} MB  "
                     f"{r['infer_mb']:>9.2f} MB  "
                     f"{r['overhead_mb']:>7.2f} MB")

output_text = "\n".join(lines)
print(output_text)

with open("memory_usage.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to memory_usage.txt")