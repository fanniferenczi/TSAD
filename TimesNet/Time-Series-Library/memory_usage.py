import torch
import torch.nn as nn
import argparse
import sys, os
from datetime import datetime, timezone, timedelta
sys.path.insert(0, "./")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # container re-indexes passed-through GPUs from 0

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

# ── Force CUDA initialization ────────────────────────────────────────
_ = torch.zeros(1).cuda()
device = torch.device("cuda:0")
print(f"Using device: {torch.cuda.get_device_name(0)}")

DATASETS = {
    # C=38 → 2^6=64 → d_model=64
    "SMD": dict(
        model_id="SMD", data="SMD",
        root_path="./dataset/SMD",
        seq_len=100, enc_in=38, c_out=38,
        d_model=64, d_ff=64, e_layers=3, top_k=3,
        anomaly_ratio=0.5,
    ),
    # C=55 → 2^6=64 → d_model=64
    "MSL": dict(
        model_id="MSL", data="MSL",
        root_path="./dataset/MSL",
        seq_len=100, enc_in=55, c_out=55,
        d_model=64, d_ff=64, e_layers=3, top_k=3,
        anomaly_ratio=1.0,
    ),
    # C=25 → 2^5=32 → d_model=32
    "SMAP": dict(
        model_id="SMAP", data="SMAP",
        root_path="./dataset/SMAP",
        seq_len=100, enc_in=25, c_out=25,
        d_model=32, d_ff=32, e_layers=3, top_k=3,
        anomaly_ratio=1.0,
    ),
    # C=51 → 2^6=64 → d_model=64
    "SWaT": dict(
        model_id="SWAT", data="SWAT",
        root_path="./dataset/SWaT",
        seq_len=100, enc_in=51, c_out=51,
        d_model=64, d_ff=64, e_layers=3, top_k=3,
        anomaly_ratio=1.0,
    ),
    # C=25 → 2^5=32 → d_model=32
    "PSM": dict(
        model_id="PSM", data="PSM",
        root_path="./dataset/PSM",
        seq_len=100, enc_in=25, c_out=25,
        d_model=32, d_ff=32, e_layers=3, top_k=3,
        anomaly_ratio=1.0,
    ),
}

# ── Fixed args shared across all datasets ────────────────────────────
BASE_ARGS = dict(
    task_name="anomaly_detection", is_training=0,
    model="TimesNet", data_path="ETTh1.csv",
    features="M", target="OT", freq="h",
    checkpoints="./checkpoints/",
    label_len=48, pred_len=0,
    dec_in=7, n_heads=8, d_layers=1,
    moving_avg=25, factor=1, distil=True,
    dropout=0.1, embed="timeF", activation="gelu",
    num_kernels=6, batch_size=128, train_epochs=10,
    patience=3, learning_rate=0.0001, des="test",
    loss="MSE", lradj="type1", use_amp=False,
    num_workers=10, itr=1, use_gpu=True, gpu=0,
    gpu_type="cuda", use_multi_gpu=False,
    devices="0,1,2,3", p_hidden_dims=[128, 128],
    p_hidden_layers=2, expand=2, d_conv=4,
    output_attention=False,
)

# ── Measure memory for each dataset ──────────────────────────────────
results = {}

for dataset_name, ds_args in DATASETS.items():
    print(f"Measuring {dataset_name}...")
    try:
        # Build args
        args = argparse.Namespace(**{**BASE_ARGS, **ds_args})

        # Clear GPU memory from previous iteration
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        # Load model
        exp = Exp_Anomaly_Detection(args)
        model = exp.model
        weight_mem = (torch.cuda.memory_allocated(device) - baseline) / 1e6

        # Dummy input matching dataset dimensions
        dummy = torch.randn(
            args.batch_size, args.seq_len, args.enc_in
        ).to(device)
        criterion = nn.MSELoss()

        # Peak training memory (forward + backward)
        torch.cuda.reset_peak_memory_stats(device)
        output = model(dummy, None, None, None)
        loss = criterion(output, dummy)
        loss.backward()
        peak_train = torch.cuda.max_memory_allocated(device) / 1e6

        # Peak inference memory (no gradients)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            output = model(dummy, None, None, None)
        peak_infer = torch.cuda.max_memory_allocated(device) / 1e6

        total_params = sum(p.numel() for p in model.parameters())

        results[dataset_name] = {
            "params":       total_params,
            "weight_exact": total_params * 4 / 1e6,  # params × 4 bytes → MB
            "weight_mb":    weight_mem,  # includes buffers, varies
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

# ── Print summary table ───────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 95)
lines.append("\n" + "=" * 95)
lines.append("MEMORY SUMMARY — TimesNet Anomaly Detection (batch_size=128)")
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