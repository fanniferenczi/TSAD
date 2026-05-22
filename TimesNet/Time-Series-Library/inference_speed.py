import torch
import argparse
import numpy as np
import sys, os
from datetime import datetime, timezone, timedelta
sys.path.insert(0, './')

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

# ── Force CUDA initialization ─────────────────────────────────────────
_ = torch.zeros(1).cuda()
device = torch.device('cuda:0')
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

# ── Fixed args shared across all datasets ─────────────────────────────
BASE_ARGS = dict(
    task_name='anomaly_detection', is_training=0,
    model='TimesNet', data_path='ETTh1.csv',
    features='M', target='OT', freq='h',
    checkpoints='./checkpoints/',
    label_len=48, pred_len=0, dec_in=7,
    n_heads=8, d_layers=1, moving_avg=25,
    factor=1, distil=True, dropout=0.1,
    embed='timeF', activation='gelu',
    num_kernels=6, batch_size=128, train_epochs=10,
    patience=3, learning_rate=0.0001, des='test',
    loss='MSE', lradj='type1', use_amp=False,
    num_workers=10, itr=1, use_gpu=True, gpu=0,
    gpu_type='cuda', use_multi_gpu=False,
    devices='0,1,2,3', p_hidden_dims=[128, 128],
    p_hidden_layers=2, expand=2, d_conv=4,
    output_attention=False,
)

N_WARMUP = 10
N_RUNS   = 100

# ── Measure each dataset ──────────────────────────────────────────────
speed_results = {}

for dataset_name, ds_args in DATASETS.items():
    print(f"\nMeasuring {dataset_name}...")
    try:
        args = argparse.Namespace(**{**BASE_ARGS, **ds_args})

        torch.cuda.empty_cache()
        exp   = Exp_Anomaly_Detection(args)
        model = exp.model
        model.eval()

        batch_results = {}

        for batch_size in [1, 128]:
            dummy = torch.randn(batch_size, args.seq_len, args.enc_in).to(device)

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

            times_ms = np.array(times_ms)
            batch_results[batch_size] = {
                'latency_mean_ms': times_ms.mean(),
                'latency_std_ms':  times_ms.std(),
                'latency_min_ms':  times_ms.min(),
                'latency_max_ms':  times_ms.max(),
                'throughput':      batch_size / (times_ms.mean() / 1000),
                's_per_iter':      times_ms.mean() / 1000,
            }
            print(f"  batch_size={batch_size:<4} "
                  f"latency={times_ms.mean():.3f} ± {times_ms.std():.3f} ms  "
                  f"throughput={batch_size / (times_ms.mean() / 1000):>10,.1f} samp/s  "
                  f"s/iter={times_ms.mean() / 1000:.4f}")

            del dummy

        speed_results[dataset_name] = {
            'seq_len': args.seq_len,
            'enc_in':  args.enc_in,
            'bs1':     batch_results[1],
            'bs128':   batch_results[128],
        }

        del model, exp
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        speed_results[dataset_name] = None

# ── Print summary table ────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("=" * 95)
lines.append("\n" + "=" * 110)
lines.append("INFERENCE SPEED SUMMARY — TimesNet Anomaly Detection (N=100 runs)")
lines.append("=" * 110)
lines.append(f"{'Dataset':<8} {'seq_len':<9} {'enc_in':<8} "
             f"{'bs=1 latency (ms)':<22} {'bs=1 s/iter':<14} "
             f"{'bs=128 latency (ms)':<22} {'bs=128 s/iter':<14} {'bs=128 throughput'}")
lines.append("-" * 110)
for name, r in speed_results.items():
    if r is None:
        lines.append(f"{name:<8} ERROR")
    else:
        lat1   = f"{r['bs1']['latency_mean_ms']:.3f} ± {r['bs1']['latency_std_ms']:.3f}"
        lat128 = f"{r['bs128']['latency_mean_ms']:.3f} ± {r['bs128']['latency_std_ms']:.3f}"
        lines.append(f"{name:<8} {r['seq_len']:<9} {r['enc_in']:<8} "
                     f"{lat1:<22} {r['bs1']['s_per_iter']:<14.4f} "
                     f"{lat128:<22} {r['bs128']['s_per_iter']:<14.4f} "
                     f"{r['bs128']['throughput']:>14,.1f}")

output_text = "\n".join(lines)
print(output_text)

with open("inference_speed.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to inference_speed.txt")
        