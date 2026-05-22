# -- Inference speed for TranAD on ALL datasets (CPU timer) ------------------
import sys
import time
import numpy as np
import torch
from datetime import datetime, timezone, timedelta

# Assumes you're in TranAD/ (the notebook already does %cd there)
sys.path.insert(0, './')

# Reuse DATASETS/BATCH_SIZE from the Memory Usage cell above when available
if 'DATASETS' not in globals():
    DATASETS = {
        'SMD':  dict(feats=38),
        'MSL':  dict(feats=55),
        'SMAP': dict(feats=25),
        'SWaT': dict(feats=51),
        'PSM':  dict(feats=25),
    }
if 'BATCH_SIZE' not in globals():
    BATCH_SIZE = 128

# src.models imports src.constants -> src.parser.parse_args().
# In notebooks, sanitize argv to avoid argparse errors from Jupyter flags.
_argv_backup = sys.argv[:]
sys.argv = [sys.argv[0]]
from src.models import TranAD
sys.argv = _argv_backup

device = torch.device('cpu')
print(f'Using device: {device}')

N_WARMUP = 10
N_RUNS = 100

# -- Measure each dataset ------------------------------------------------------
speed_results = {}

for dataset_name, ds_args in DATASETS.items():
    print(f"Measuring {dataset_name}...", end=' ', flush=True)
    try:
        feats = int(ds_args['feats'])

        model = TranAD(feats).double().to(device).eval()
        dummy = torch.randn(model.n_window, BATCH_SIZE, feats, dtype=torch.float64, device=device)
        elem = dummy[-1:, :, :]

        # Warmup
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model(dummy, elem)

        # CPU timing
        times_ms = []
        with torch.no_grad():
            for _ in range(N_RUNS):
                t0 = time.perf_counter()
                _ = model(dummy, elem)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        times_ms = np.asarray(times_ms, dtype=np.float64)

        speed_results[dataset_name] = {
            'latency_mean_ms': float(times_ms.mean()),
            'latency_std_ms': float(times_ms.std(ddof=0)),
            'latency_min_ms': float(times_ms.min()),
            'latency_max_ms': float(times_ms.max()),
            'throughput': float(BATCH_SIZE / (times_ms.mean() / 1000.0)),
            's_per_iter': float(times_ms.mean() / 1000.0),
            'window': int(model.n_window),
            'feats': feats,
            'batch_size': int(BATCH_SIZE),
        }
        print(f"done ({times_ms.mean():.2f} ms)")

        del model, dummy, elem

    except Exception as e:
        print(f"ERROR: {e}")
        speed_results[dataset_name] = None

# -- Print summary table -------------------------------------------------------
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('=' * 98)
lines.append(f"INFERENCE SPEED SUMMARY - TranAD (batch_size={BATCH_SIZE}, N={N_RUNS} runs)")
lines.append('=' * 98)
lines.append(f"{'Dataset':<8} {'window':<9} {'feats':<8} {'B':<4} {'Latency mean+-std (ms)':<26} {'Throughput (samp/s)':<22} {'s/iter'}")
lines.append('-' * 98)
for name, r in speed_results.items():
    if r is None:
        lines.append(f"{name:<8} ERROR")
    else:
        lat = f"{r['latency_mean_ms']:.3f} +- {r['latency_std_ms']:.3f}"
        lines.append(f"{name:<8} {r['window']:<9} {r['feats']:<8} {r['batch_size']:<4} {lat:<26} {r['throughput']:>18,.1f}     {r['s_per_iter']:.4f}")

output_text = "\n".join(lines)
print(output_text)

with open("inference_speed.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to inference_speed.txt")
