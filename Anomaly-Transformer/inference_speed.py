import torch
import numpy as np
from datetime import datetime, timezone, timedelta
from model.AnomalyTransformer import AnomalyTransformer

device = torch.device("cuda:0")

# ── Force CUDA context initialization ─────────────────────────────────────
_ = torch.zeros(1).to(device)

DATASETS = {
    'SMD':   dict(enc_in=38, seq_len=100),
    'MSL':   dict(enc_in=55, seq_len=100),
    'SMAP':  dict(enc_in=25, seq_len=100),
    'SWaT':  dict(enc_in=51, seq_len=100),
    'PSM':   dict(enc_in=25, seq_len=100),
}

BATCH_SIZE = 128
N_RUNS     = 100   # number of timed iterations (matches ModernTCN)
N_WARMUP   = 10    # warmup runs to stabilize GPU clock and cuDNN

results = {}

for dataset_name, ds_args in DATASETS.items():
    enc_in  = ds_args['enc_in']
    seq_len = ds_args['seq_len']

    # ── Build model ────────────────────────────────────────────────────────
    model = AnomalyTransformer(
        win_size=seq_len, enc_in=enc_in, c_out=enc_in, e_layers=3
    ).float().to(device)
    model.eval()

    dummy = torch.randn(BATCH_SIZE, seq_len, enc_in, device=device)

    # ── Warmup runs ────────────────────────────────────────────────────────
    # Ensures cuDNN selects its algorithm and GPU clocks are stable
    # before any timing begins — same rationale as ModernTCN
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(dummy)

    # ── Timed runs using CUDA Events ───────────────────────────────────────
    # CUDA Events measure GPU-side time directly, avoiding Python overhead
    # This matches the ModernTCN measurement methodology
    latencies = []

    with torch.no_grad():
        for _ in range(N_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(dummy)
            end.record()

            # Synchronize to ensure GPU has finished before reading time
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))  # milliseconds

    latencies  = np.array(latencies)
    mean_ms    = latencies.mean()
    std_ms     = latencies.std()
    throughput = (BATCH_SIZE / (mean_ms / 1000))  # samples per second
    s_per_iter = mean_ms / 1000                   # seconds per iteration

    results[dataset_name] = {
        'seq_len':    seq_len,
        'enc_in':     enc_in,
        'mean_ms':    mean_ms,
        'std_ms':     std_ms,
        'throughput': throughput,
        's_per_iter': s_per_iter,
    }

    del model, dummy
    torch.cuda.empty_cache()

# ── Summary table ──────────────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('=' * 98)
lines.append(f'INFERENCE SPEED SUMMARY — AnomalyTransformer '
             f'(batch_size={BATCH_SIZE}, N={N_RUNS} runs)')
lines.append('=' * 98)
lines.append(f"{'Dataset':<8} {'seq_len':<9} {'enc_in':<8} {'B':<5} "
             f"{'Latency mean±std (ms)':<28} {'Throughput (samp/s)':<23} {'s/iter'}")
lines.append('-' * 98)
for name, r in results.items():
    lines.append(
        f"{name:<8} {r['seq_len']:<9} {r['enc_in']:<8} {BATCH_SIZE:<5} "
        f"{r['mean_ms']:.3f} ± {r['std_ms']:<19.3f}"
        f"{r['throughput']:>20,.1f}     {r['s_per_iter']:.4f}"
    )

output_text = "\n".join(lines)
print(output_text)

with open("inference_speed.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to inference_speed.txt")