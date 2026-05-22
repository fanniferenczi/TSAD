import sys, torch, numpy as np
from datetime import datetime, timezone, timedelta
sys.path.insert(0, './GTA')

from models.gta import GTA

device = torch.device("cuda:1")  # adjust to your available GPU

# ── Force CUDA context initialization ─────────────────────────────────────
_ = torch.zeros(1).to(device)

# ── GTA hyperparameters (paper settings, Chen et al., 2022, §V-B-4) ──────
SEQ_LEN    = 60
LABEL_LEN  = 30
PRED_LEN   = 1
NUM_LEVELS = 3
FACTOR     = 5
D_MODEL    = 128
N_HEADS    = 8
E_LAYERS   = 3
D_LAYERS   = 2
D_FF       = 128
DROPOUT    = 0.05
ATTN       = 'prob'
EMBED      = 'fixed'
ACTIVATION = 'gelu'

BATCH_SIZE = 128
N_RUNS     = 100
N_WARMUP   = 10

DATASETS = {
    'SMD':  dict(num_nodes=38),
    'MSL':  dict(num_nodes=55),
    'SMAP': dict(num_nodes=25),
    'SWaT': dict(num_nodes=51),
    'PSM':  dict(num_nodes=25),
}

results = {}

for name, cfg in DATASETS.items():
    num_nodes = cfg['num_nodes']

    # ── Build model ────────────────────────────────────────────────────────
    model = GTA(
        num_nodes, SEQ_LEN, LABEL_LEN, PRED_LEN, NUM_LEVELS,
        FACTOR, D_MODEL, N_HEADS, E_LAYERS, D_LAYERS, D_FF,
        DROPOUT, ATTN, EMBED, name, ACTIVATION, device
    ).double().to(device)
    model.eval()

    # GTA forward: (x, y, x_mark, y_mark)
    x      = torch.randn(BATCH_SIZE, SEQ_LEN, num_nodes,
                         device=device, dtype=torch.float64)
    y      = torch.randn(BATCH_SIZE, LABEL_LEN + PRED_LEN, num_nodes,
                         device=device, dtype=torch.float64)
    x_mark = torch.randn(BATCH_SIZE, SEQ_LEN, 6,
                         device=device, dtype=torch.float64)
    y_mark = torch.randn(BATCH_SIZE, LABEL_LEN + PRED_LEN, 6,
                         device=device, dtype=torch.float64)

    # ── Warmup runs ────────────────────────────────────────────────────────
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x, y, x_mark, y_mark)

    # ── Timed runs using CUDA Events ───────────────────────────────────────
    latencies = []

    with torch.no_grad():
        for _ in range(N_RUNS):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)

            start.record()
            _ = model(x, y, x_mark, y_mark)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))  # milliseconds

    latencies  = np.array(latencies)
    mean_ms    = latencies.mean()
    std_ms     = latencies.std()
    throughput = BATCH_SIZE / (mean_ms / 1000)  # samples per second
    s_per_iter = mean_ms / 1000                 # seconds per iteration

    results[name] = {
        'num_nodes':  num_nodes,
        'mean_ms':    mean_ms,
        'std_ms':     std_ms,
        'throughput': throughput,
        's_per_iter': s_per_iter,
    }

    del model, x, y, x_mark, y_mark
    torch.cuda.empty_cache()

# ── Summary table ──────────────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('=' * 100)
lines.append(f'INFERENCE SPEED SUMMARY — GTA '
             f'(batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, float64, N={N_RUNS} runs)')
lines.append('=' * 100)
lines.append(f"{'Dataset':<8} {'num_nodes':<11} {'B':<5} "
             f"{'Latency mean±std (ms)':<28} {'Throughput (samp/s)':<23} {'s/iter'}")
lines.append('-' * 100)
for name, r in results.items():
    lines.append(
        f"{name:<8} {r['num_nodes']:<11} {BATCH_SIZE:<5} "
        f"{r['mean_ms']:.3f} ± {r['std_ms']:<19.3f}"
        f"{r['throughput']:>20,.1f}     {r['s_per_iter']:.4f}"
    )

output_text = "\n".join(lines)
print(output_text)

with open("inference_speed.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to inference_speed.txt")