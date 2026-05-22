import sys, torch, torch.nn as nn
from datetime import datetime, timezone, timedelta
sys.path.insert(0, './GTA')

from models.gta import GTA

device = torch.device("cuda:3")  # adjust to your available GPU

# Force CUDA context once before measurements
_ = torch.zeros(1).to(device)
torch.cuda.synchronize()

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
BATCH_SIZE = 128  # consistent with cross-model comparability

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

    # ── Params (CPU, float64 to match .double() runtime) ─────────────────
    model_cpu = GTA(
        num_nodes, SEQ_LEN, LABEL_LEN, PRED_LEN, NUM_LEVELS,
        FACTOR, D_MODEL, N_HEADS, E_LAYERS, D_LAYERS, D_FF,
        DROPOUT, ATTN, EMBED, name, ACTIVATION, torch.device('cpu')
    ).double()
    total_params = sum(p.numel() for p in model_cpu.parameters())
    # float64 = 8 bytes per parameter
    params_mb = total_params * 8 / 1024**2
    del model_cpu

    # ── GPU allocated (weights + buffers on GPU) ──────────────────────────
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)

    model = GTA(
        num_nodes, SEQ_LEN, LABEL_LEN, PRED_LEN, NUM_LEVELS,
        FACTOR, D_MODEL, N_HEADS, E_LAYERS, D_LAYERS, D_FF,
        DROPOUT, ATTN, EMBED, name, ACTIVATION, device
    ).double().to(device)

    torch.cuda.synchronize()
    gpu_allocated = (torch.cuda.memory_allocated(device) - baseline) / 1024**2

    # ── Prepare inputs ────────────────────────────────────────────────────
    # GTA forward: (x, y, x_mark, y_mark)
    # x:      (batch, seq_len, num_nodes)
    # y:      (batch, label_len + pred_len, num_nodes)
    # x_mark: (batch, seq_len, 6)
    # y_mark: (batch, label_len + pred_len, 6)
    x      = torch.randn(BATCH_SIZE, SEQ_LEN, num_nodes,
                         device=device, dtype=torch.float64)
    y      = torch.randn(BATCH_SIZE, LABEL_LEN + PRED_LEN, num_nodes,
                         device=device, dtype=torch.float64)
    x_mark = torch.randn(BATCH_SIZE, SEQ_LEN, 6,
                         device=device, dtype=torch.float64)
    y_mark = torch.randn(BATCH_SIZE, LABEL_LEN + PRED_LEN, 6,
                         device=device, dtype=torch.float64)

    criterion = nn.MSELoss()

    # ── Peak training memory ──────────────────────────────────────────────
    # GTA training loss includes sparsity regularization term:
    # loss = MSE + sum(|logits[:,0]|)  (Chen et al., 2022, §IV-B, Eq. 4)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    model.zero_grad(set_to_none=True)
    output = model(x, y, x_mark, y_mark)
    # GTA returns output directly (not a tuple)
    if isinstance(output, tuple):
        output = output[0]
    # target is the last pred_len steps of y
    target = y[:, -PRED_LEN:, :]
    loss = criterion(output, target) + \
           torch.sum(torch.abs(model.gt_embedding.gc_module.logits[:, 0]))
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    peak_train = torch.cuda.max_memory_allocated(device) / 1024**2

    # ── Peak inference memory ─────────────────────────────────────────────
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        output_inf = model(x, y, x_mark, y_mark)

    torch.cuda.synchronize()
    peak_infer = torch.cuda.max_memory_allocated(device) / 1024**2

    results[name] = dict(
        params=total_params,
        params_mb=params_mb,
        gpu_allocated=gpu_allocated,
        peak_train=peak_train,
        overhead=peak_train - gpu_allocated,
        peak_infer=peak_infer,
    )

    del model, optimizer, x, y, x_mark, y_mark, output, loss
    torch.cuda.empty_cache()

# ── Print table ───────────────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('=' * 100)
lines.append(f"MEMORY SUMMARY — GTA (batch_size={BATCH_SIZE}, float64, seq_len={SEQ_LEN})")
lines.append('=' * 100)
lines.append(f"{'Dataset':<8} {'Params':>12} {'Params MB':>10} {'GPU alloc':>10} "
             f"{'Peak train':>12} {'Overhead':>10} {'Peak infer':>12}")
lines.append('-' * 100)
for name, r in results.items():
    lines.append(f"{name:<8} {r['params']:>12,} {r['params_mb']:>9.2f} MB "
                 f"{r['gpu_allocated']:>9.2f} MB {r['peak_train']:>11.2f} MB "
                 f"{r['overhead']:>9.2f} MB {r['peak_infer']:>11.2f} MB")

output_text = "\n".join(lines)
print(output_text)

with open("memory_usage.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to memory_usage.txt")