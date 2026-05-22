import torch
import torch.nn as nn
from datetime import datetime, timezone, timedelta
from model.AnomalyTransformer import AnomalyTransformer
device = torch.device("cuda:0")

# Force CUDA context once before measurements
_ = torch.zeros(1).to(device)
torch.cuda.synchronize()

DATASETS = {
    "SMD":   dict(enc_in=38),
    "MSL":   dict(enc_in=55),
    "SMAP":  dict(enc_in=25),
    "SWaT":  dict(enc_in=51),
    "PSM":   dict(enc_in=25),
}

results = {}

for name, cfg in DATASETS.items():
    enc_in = cfg["enc_in"]

    # ── Params ───────────────────────────────────────────────────────────
    model_cpu = AnomalyTransformer(win_size=100, enc_in=enc_in, c_out=enc_in, e_layers=3)
    total_params = sum(p.numel() for p in model_cpu.parameters())
    params_mb = total_params * 4 / 1024**2
    del model_cpu

    # ── GPU allocated (weights + buffers) ────────────────────────────────
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)
    model = AnomalyTransformer(win_size=100, enc_in=enc_in, c_out=enc_in, e_layers=3).to(device)
    gpu_allocated = (torch.cuda.memory_allocated(device) - baseline) / 1024**2

    # ── Peak training (full minimax step) ────────────────────────────────
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(128, 100, enc_in, device=device)
    criterion = nn.MSELoss()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.zero_grad(set_to_none=True)
    output, series, prior, sigma = model(x)
    loss = criterion(output, x)
    loss.backward()

    model.zero_grad(set_to_none=True)
    output2, series2, prior2, sigma2 = model(x)
    loss2 = criterion(output2, x)
    loss2.backward()

    optimizer.step()
    torch.cuda.synchronize()
    peak_train = torch.cuda.max_memory_allocated(device) / 1024**2

    # ── Peak inference ───────────────────────────────────────────────────
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _, _, _, _ = model(x)

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

    del model, optimizer, x, output, output2, loss, loss2
    torch.cuda.empty_cache()

# ── Print table ──────────────────────────────────────────────────────────
lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('=' * 95)
lines.append("MEMORY SUMMARY — AnomalyTransformer (batch_size=128)")
lines.append('=' * 95)
lines.append(f"{'Dataset':<8} {'Params':>12} {'Params MB':>10} {'GPU alloc':>10} {'Peak train':>12} {'Overhead':>10} {'Peak infer':>12}")
lines.append('-' * 95)
for name, r in results.items():
    lines.append(f"{name:<8} {r['params']:>12,} {r['params_mb']:>9.2f} MB {r['gpu_allocated']:>9.2f} MB "
                 f"{r['peak_train']:>11.2f} MB {r['overhead']:>9.2f} MB {r['peak_infer']:>11.2f} MB")

output_text = "\n".join(lines)
print(output_text)

with open("memory_usage.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to memory_usage.txt")
