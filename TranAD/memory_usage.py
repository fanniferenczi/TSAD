# -- Memory usage for TranAD anomaly detection (batch_size=128) -------------------
import sys
import torch
import torch.nn as nn

# Assumes you're in TranAD/ (the notebook already does %cd there)
sys.path.insert(0, './')

# src.models imports src.constants -> src.parser.parse_args().
# In notebooks, sanitize argv to avoid argparse errors from Jupyter flags.
_argv_backup = sys.argv[:]
sys.argv = [sys.argv[0]]
from src.models import TranAD
sys.argv = _argv_backup

device = torch.device('cpu')
print(f'Using device: {device}')

# Feature dimensions per dataset (consistent with this workspace setup).
DATASETS = {
    'SMD':  dict(feats=38),
    'MSL':  dict(feats=55),
    'SMAP': dict(feats=25),
    'SWaT': dict(feats=51),
    'PSM':  dict(feats=25),
}

BATCH_SIZE = 128
results = {}

for dataset_name, ds_args in DATASETS.items():
    print(f'Measuring {dataset_name}...')
    try:
        feats = ds_args['feats']

        # Build model on CPU
        model = TranAD(feats).double().to(device)

        # Weights including buffers as actually allocated tensors on CPU
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        weight_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
        weight_mem = weight_bytes / 1e6

        # TranAD input shape: [window, batch, feats]
        dummy = torch.randn(
            model.n_window, BATCH_SIZE, feats,
            device=device, dtype=torch.float64
        )
        elem = dummy[-1:, :, :]
        criterion = nn.MSELoss()

        # Train memory estimate (weights + input + output + gradients)
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(dummy, elem)
        output = out[1] if isinstance(out, tuple) else out
        loss = criterion(output, dummy)
        loss.backward()

        input_mb = (dummy.numel() * dummy.element_size()) / 1e6
        output_mb = (output.numel() * output.element_size()) / 1e6
        grad_mb = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters() if p.grad is not None
        ) / 1e6
        peak_train = weight_mem + input_mb + output_mb + grad_mb

        # Inference memory estimate (weights + input + output)
        model.eval()
        with torch.no_grad():
            out_eval = model(dummy, elem)
            out_infer = out_eval[1] if isinstance(out_eval, tuple) else out_eval
        infer_output_mb = (out_infer.numel() * out_infer.element_size()) / 1e6
        peak_infer = weight_mem + input_mb + infer_output_mb

        total_params = sum(p.numel() for p in model.parameters())

        results[dataset_name] = {
            'params': total_params,
            'weight_exact': total_params * 8 / 1e6,  # fp64 params x 8 bytes -> MB
            'weight_mb': weight_mem,                 # CPU allocated weights (+buffers)
            'train_mb': peak_train,
            'infer_mb': peak_infer,
            'overhead_mb': peak_train - weight_mem,
        }

        # Clean up before next dataset
        del model, dummy, elem, out, output, out_eval, out_infer, loss

    except Exception as e:
        print(f'  ERROR: {e}')
        results[dataset_name] = None

import sys, io
from datetime import datetime, timezone, timedelta

lines = []
lines.append(f"Run: {datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('=' * 95)
lines.append('MEMORY SUMMARY - TranAD Anomaly Detection (batch_size=128)')
lines.append('=' * 95)
lines.append(f"{'Dataset':<8} {'Params':<12} {'Weights exact':<15} {'Weights CPU':<13} "
             f"{'Train peak':<13} {'Infer peak':<13} {'Overhead':<10}")
lines.append(f"{'':>20} {'(paramsx8B)':>15} {'(allocated)':>13} "
             f"{'(MB)':>13} {'(MB)':>13} {'(MB)':>10}")
lines.append('-' * 95)
for name, r in results.items():
    if r is None:
        lines.append(f'{name:<8} ERROR')
    else:
        lines.append(
            f"{name:<8} {r['params']:>10,}   {r['weight_exact']:>11.2f} MB  "
            f"{r['weight_mb']:>9.2f} MB  "
            f"{r['train_mb']:>9.2f} MB  "
            f"{r['infer_mb']:>9.2f} MB  "
            f"{r['overhead_mb']:>7.2f} MB"
        )

lines.append("\n── Verification 1: independent weight size check ──────────────────")
for name, hp in DATASETS.items():
    model = TranAD(hp['feats']).double().cpu()

    method_a = sum(p.numel() * p.element_size()
                   for p in model.parameters()) / 1e6

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    method_b = buffer.tell() / 1e6

    seen = set()
    method_c = 0
    for p in model.parameters():
        storage_id = p.storage().data_ptr()
        if storage_id not in seen:
            seen.add(storage_id)
            method_c += p.storage().size() * p.element_size()
    method_c /= 1e6

    lines.append(f"  {name:<8} A(numel×bytes)={method_a:.3f}MB  "
                 f"B(serialized)={method_b:.3f}MB  "
                 f"C(storage)={method_c:.3f}MB")
    del model

lines.append("\n── Verification 2: overhead decomposition ─────────────────────────")
lines.append(f"  {'Dataset':<8} {'input':>8} {'output':>8} {'grads':>8} "
             f"{'sum':>8} {'overhead':>10} {'match?'}")
lines.append(f"  {'-'*65}")
for name, hp in DATASETS.items():
    feats = hp['feats']
    model = TranAD(feats).double().cpu().train()
    dummy = torch.randn(model.n_window, BATCH_SIZE, feats, dtype=torch.float64)
    elem  = dummy[-1:, :, :]

    out = model(dummy, elem)
    output = out[1] if isinstance(out, tuple) else out
    loss = nn.MSELoss()(output, dummy)
    model.zero_grad()
    loss.backward()

    input_mb  = dummy.numel()  * dummy.element_size()  / 1e6
    output_mb = output.numel() * output.element_size() / 1e6
    grad_mb   = sum(p.grad.numel() * p.grad.element_size()
                    for p in model.parameters() if p.grad is not None) / 1e6

    component_sum = input_mb + output_mb + grad_mb
    overhead = results[name]['overhead_mb']
    match = 'OK' if abs(component_sum - overhead) < 0.01 else 'MISMATCH'

    lines.append(f"  {name:<8} {input_mb:>8.3f} {output_mb:>8.3f} {grad_mb:>8.3f} "
                 f"{component_sum:>8.3f} {overhead:>10.3f} {match}")
    del model, dummy, elem, out, output, loss

output_text = "\n".join(lines)
print(output_text)

with open("memory_usage.txt", "w") as f:
    f.write(output_text + "\n")
print("\nResults written to memory_usage.txt")