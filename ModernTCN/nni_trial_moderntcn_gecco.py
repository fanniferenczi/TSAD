import os
import re
import subprocess
import sys
from pathlib import Path

import nni
import numpy as np


def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default


def run_trial(params):
    repo_root = Path(__file__).resolve().parent  # ModernTCN/
    det_dir = repo_root / "ModernTCN-detection"
    run_py = det_dir / "run.py"
    gecco_dir = det_dir / "all_datasets" / "GECCO"

    if not run_py.exists():
        raise FileNotFoundError(f"Missing ModernTCN-detection/run.py at {run_py}")
    if not gecco_dir.exists():
        raise FileNotFoundError(f"Missing GECCO folder at {gecco_dir}")

    train_path = gecco_dir / "train.npy"
    test_path = gecco_dir / "test.npy"
    label_path = gecco_dir / "test_label.npy"
    if not (train_path.exists() and test_path.exists() and label_path.exists()):
        raise FileNotFoundError(
            "GECCO expects: train.npy, test.npy, test_label.npy in ModernTCN-detection/all_datasets/GECCO/"
        )

    train = np.load(train_path)
    test = np.load(test_path)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError(f"Expected 2D arrays (T,C). Got train{train.shape}, test{test.shape}")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Train/test feature dims don't match")

    enc_in = int(train.shape[1])

    seq_len = _safe_int(params.get("seq_len"), 100)

    patch_size = _safe_int(params.get("patch_size"), 8)
    patch_stride = _safe_int(params.get("patch_stride"), 4)
    patch_size = max(1, min(patch_size, seq_len))
    patch_stride = max(1, min(patch_stride, patch_size))

    num_blocks = _safe_int(params.get("num_blocks"), 1)
    dims = _safe_int(params.get("dims"), 64)
    large_size = _safe_int(params.get("large_size"), 51)
    small_size = _safe_int(params.get("small_size"), 5)

    anomaly_ratio = float(params.get("anomaly_ratio", 0.5))
    ffn_ratio = _safe_int(params.get("ffn_ratio"), 1)

    dropout = float(params.get("dropout", 0.1))
    learning_rate = float(params.get("learning_rate", 3e-4))

    use_multi_scale = params.get("use_multi_scale", False)
    use_multi_scale_str = "True" if bool(use_multi_scale) else "False"

    batch_size = _safe_int(params.get("batch_size"), 128)
    train_epochs = _safe_int(params.get("train_epochs"), 2)
    patience = _safe_int(params.get("patience"), 5)

    seed = 2021

    cmd = [
        sys.executable,
        "-u",
        str(run_py),
        "--task_name",
        "anomaly_detection",
        "--is_training",
        "1",
        "--root_path",
        str(gecco_dir) + "/",
        "--model_id",
        f"GECCO_nni_s{seed}",
        "--model",
        "ModernTCN",
        "--data",
        "GECCO",
        "--features",
        "M",
        "--seq_len",
        str(seq_len),
        "--label_len",
        "0",
        "--pred_len",
        "0",
        "--enc_in",
        str(enc_in),
        "--c_out",
        str(enc_in),
        "--num_blocks",
        str(num_blocks),
        "--dims",
        str(dims),
        "--large_size",
        str(large_size),
        "--small_size",
        str(small_size),
        "--ffn_ratio",
        str(ffn_ratio),
        "--patch_size",
        str(patch_size),
        "--patch_stride",
        str(patch_stride),
        "--dropout",
        str(dropout),
        "--head_dropout",
        "0.0",
        "--anomaly_ratio",
        str(anomaly_ratio),
        "--learning_rate",
        str(learning_rate),
        "--batch_size",
        str(batch_size),
        "--train_epochs",
        str(train_epochs),
        "--patience",
        str(patience),
        "--itr",
        "1",
        "--des",
        "NNI",
        "--use_multi_scale",
        use_multi_scale_str,
        "--small_kernel_merged",
        "False",
        "--random_seed",
        str(seed),
    ]

    result = subprocess.run(cmd, cwd=str(det_dir), capture_output=True, text=True)
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    match = re.search(r"F-score\s*:\s*([0-9.]+)", output)
    f1 = float(match.group(1)) if match else 0.0

    if result.returncode != 0:
        # still report something so the trial is recorded; keep it minimal
        nni.report_final_result(0.0)
        return

    nni.report_final_result(f1)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    run_trial(params)
