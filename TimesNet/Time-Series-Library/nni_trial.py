import nni
import subprocess
import re
import os

def run_trial(params):
    # In nni_trial.py — change train_epochs to 3
    cmd = [
        "python", "-u", "run.py",
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--root_path", "./dataset/GECCO",
        "--model_id", "GECCO",
        "--model", "TimesNet",
        "--data", "GECCO",
        "--features", "M",
        "--pred_len", "0",
        "--d_model", "32",
        "--d_ff", "32",
        "--enc_in", "9",
        "--c_out", "9",
        "--batch_size", "128",
        "--train_epochs", "3",       # ← reduced from 10 to 3
        "--anomaly_ratio", "0.01",   # ← fix this, find best after
        "--seq_len",  str(params["seq_len"]),
        "--top_k",    str(params["top_k"]),
        "--e_layers", str(params["e_layers"]),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Parse F1 score from output
    match = re.search(r"F-score\s*:\s*([0-9.]+)", output)
    f1 = float(match.group(1)) if match else 0.0
    
    # Report F1 to NNI (it maximizes this)
    nni.report_final_result(f1)

if __name__ == "__main__":
    params = nni.get_next_parameter()
    run_trial(params)