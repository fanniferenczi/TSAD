"""
Download and prepare the GECCO 2018 Water Quality dataset for TimesNet anomaly detection.

Source: https://zenodo.org/records/3884398
The dataset contains water quality sensor readings with anomaly labels (EVENT column).

Columns:
  - Time: datetime
  - Tp: Temperature
  - Cl: Chlorine
  - pH: pH level
  - Redox: Redox potential
  - Leit: Conductivity
  - Trueb: Turbidity (NTU)
  - Trueb_FNU: Turbidity (FNU)  [may not exist in all versions]
  - Fm: Flow meter
  - EVENT: Anomaly label (True/False)

Input:  TSAD/datasets/GECCO/gecco2018_water_quality.csv (raw data repo)
Output: TSAD/TimesNet/Time-Series-Library/dataset/GECCO/ with train.npy, test.npy, test_label.npy
"""

import os
import numpy as np
import pandas as pd

# Paths relative to the repo structure: TSAD/TimesNet/Time-Series-Library/scripts/prepare_gecco.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSL_DIR = os.path.dirname(SCRIPT_DIR)                          # Time-Series-Library/
TSAD_ROOT = os.path.dirname(os.path.dirname(TSL_DIR))           # TSAD/
RAW_DATA_DIR = os.path.join(TSAD_ROOT, "datasets", "GECCO")     # TSAD/datasets/GECCO/
OUTPUT_DIR = os.path.join(TSL_DIR, "dataset", "GECCO")          # Time-Series-Library/dataset/GECCO/


def prepare_gecco(csv_path: str, output_dir: str = OUTPUT_DIR, train_ratio: float = 0.6):
    """
    Convert the GECCO CSV into .npy files for TimesNet anomaly detection.

    Creates:
      - train.npy: feature array (N_train, num_features), no anomalies
      - test.npy: feature array (N_test, num_features)
      - test_label.npy: label array (N_test,) with 0/1
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, sep=",")

    # Identify the anomaly label column (EVENT or event)
    event_col = None
    for col in df.columns:
        if col.strip().upper() == "EVENT":
            event_col = col.strip()
            break
    if event_col is None:
        raise ValueError(f"Could not find EVENT column. Columns: {list(df.columns)}")

    # Identify feature columns (exclude time, label, and index columns)
    exclude = {"Time", "time", "datetime", "Datetime", event_col}
    feature_cols = [c for c in df.columns if c.strip() not in exclude and not c.startswith("Unnamed")]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Label column: {event_col}")

    # Convert features to float, handle NaN
    data = df[feature_cols].apply(pd.to_numeric, errors="coerce").values
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)

    # Convert labels: True/False or 1/0
    labels = df[event_col].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    labels = labels.fillna(0).values.astype(np.int64)

    print(f"Total samples: {len(data)}, Features: {data.shape[1]}")
    print(f"Anomaly ratio: {labels.sum() / len(labels) * 100:.2f}%")

    # Split: first train_ratio for training (ideally anomaly-free), rest for test
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_labels = labels[split_idx:]

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Test anomaly ratio: {test_labels.sum() / len(test_labels) * 100:.2f}%")

    # Save
    np.save(os.path.join(output_dir, "train.npy"), train_data)
    np.save(os.path.join(output_dir, "test.npy"), test_data)
    np.save(os.path.join(output_dir, "test_label.npy"), test_labels)
    print(f"Saved train.npy, test.npy, test_label.npy to {output_dir}")

    return data.shape[1]  # return number of features


if __name__ == "__main__":
    csv_path = os.path.join(RAW_DATA_DIR, "gecco2018_water_quality.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. "
            f"Place the raw CSV in TSAD/datasets/GECCO/"
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_features = prepare_gecco(csv_path)
    print(f"\nDone! Use --enc_in {num_features} --c_out {num_features} when running TimesNet.")
