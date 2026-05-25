import numpy as np
import pandas as pd

def characterise_anomalies(labels, dataset_name):
    labels = np.array(labels, dtype=int)
    n = len(labels)
    n_anomalous = labels.sum()
    anomaly_rate = n_anomalous / n * 100

    # Find contiguous anomalous segments
    segments = []
    in_segment = False
    start = None
    for i, l in enumerate(labels):
        if l == 1 and not in_segment:
            in_segment = True
            start = i
        elif l == 0 and in_segment:
            in_segment = False
            segments.append(i - start)
    if in_segment:
        segments.append(n - start)

    segment_lengths = np.array(segments) if segments else np.array([0])

    # Summary row
    summary = {
        'dataset': dataset_name,
        'total_timesteps': n,
        'anomalous_timesteps': int(n_anomalous),
        'anomaly_rate_pct': round(anomaly_rate, 2),
        'n_segments': len(segments),
        'seg_len_min': int(segment_lengths.min()),
        'seg_len_max': int(segment_lengths.max()),
        'seg_len_mean': round(float(segment_lengths.mean()), 1),
        'seg_len_median': round(float(np.median(segment_lengths)), 1),
    }

    # Segment length bin counts
    bins = [1, 2, 5, 10, 50, 100, 500, float('inf')]
    bin_labels = ['1_point', '2_4', '5_9', '10_49', '50_99', '100_499', '500plus']
    counts = np.histogram(segment_lengths, bins=bins)[0]
    for label_b, count in zip(bin_labels, counts):
        summary[f'seg_bin_{label_b}'] = int(count)

    # Per-segment detail rows
    segment_rows = [
        {'dataset': dataset_name, 'segment_id': i, 'length': int(l)}
        for i, l in enumerate(segments)
    ]

    return summary, segment_rows

def check_label_structure_npy(path, dataset_name):
    labels = np.load(path)
    print(f"\n{dataset_name}")
    print(f"  Shape:  {labels.shape}")
    print(f"  Unique values: {np.unique(labels)}")
    print(f"  Dtype: {labels.dtype}")
    
    if labels.ndim == 1:
        print(f"  -> Single label vector (one label per time step)")
    elif labels.ndim == 2:
        rows, cols = labels.shape
        print(f"  -> 2D label array ({rows} time steps, {cols} channels)")
        if np.all(labels == labels[:, [0]]):
            print(f"  -> All channels identical (effectively single label)")
        else:
            print(f"  -> Channels differ (true per-channel labels)")

def check_label_structure_csv(path, dataset_name, label_col=None):
    df = pd.read_csv(path)
    print(f"\n{dataset_name}")
    print(f"  Columns: {list(df.columns)}")
    if label_col:
        labels = df[label_col].values
    else:
        # assume last column is the label
        labels = df.iloc[:, -1].values
        print(f"  Using last column: '{df.columns[-1]}'")
    print(f"  Shape:  {labels.shape}")
    print(f"  Unique values: {np.unique(labels)}")

if __name__ == "__main__":
    all_summaries = []
    all_segments = []

    datasets = [
        ('MSL',  lambda: np.load('MSL/MSL_test_label.npy').astype(int)),
        ('SMAP', lambda: np.load('SMAP/SMAP_test_label.npy').astype(int)),
        ('SMD',  lambda: np.load('SMD/SMD_test_label.npy').astype(int)),
        ('PSM',  lambda: pd.read_csv('PSM/test_label.csv')['label'].values),
        ('SWaT', lambda: pd.read_csv('SWaT/swat2.csv')['Normal/Attack'].values),
    ]

    for name, loader in datasets:
        print(f"Processing {name}...")
        labels = loader()
        summary, segment_rows = characterise_anomalies(labels, name)
        all_summaries.append(summary)
        all_segments.extend(segment_rows)

    # Write summary table
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('anomaly_summary.csv', index=False)
    print("\nWrote anomaly_summary.csv")

    # Write per-segment detail table
    segments_df = pd.DataFrame(all_segments)
    segments_df.to_csv('anomaly_segments.csv', index=False)
    print("Wrote anomaly_segments.csv")

    print("\nSummary:")
    print(summary_df[['dataset', 'anomaly_rate_pct', 'n_segments',
                       'seg_len_min', 'seg_len_max', 'seg_len_mean',
                       'seg_len_median']].to_string(index=False))
    
    