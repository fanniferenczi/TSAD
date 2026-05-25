import pandas as pd
import numpy as np
import ast

df = pd.read_csv('labeled_anomalies.csv')

def parse_class(class_val):
    """Parse the class column which can be a string like 
    'contextual' or a list like '[point, contextual]'"""
    class_val = str(class_val).strip()
    
    # If it looks like a list
    if class_val.startswith('['):
        # Remove brackets and split
        class_val = class_val.replace('[', '').replace(']', '')
        types = [c.strip() for c in class_val.split(',')]
        return types
    else:
        return [class_val]

# Expand each row into one row per anomaly type
rows = []
for _, row in df.iterrows():
    types = parse_class(row['class'])
    for t in types:
        rows.append({
            'chan_id':    row['chan_id'],
            'spacecraft': row['spacecraft'],
            'type':       t.lower().strip(),
            'num_values': row['num_values']
        })

df_expanded = pd.DataFrame(rows)

# Filter to SMAP and MSL
df_nasa = df_expanded[df_expanded['spacecraft'].isin(['SMAP', 'MSL'])]

print("Anomaly type counts per dataset:")
print("(each row = one anomalous segment, not one channel)\n")

for spacecraft in ['SMAP', 'MSL']:
    subset = df_nasa[df_nasa['spacecraft'] == spacecraft]
    total  = len(subset)
    counts = subset['type'].value_counts()

    print(f"\n{'='*40}")
    print(f"{spacecraft}")
    print(f"{'='*40}")
    print(f"Total anomalous segments: {total}")
    for anomaly_type, count in counts.items():
        pct = count / total * 100
        print(f"  {anomaly_type:>12}: {count:>3}  ({pct:.1f}%)")

# Summary table
summary = df_nasa.groupby(
    ['spacecraft', 'type']).size().unstack(fill_value=0)
summary['total'] = summary.sum(axis=1)
print("\nSummary table:")
print(summary)

summary.to_csv('nasa_anomaly_types.csv')
print("\nWrote nasa_anomaly_types.csv")