import ast
import csv

INPUT  = "labeled_anomalies.csv"
OUTPUT = "msl_type.csv"

rows = []
with open(INPUT, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["spacecraft"] != "MSL":
            continue
        sequences = ast.literal_eval(row["anomaly_sequences"])
        raw = row["class"].strip()[1:-1]  # strip [ ]
        classes = [c.strip() for c in raw.split(",")]
        for (start, end), anom_class in zip(sequences, classes):
            rows.append({
                "chan_id":       row["chan_id"],
                "start":         start,
                "end":           end,
                "anomaly_class": anom_class.strip(),
                "num_values":    row["num_values"],
            })

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["chan_id", "start", "end", "anomaly_class", "num_values"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")
