import pandas as pd
import os

root_path = './data/GECCO/'
input_file = 'gecco2018_water_quality.csv'  

df = pd.read_csv(os.path.join(root_path, input_file))

# drop unnamed index column
df = df.drop(columns=[df.columns[0]])

# parse timestamp
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)

# convert boolean label to int
df['EVENT'] = df['EVENT'].astype(int)

# split: first 80% train, last 20% test
split = int(len(df) * 0.8)
df_train = df.iloc[:split]
df_test = df.iloc[split:]

print(f'Total rows: {len(df)}')
print(f'Train rows: {len(df_train)}')
print(f'Test rows: {len(df_test)}')
print(f'Train anomaly rate: {df_train["EVENT"].mean():.4f}')
print(f'Test anomaly rate: {df_test["EVENT"].mean():.4f}')

os.makedirs(root_path, exist_ok=True)
df_train.to_csv(os.path.join(root_path, 'gecco_train.csv'), index=False)
df_test.to_csv(os.path.join(root_path, 'gecco_test.csv'), index=False)
print('Done.')