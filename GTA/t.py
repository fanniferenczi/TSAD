import pandas as pd

#root = './data/SWaT/'
#df_normal = pd.read_csv(f'{root}SWaT_normaldata_downsampled.csv', header=0)
#df_attack = pd.read_csv(f'{root}SWaT_attackdata_downsampled.csv', header=0)

#print('Normal columns:', df_normal.columns.tolist())
#print('Attack columns:', df_attack.columns.tolist())
#print('Normal shape:', df_normal.shape)
#print('Attack shape:', df_attack.shape)
#print('Normal dtypes:\n', df_normal.dtypes)
#print('Attack dtypes:\n', df_attack.dtypes)
#print('Normal NaN count:', df_normal.isna().sum().sum())
#print('Attack NaN count:', df_attack.isna().sum().sum())

#print('-----------------------------')
#df = pd.read_csv('./data/PSM/train.csv')
#print(df.shape) 


#df = pd.read_csv('./data/GECCO/gecco2018_water_quality.csv')
#print(df.shape)
#print(df['EVENT'].value_counts())
#print(df.head())

#import pandas as pd
#df_test = pd.read_csv('./data/GECCO/gecco_test.csv')
#print(f'Test anomaly rate: {df_test["EVENT"].mean():.4f}')
#print(f'Test anomalies: {df_test["EVENT"].sum()}')

#import pandas as pd
#df_train = pd.read_csv('./data/GECCO/gecco_train.csv')
#print(f'Train anomaly rate: {df_train["EVENT"].mean():.4f}')
#print(f'Train anomalies: {df_train["EVENT"].sum()}')

import pandas as pd
import numpy as np

df = pd.read_csv('./data/GECCO/gecco2018_water_quality.csv')

# drop unnamed index
df = df.drop(columns=[df.columns[0]])
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values('Time').reset_index(drop=True)
df['EVENT'] = df['EVENT'].astype(int)

print('=' * 50)
print('GECCO DATA EXPLORATION')
print('=' * 50)

print(f'\n[1] Basic stats:')
print(f'  Total rows: {len(df)}')
print(f'  Time range: {df["Time"].min()} to {df["Time"].max()}')
print(f'  Total anomalies: {df["EVENT"].sum()}')
print(f'  Overall anomaly rate: {df["EVENT"].mean():.4f}')

print(f'\n[2] Anomaly segment analysis:')
# find contiguous anomaly segments
in_anomaly = False
segments = []
start = None
for i, row in df.iterrows():
    if row['EVENT'] == 1 and not in_anomaly:
        in_anomaly = True
        start = i
    elif row['EVENT'] == 0 and in_anomaly:
        in_anomaly = False
        segments.append((start, i-1, i-start))
if in_anomaly:
    segments.append((start, len(df)-1, len(df)-start))

print(f'  Number of anomaly segments: {len(segments)}')
for idx, (s, e, length) in enumerate(segments):
    print(f'  Segment {idx+1}: rows {s}-{e}, length={length}, '
          f'time={df["Time"][s]} to {df["Time"][e]}')

print(f'\n[3] Temporal distribution of anomalies:')
df['month'] = df['Time'].dt.month
df['day'] = df['Time'].dt.day
monthly = df.groupby('month')['EVENT'].agg(['sum', 'count', 'mean'])
monthly.columns = ['anomalies', 'total', 'rate']
print(monthly)

print(f'\n[4] Normal data blocks:')
normal_df = df[df['EVENT'] == 0]
print(f'  Total normal rows: {len(normal_df)}')
print(f'  Normal rate: {len(normal_df)/len(df):.4f}')

print(f'\n[5] Feature stats:')
feature_cols = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
print(f'  NaN count: {df[feature_cols].isna().sum().sum()}')
print(f'  Value ranges:')
for col in feature_cols:
    print(f'    {col}: [{df[col].min():.4f}, {df[col].max():.4f}]')