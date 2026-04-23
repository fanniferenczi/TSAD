import pandas as pd

root_path = './data/SWaT/'

# --- Normal data ---
print('Processing normal data...')
df_normal = pd.read_excel(f'{root_path}SWaT_Dataset_Normal_v1.xlsx', header=1)
print(f'Raw normal rows: {len(df_normal)}')

df_normal[' Timestamp'] = pd.to_datetime(
    df_normal[' Timestamp'], format='mixed', dayfirst=True
)
df_normal = df_normal.set_index(' Timestamp')

# separate string column before resampling
label_col_normal = df_normal[['Normal/Attack']]
numeric_normal = df_normal.drop(columns=['Normal/Attack'])

numeric_resampled_normal = numeric_normal.resample('10s').median()
label_resampled_normal = label_col_normal.resample('10s').agg(
    lambda x: x.mode()[0] if len(x) > 0 else 'Normal'
)

df_normal_downsampled = numeric_resampled_normal.copy()
df_normal_downsampled['Normal/Attack'] = label_resampled_normal['Normal/Attack']
df_normal_downsampled.index.name = ' Timestamp'
df_normal_downsampled = df_normal_downsampled.reset_index()

# strip leading/trailing spaces from column names
df_normal_downsampled.columns = df_normal_downsampled.columns.str.strip()

df_normal_downsampled.to_csv(f'{root_path}SWaT_normaldata_downsampled.csv', index=False)
print(f'Downsampled normal rows: {len(df_normal_downsampled)}')  # expect ~49,619
print(f'Normal NaN count: {df_normal_downsampled.isna().sum().sum()}')

# --- Attack data ---
print('Processing attack data...')
df_attack = pd.read_excel(f'{root_path}SWaT_Dataset_Attack_v0.xlsx', header=1)
print(f'Raw attack rows: {len(df_attack)}')

df_attack[' Timestamp'] = pd.to_datetime(
    df_attack[' Timestamp'], format='mixed', dayfirst=True
)
df_attack = df_attack.set_index(' Timestamp')

label_col_attack = df_attack[['Normal/Attack']]
numeric_attack = df_attack.drop(columns=['Normal/Attack'])

numeric_resampled_attack = numeric_attack.resample('10s').median()
label_resampled_attack = label_col_attack.resample('10s').agg(
    lambda x: x.mode()[0] if len(x) > 0 else 'Normal'
)

df_attack_downsampled = numeric_resampled_attack.copy()
df_attack_downsampled['Normal/Attack'] = label_resampled_attack['Normal/Attack']
df_attack_downsampled.index.name = ' Timestamp'
df_attack_downsampled = df_attack_downsampled.reset_index()

# strip leading/trailing spaces from column names
df_attack_downsampled.columns = df_attack_downsampled.columns.str.strip()

# fill NaN values introduced by resampling
df_attack_downsampled = df_attack_downsampled.fillna(method='ffill').fillna(method='bfill')

df_attack_downsampled.to_csv(f'{root_path}SWaT_attackdata_downsampled.csv', index=False)
print(f'Downsampled attack rows: {len(df_attack_downsampled)}')  # expect ~44,931
print(f'Attack NaN count after fix: {df_attack_downsampled.isna().sum().sum()}')

# verify columns match between normal and attack
normal_cols = df_normal_downsampled.columns.tolist()
attack_cols = df_attack_downsampled.columns.tolist()
print(f'Columns match: {normal_cols == attack_cols}')
if normal_cols != attack_cols:
    print('Mismatched columns:')
    for n, a in zip(normal_cols, attack_cols):
        if n != a:
            print(f'  normal: "{n}" vs attack: "{a}"')

print('Done.')