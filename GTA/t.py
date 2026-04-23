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


df = pd.read_csv('./data/GECCO/gecco2018_water_quality.csv')
print(df.shape)
print(df['EVENT'].value_counts())
print(df.head())