import pandas as pd
import numpy as np

df = pd.read_csv('../DataSet/enhanced_train.csv')

df = df.sort_values(['store','item','date'])

df['lag_1'] = df.groupby(['store','item'])['sales'].shift(1)

df['lag_7'] = df.groupby(['store','item'])['sales'].shift(7)

df['lag_30'] = df.groupby(['store','item'])['sales'].shift(30)

df['rolling_mean_7'] = (
    df.groupby(['store', 'item'])['sales']
    .shift(1)
    .rolling(window=7, min_periods=1)
    .mean()
)

df['rolling_std_7'] = (
    df.groupby(['store', 'item'])['sales']
    .shift(1)
    .rolling(window=7, min_periods=1)
    .std()
)

df['rolling_mean_30'] = (
    df.groupby(['store', 'item'])['sales']
    .shift(1)
    .rolling(window=30, min_periods=1)
    .mean()
)


df['diff_1'] = df.groupby(['store', 'item'])['sales'].diff(1)
df['pct_change_7'] = df.groupby(['store', 'item'])['sales'].pct_change(7)


df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

df = df.dropna(subset=['lag_1', 'lag_7', 'lag_30'])

df.to_csv('../DataSet/features_v1.csv', index=False)
print("Feature engineering complete. Saved as features_v1.csv")

