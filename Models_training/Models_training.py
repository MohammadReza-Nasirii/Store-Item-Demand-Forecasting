import pandas as pd
import numpy as np

df = pd.read_csv('../DataSet/enhanced_train.csv')

df = df.sort_values(['store','item','date'])

df['lag_1'] = df.groupby(['store','item'])['sales'].shift(1)

df['lag_7'] = df.groupby(['store','item'])['sales'].shift(7)

df['lag_30'] = df.groupby(['store','item'])['sales'].shift(30)

df['rolling_1'] = df.groupby(['store','item'])['sales'].shift(1).rolling(window=7, min_periods=1).mean()

df['rolling_7'] = df.groupby(['store','item'])['sales'].shift(7)

df['rolling_30'] = df.groupby(['store','item'])['sales'].shift(30)
