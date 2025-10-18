import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('../DataSet/features_v1.csv')

test = df[(df['store']==1) & (df['item']==1)]

test.set_index('date', inplace=True)

test = test.sort_index()

print(test.head())