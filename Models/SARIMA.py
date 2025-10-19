import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


df = pd.read_csv('../DataSet/features_v1.csv')

test = df[(df['store']==1) & (df['item']==1)]

test.set_index('date', inplace=True)

test = test.sort_index()

test['sales_log'] = np.log1p(test['sales'])

test['sales_diff'] = test['sales_log'].diff()

test['sales_diff_seasonal'] = test['sales_log'].diff(7)

test = test.dropna()

result = adfuller(test['sales_diff_seasonal'], autolag='AIC')

# print('p-value : ',result[1])


