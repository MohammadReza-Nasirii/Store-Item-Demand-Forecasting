import pandas as pd
import XGboost as xgb
import sklearn as sk

test = pd.read_csv('../DataSet/features_v1.csv')

test = test[(test['store'] == 1) & (test['item'] == 1)]

feature_cols = ['store', 'item', 'day_of_week', 'month', 'is_weekend',
                'lag_1', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30',
                'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

X = test[feature_cols]

y = test['sales']

print(X.head())


