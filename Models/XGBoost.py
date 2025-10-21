import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


test = pd.read_csv('../DataSet/features_v1.csv')

test = test[(test['store'] == 1) & (test['item'] == 1)]

feature_cols = ['store', 'item', 'day_of_week', 'month', 'is_weekend',
                'lag_1', 'lag_7', 'lag_30', 'rolling_7', 'rolling_30',
                'dow_sin', 'dow_cos', 'month_sin', 'month_cos']

X = test[feature_cols]

y = test['sales']

split_date = '2017-10-31'

X_train = X.loc[test['date'] <= split_date]

y_train = y.loc[test['date'] <= split_date]

X_test = X.loc[test['date'] > split_date]

y_test = y.loc[test['date'] > split_date]

model = XGBRegressor(random_state=42)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")



