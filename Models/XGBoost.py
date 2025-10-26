import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('TkAgg')


# --- Load and Prepare Data ---
data = pd.read_csv('../DataSet/features_v1.csv')

# Focus on one store-item pair
data = data[(data['store'] == 1) & (data['item'] == 1)]

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# --- Feature Selection ---
feature_cols = [
    'store', 'item', 'day_of_week', 'month', 'is_weekend',
    'lag_1', 'lag_7', 'lag_30', 'rolling_std_7', 'rolling_mean_7',
    'rolling_mean_30', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'
]

X = data[feature_cols].astype(float)
y = data['sales']

# --- Train/Test Split ---
split_date = '2017-10-31'
X_train = X[data['date'] <= split_date]
y_train = y[data['date'] <= split_date]
X_test = X[data['date'] > split_date]
y_test = y[data['date'] > split_date]

# --- Convert to DMatrix (XGBoost's optimized data format) ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# --- Parameters ---
params = {
    'objective': 'reg:squarederror',  # Regression objective
    'eval_metric': 'rmse',            # Root Mean Squared Error metric
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# --- Train model with early stopping ---
model_final = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,                  # Max number of boosting rounds
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=50,             # Stop if no improvement for 50 rounds
    verbose_eval=50                       # Print every 50 iterations
)

# --- Predictions ---
preds = model_final.predict(dvalid)

# --- Evaluation ---
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n✅ Final Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --- Feature Importance ---
xgb.plot_importance(model_final, max_num_features=10, importance_type='gain', title='Top 10 Important Features')
plt.tight_layout()
plt.show()
