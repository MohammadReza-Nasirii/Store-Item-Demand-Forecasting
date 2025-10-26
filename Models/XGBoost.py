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

# --- Convert to DMatrix (optimized for XGBoost) ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# --- Model Parameters ---
params = {
    'objective': 'reg:squarederror',  # For regression task
    'eval_metric': 'rmse',            # Use RMSE as evaluation metric
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
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=50,
    verbose_eval=50
)

# --- Predictions & Evaluation ---
preds = model_final.predict(dvalid)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\n✅ Final Model Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --- Start Forecasting for next 30 days ---
forecast_df = data.copy()  # Copy to append new predictions

future_dates = pd.date_range(start=data['date'].max() + pd.Timedelta(days=1),
                             periods=30, freq='D')

# Empty list to store predictions
future_preds = []

for next_date in future_dates:
    # --- Build time-based features ---
    day_of_week = next_date.dayofweek
    month = next_date.month
    is_weekend = int(day_of_week >= 5)

    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # --- Generate lag & rolling features dynamically ---
    lag_1 = forecast_df['sales'].iloc[-1]
    lag_7 = forecast_df['sales'].iloc[-7] if len(forecast_df) >= 7 else lag_1
    lag_30 = forecast_df['sales'].iloc[-30] if len(forecast_df) >= 30 else lag_7

    rolling_mean_7 = forecast_df['sales'].tail(7).mean()
    rolling_mean_30 = forecast_df['sales'].tail(30).mean()
    rolling_std_7 = forecast_df['sales'].tail(7).std()

    # --- Create the feature vector for prediction ---
    next_row = pd.DataFrame({
        'store': [1],
        'item': [1],
        'day_of_week': [day_of_week],
        'month': [month],
        'is_weekend': [is_weekend],
        'lag_1': [lag_1],
        'lag_7': [lag_7],
        'lag_30': [lag_30],
        'rolling_std_7': [rolling_std_7],
        'rolling_mean_7': [rolling_mean_7],
        'rolling_mean_30': [rolling_mean_30],
        'dow_sin': [dow_sin],
        'dow_cos': [dow_cos],
        'month_sin': [month_sin],
        'month_cos': [month_cos]
    })

    # --- Predict next day's sales ---
    pred = model_final.predict(xgb.DMatrix(next_row))[0]
    future_preds.append(pred)

    # --- Add prediction to forecast_df to update lag features ---
    forecast_df = pd.concat(
        [forecast_df, pd.DataFrame({'date': [next_date], 'sales': [pred]})],
        ignore_index=True
    )

# --- Combine future predictions with future dates ---
future_df = pd.DataFrame({'date': future_dates, 'predicted_sales': future_preds})

# --- Plot the result ---
plt.figure(figsize=(10, 5))
plt.plot(data['date'], data['sales'], label='Actual Sales', color='blue')
plt.plot(future_df['date'], future_df['predicted_sales'], label='Forecasted Sales', color='orange')
plt.title('30-Day Future Sales Forecast (XGBoost)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()
