import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================================
# 1. Load and prepare data
# ================================
train_data = pd.read_csv('../DataSet/features_v1.csv')
test = pd.read_csv('../DataSet/enhanced_test.csv')

# Filter for one store-item
train_data = train_data[(train_data['store'] == 1) & (train_data['item'] == 1)]

# Ensure 'date' is datetime and sorted
train_data['date'] = pd.to_datetime(train_data['date'])
train_data = train_data.sort_values('date')
train_data.set_index('date', inplace=True)

# Feature engineering
train_data['sales_log'] = np.log1p(train_data['sales'])
train_data['sales_diff'] = train_data['sales_log'].diff()
train_data['sales_diff_seasonal'] = train_data['sales_log'].diff(7)
train_data = train_data.dropna()

# Split data into train and validation (time-based)
data = train_data.copy()
train_data = data.loc[:'2017-10-31']
val_data = data.loc['2017-11-01':]

# ================================
# 2. Train SARIMA model
# ================================
model = SARIMAX(
    endog=train_data['sales_log'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)
)

result_1 = model.fit(disp=False)

# ================================
# 3. Forecast and evaluate
# ================================
forecast = result_1.get_forecast(steps=len(val_data))
predicted_sales = np.expm1(forecast.predicted_mean)
forecast_index = forecast.predicted_mean.index

# Calculate errors
mae = mean_absolute_error(val_data['sales'], predicted_sales)
rmse = np.sqrt(mean_squared_error(val_data['sales'], predicted_sales))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ================================
# 4. Plot results
# ================================
plt.figure(figsize=(10,5))
plt.plot(val_data.index, val_data['sales'], label='Actual Sales', color='blue')
plt.plot(forecast_index, predicted_sales, label='Predicted Sales', color='orange')
plt.title('Actual vs Predicted Sales (Validation Set)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()
