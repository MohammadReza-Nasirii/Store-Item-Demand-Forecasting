import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# --- Load Data ---
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

# --- Train / Test Split ---
split_date = '2017-10-31'
X_train = X[data['date'] <= split_date]
y_train = y[data['date'] <= split_date]
X_test = X[data['date'] > split_date]
y_test = y[data['date'] > split_date]

# --- Model Initialization ---
model = XGBRegressor(random_state=42)

# --- Fit Model ---
model.fit(X_train, y_train)

# --- Predictions ---
preds = model.predict(X_test)

# --- Evaluation ---
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Initial Model → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# --- Feature Importance ---
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', legend=False)
plt.title('Feature Importance - XGBoost Model')
plt.tight_layout()
plt.show()

# --- Hyperparameter Tuning ---
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Parameters:", random_search.best_params_)
print("Best Score (MAE):", -random_search.best_score_)

best_params = random_search.best_params_

model_final = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    **best_params
)

model_final.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbose=True
)

print(model_final.best_iteration)

