import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

test = pd.read_csv('../DataSet/features_v1.csv')

test = test[(test['store'] == 1) & (test['item'] == 1)]

feature_cols = ['store', 'item', 'day_of_week', 'month', 'is_weekend',
                'lag_1', 'lag_7', 'lag_30', 'rolling_std_7', 'rolling_mean_30',
                'dow_sin', 'dow_cos', 'month_sin', 'month_cos','rolling_mean_7']

X = test[feature_cols]

y = test['sales']

split_date = '2017-10-31'

X_train = X.loc[test['date'] <= split_date]

y_train = y.loc[test['date'] <= split_date]

X_test = X.loc[test['date'] > split_date]

y_test = y.loc[test['date'] > split_date]

model = XGBRegressor(random_state=42)

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

rmse = np.sqrt(mean_squared_error(y_test, preds))

# print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(test.head())

plt.figure(figsize=(10,5))

sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')

plt.title('Feature Importance - XGBoost Model')

plt.tight_layout()

# plt.show()

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)

print("Best Score:", -random_search.best_score_)
