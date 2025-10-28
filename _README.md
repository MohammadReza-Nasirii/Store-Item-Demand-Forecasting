
# 🏪 Store Item Demand Forecasting — Complete Project Documentation

## 📘 Overview  
This project aims to **forecast daily sales for retail store items** using advanced **time series forecasting** methods.  
It combines both **classical statistical modeling (SARIMA)** and **modern machine learning (XGBoost)** approaches to achieve a balance between interpretability and predictive power.  

The project walks through the entire data science workflow — from **data preparation** and **feature engineering**, to **model training**, **evaluation**, and **forecasting future demand**.

---

## 🎯 Project Objectives  
1. Build a **robust forecasting system** capable of predicting item-level daily sales.  
2. Compare the performance of **SARIMA** (statistical) and **XGBoost** (machine learning) models.  
3. Engineer meaningful time-based and lag features to capture seasonality and trends.  
4. Generate a **30-day forward forecast** for a selected store-item combination.  
5. Visualize and interpret model outputs for insights and business applications.

---

## 🧠 Problem Description  
Retailers must anticipate product demand to manage **inventory**, **supply chain**, and **revenue planning**.  
Accurate forecasting helps:  
- Minimize overstocking and understocking  
- Improve logistics and procurement  
- Optimize marketing and pricing strategies  

The dataset includes daily sales data for multiple stores and items over several years.

---

## 📂 Project Structure

```
store-item-demand-forecasting/
│
├── DataSet/
│   ├── train.csv
│   ├── enhanced_test.csv
│   ├── features_v1.csv
│
├── Models/
│   ├── SARIMA.py
│   ├── XGBoost.py
│
├── Reports/
│   └── Results.pdf
│
└── README.md
```

---

## 🧱 Step 1 — Data Preparation  

### 🔹 Tasks Performed  
- Loaded and sorted the dataset by `store`, `item`, and `date`.  
- Checked for missing values and data consistency.  
- Created a **datetime index** for time-series operations.  

### 🔹 Key Features Generated  
| Feature | Description |
|----------|--------------|
| `lag_1`, `lag_7`, `lag_30` | Sales from previous 1, 7, and 30 days |
| `rolling_mean_7`, `rolling_mean_30` | Average sales over the past week and month |
| `rolling_std_7` | Sales variability (standard deviation) in the past 7 days |
| `day_of_week`, `month`, `year` | Temporal identifiers |
| `is_weekend` | Binary flag for weekends |
| `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | **Cyclical encodings** for weekly and monthly patterns |

This phase provided a feature-rich dataset capturing both **seasonality** and **short-term trends**.

---

## 📈 Step 2 — Exploratory Data Analysis (EDA)

We performed visual exploration using **Matplotlib** and **Seaborn** to uncover patterns such as:
- Weekly and monthly seasonality in sales.
- Periodic fluctuations due to holidays or weekends.
- Distribution and variance of sales across items and stores.

This step helped shape our feature engineering and model selection strategies.

---

## ⚙️ Step 3 — SARIMA Model (Statistical Baseline)

### 🧩 Model Logic
SARIMA (Seasonal AutoRegressive Integrated Moving Average) captures both:
- **Trend and autocorrelation** (via AR & MA components)
- **Seasonal cycles** (via seasonal parameters)

### 🧠 Why SARIMA?
It’s interpretable and ideal for **univariate time series** with strong seasonality, giving us a solid baseline for comparison.

### 🧪 Implementation Steps
1. Transformed sales data using logarithm (`log1p`) to stabilize variance.  
2. Applied **differencing** (normal & seasonal) to achieve stationarity.  
3. Verified stationarity using **ADF (Augmented Dickey-Fuller)** test.  
4. Trained SARIMA(1,1,1)(1,1,1,7) model.  
5. Evaluated using AIC, MAE, and RMSE.  

### 📊 Result Summary
| Metric | Value |
|--------|--------|
| MAE | 4.32 |
| RMSE | 5.21 |
| AIC | -4.04 |

SARIMA successfully modeled weekly patterns but struggled with nonlinearities — motivating a transition to XGBoost.

---

## 🚀 Step 4 — XGBoost Model (Machine Learning)

### 🧩 Why XGBoost?
Unlike SARIMA, **XGBoost (Extreme Gradient Boosting)** handles:
- Nonlinear relationships  
- High-dimensional features  
- Interactions between time and external signals  

This makes it ideal for structured, feature-engineered time series.

### ⚙️ Implementation Process
1. Used the engineered features (`lags`, `rolling stats`, `cyclical time encodings`).  
2. Split dataset by time (training: before 2017-10-31, validation: after).  
3. Trained `XGBRegressor` with parameters tuned using **RandomizedSearchCV**.  
4. Introduced **early stopping** to prevent overfitting.  
5. Evaluated predictions on the validation set.

---

## 📊 Step 5 — Model Evaluation and Feature Importance

### Evaluation Metrics:
| Metric | Meaning | Value |
|--------|----------|--------|
| **MAE** | Mean Absolute Error | **3.91** |
| **RMSE** | Root Mean Squared Error | **4.80** |

📈 **Result Interpretation:**
- The XGBoost model reduced RMSE compared to SARIMA (better accuracy).  
- MAE of 3.91 indicates that on average, predictions deviate from actual sales by about 4 units.  

### 🔍 Feature Importance Insights
Top contributing features were:
1. `rolling_mean_7`
2. `rolling_mean_30`
3. `dow_sin`
4. `month_cos`
5. `day_of_week`

These highlight the dominant influence of **seasonality and recent trends** in sales forecasting.

---

## 🔮 Step 6 — Future Forecasting (30 Days Ahead)

### 🔹 Process
- The model dynamically generates future features using previous predictions:
  - Updates lags (`lag_1`, `lag_7`, `lag_30`)
  - Recomputes rolling means & cyclical encodings
- Forecasted **30 consecutive future days** after the last date in training data.

### 🔹 Output
| Metric | Value |
|--------|--------|
| Forecast Range | 2018-01-01 → 2018-01-30 |
| MAE | 3.91 |
| RMSE | 4.80 |
| Model File | `xgboost_final_model.json` |
| Predictions File | `future_predictions.csv` |

---

## 📉 Step 7 — Visualization and Diagnostics

### 1. Actual vs Forecast Plot  
Blue = real sales, Orange = model predictions.  
Shows how the forecast follows historical seasonality.

### 2. Error Distribution Histogram  
Shows the spread of prediction errors (Actual - Predicted).  
A symmetric bell-shaped curve around zero indicates **low bias**.

---

## 💾 Outputs
After full execution, the following artifacts are generated:
```
📂 /Models
├── xgboost_final_model.json    → Trained model
├── sarima_model.pkl            → Statistical model
├── future_predictions.csv      → 30-day forecast
└── feature_importance.png      → Visual insight
```

---

## 🧩 Technologies Used
| Category | Tools |
|-----------|--------|
| **Language** | Python 3.11 |
| **Libraries** | pandas, numpy, matplotlib, seaborn |
| **Modeling** | statsmodels, xgboost |
| **Evaluation** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## 💡 Key Takeaways
- SARIMA captures interpretable linear seasonality; XGBoost models complex nonlinear trends.
- Rolling and lag features are critical for temporal dependency.
- Cyclic encoding of time improved performance significantly.
- A hybrid or ensemble approach could combine the strengths of both.

---

## 🚀 Future Work
- Expand forecasts across all `store-item` pairs (multi-series modeling).
- Deploy model as an API with **FastAPI** or **Flask**.
- Integrate with **MLOps tools** for automated retraining and monitoring.
- Explore **LightGBM**, **CatBoost**, or **LSTM/Transformer** models for deeper sequence understanding.

---

## 🧾 Author Notes
Developed as part of an end-to-end time series forecasting project to demonstrate a hybrid statistical + machine learning approach.  
All code, explanations, and documentation are written for **clarity, reproducibility, and educational value**.

---

✨ **Final Result Summary**

| Model | MAE | RMSE | Comment |
|-------|------|-------|----------|
| SARIMA | 4.32 | 5.21 | Baseline (interpretable) |
| XGBoost | **3.91** | **4.80** | Improved accuracy and robustness |
