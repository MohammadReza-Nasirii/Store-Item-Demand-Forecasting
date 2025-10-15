import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns

df = pd.read_csv('DataSet/enhanced_train.csv')

df = df.sort_values('date')

daily_sales = df.groupby('date')['sales'].sum()

plt.figure(figsize=(10, 4))

plt.plot(daily_sales.index, daily_sales.values, label='Daily Sales')

plt.title('Trend of Daily Sales')

plt.xlabel('Date')

plt.ylabel('Total Sales')

plt.legend()

plt.tight_layout()

plt.show()

plt.figure(figsize=(10,4))

sns.histplot(x='sales', data=df, bins=50, kde=True, color='skyblue')

plt.title('Distribution of Sales')

plt.xlabel('Sales')

plt.ylabel('Frequency')

plt.tight_layout()

plt.show()

plt.figure(figsize=(6,4))

sns.boxplot(y='sales', data=df, color='lightcoral')

plt.title('Boxplot of Sales')

plt.ylabel('Sales')

plt.tight_layout()

plt.show()

sales_by_dow = df.groupby('day_of_week')['sales'].mean().reset_index()

plt.figure(figsize=(8,4))

sns.barplot(x='day_of_week', y='sales', data=sales_by_dow, palette='coolwarm')

plt.title('Average Sales by Day of Week')

plt.xlabel('Day of Week (0=Mon, 6=Sun)')

plt.ylabel('Average Sales')

plt.tight_layout()

plt.show()

sales_by_month = df.groupby('month')['sales'].mean().reset_index()

plt.figure(figsize=(8,4))

sns.lineplot(x='month', y='sales', data=sales_by_month, marker='o', color='teal')

plt.title('Average Sales by Month')

plt.xlabel('Month')

plt.ylabel('Average Sales')

plt.tight_layout()

plt.show()

numeric_cols = ['sales', 'day_of_week', 'is_weekend', 'month', 'year']

corr_matrix = df[numeric_cols].corr(method='pearson')

plt.figure(figsize=(8,6))

sns.heatmap(corr_matrix, annot=True, cmap='cool', linewidths=0.5, fmt='.2f')

plt.title('Correlation Heatmap of Numerical Features')

plt.tight_layout()

plt.show()

Q = df['sales'].describe()

Q1 = df['sales'].quantile(0.25)

Q3 = df['sales'].quantile(0.75)

IQR = Q3 - Q1

Lower = Q1 - 1.5 * IQR

Upper = Q3 + 1.5 * IQR

Outliers = df[(df['sales'] < Lower) | (df['sales'] > Upper)]

sns.boxplot(y='sales',  data=df)

plt.show()
