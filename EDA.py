import pandas as pd
import matplotlib.pyplot as plt
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