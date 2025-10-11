import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../DataSet/enhanced_train.csv')
df['date'] = pd.to_datetime(df['date'])
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
