import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

Data = pd.read_csv('DataSet/test.csv')

Data.set_index('id', inplace=True)

Data['date'] = pd.to_datetime(Data['date'])

Data['month_day'] = Data['date'].dt.month.astype(str).str.zfill(2) + '-' + Data['date'].dt.day.astype(str).str.zfill(2)

Data['day_of_week'] = Data['date'].dt.dayofweek

Data['month'] = Data['date'].dt.month

Data['year'] = Data['date'].dt.year

Data['is_weekend'] = (Data['day_of_week'] >= 5).astype(int)
#
# print("Shape of dataset:", Data.shape)
#
# print("Dataset info:\n", Data.info())
#
# print("Descriptive statistics:\n", Data.describe())
#
# print("Missing values per column:\n", Data.isnull().sum())

store_counts = Data['store'].value_counts()

item_counts = Data['item'].value_counts()

# print("Count of records per store:\n", store_counts)
#
# print("Count of records per item:\n", item_counts)

plt.figure(figsize=(10, 6))

store_counts.plot(kind='bar', color='skyblue')

plt.title('Distribution of Stores')

plt.xlabel('Store')

plt.ylabel('Count')

# plt.show()

plt.figure(figsize=(10, 6))

item_counts.plot(kind='bar', color='lightgreen')

plt.title('Distribution of Items')

plt.xlabel('Item')

plt.ylabel('Count')

# plt.show()

daily_counts = Data['date'].value_counts().sort_index()

# print("Daily record counts:\n", daily_counts.head())