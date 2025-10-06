import kagglehub
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Download latest version
#path = kagglehub.dataset_download("muharremg/dataset-demand-forecasting")

#print("Path to dataset files:", path)

# Destination = 'DataSet/'

#shutil.copytree(path, Destination, dirs_exist_ok=True)

#print(f"Dataset copied to: {Destination}")

# Data = pd.read_csv('DataSet/test.csv')
#
Data_2 = pd.read_csv('DataSet/train.csv')

print(Data_2.head())
#
# Data.set_index('id', inplace=True)
#
# Data['date'] = pd.to_datetime(Data['date'])
#
# Data['date'] = Data['date'].dt.month.astype(str).str.zfill(2) + '-' + Data['date'].dt.day.astype(str).str.zfill(2)
#
# print(Data.shape)
#
# print(Data.head())


train = pd.read_csv('Dataset/train.csv')

train.set_index(keys='id', inplace=True)

train['date'] = pd.to_datetime(train['date'])

train['month_day'] = train['date'].dt.month.astype(str).str.zfill(2) + '-' + train['date'].dt.day.astype(str).str.zfill(2)

train['day_of_week'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)

print("شکل دیتاست:", train.shape)
print("اطلاعات دیتاست:\n", train.info())
print("آمار توصیفی:\n", train.describe())
print("مقدار خالی در هر ستون:\n", train.isnull().sum())

item_sales_mean = train.groupby('item')['sales'].mean()
print("میانگین فروش بر اساس item:\n", item_sales_mean.head())
plt.figure(figsize=(10, 6))
plt.hist(train['sales'], bins=50, color='skyblue')
plt.title('توزیع فروش')
plt.xlabel('فروش')
plt.ylabel('تعداد')
plt.show()


plt.figure(figsize=(12, 6))
item_sales_mean.plot(kind='bar')
plt.title('میانگین فروش بر اساس آیتم')
plt.xlabel('آیتم')
plt.ylabel('میانگین فروش')
plt.show()

