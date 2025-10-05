import kagglehub
import shutil
import pandas as pd

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
