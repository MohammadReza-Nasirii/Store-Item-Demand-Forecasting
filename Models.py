import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
from sklearn.model_selection import train_test_split

Train = pd.read_csv('DataSet/enhanced_train.csv')

Test = pd.read_csv('DataSet/enhanced_test.csv')

print(Train.head())
print(Test.head())