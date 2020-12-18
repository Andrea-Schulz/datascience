import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import read_csv as rcsv

df, schema = rcsv.read_csv()

####################################################
#### stats
# summary statistics associated with the quantitative variables in the dataset
print(df.shape)
df.describe()



print('done')