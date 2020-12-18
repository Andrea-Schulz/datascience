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
#### supervised learning
####################################################

#### prep data:
# remove missing values
df = df[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction', 'Salary']].dropna()

#### define datasets
X = df[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=42)

#### Four steps:

#Instantiate
lm_model = LinearRegression(normalize=True)
#Fit - why does this break?
lm_model.fit(X_train, y_train)
#Predict
y_test_preds = lm_model.predict(X_test)
#Score
r2_test = r2_score(y_test, y_test_preds)


print('done')