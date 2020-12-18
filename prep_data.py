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
#### prep data
####################################################

#### work with missing values
# remove
df = df[['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction', 'Salary']].dropna()
df  = df.dropna(subset=['Salary'], axis=0)
# impute

# work around

#### dummy all categorical columns in df
def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of categorical columns
    dummy_na - dummy NA vals or not
    OUTPUT:
    df - a new dataframe all non-categorical columns & all categorical columns replaced with dummies
    '''
    for col in cat_cols:
        try:
            dummy_df = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)
            df = pd.concat([df.drop(col, axis=1), dummy_df], axis=1)
        except:
            continue
    return df

cat_df = df.select_dtypes(include=['object'])
cat_cols = cat_df.columns

df_dummies = create_dummy_df(df, cat_cols, dummy_na=True)





print('done')