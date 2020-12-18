import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

def read_csv():
    df = pd.read_csv("C:/Users/SCHUAN/PycharmProjects/datascience/data/survey_results_public.csv")  # df with answers
    schema = pd.read_csv("C:/Users/SCHUAN/PycharmProjects/datascience/data/survey_results_schema.csv")  # df with questions
    # df = pd.DataFrame({'A': ['a','f','b','c'], 'B': [1,None,None,4], 'C': [6,7,8,9]})
    return df, schema
df, schema = read_csv()

def get_description(column_name, schema=schema):
    '''
    INPUT - schema - pandas dataframe with the schema of the developers survey
            column_name - string - the name of the column you would like to know about
    OUTPUT -
            desc - string - the description of the column
    '''
    desc = schema.set_index('Column').loc[f'{column_name}']['Question']
    return desc
desc = get_description('')

print('done')