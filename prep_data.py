import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import custom_functions as f
import analysis_functions as a


def dropna_subset(df, subset):
    # drop all missing values from target column
    df = df.dropna(subset=[subset]).copy(deep=True)
    return df

def new_happy_bins(df):
    # make 3 new categories for target column (excl. NaN)
    df['JobSat_bins'] = df['JobSat'].replace({'Very satisfied': 'satisfied',
                                              'Slightly satisfied': 'satisfied',
                                              'Very dissatisfied': 'dissatisfied',
                                              'Slightly dissatisfied': 'dissatisfied',
                                              'Neither satisfied nor dissatisfied': 'neither'})
    happy_index = ['Very satisfied', 'Slightly satisfied', 'Neither satisfied nor dissatisfied',
                   'Slightly dissatisfied', 'Very dissatisfied', 'not answered']
    happy_index_bins = ['satisfied', 'dissatisfied', 'neither']
    return df, happy_index, happy_index_bins
