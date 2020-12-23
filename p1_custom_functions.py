import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# get data as dataframe
def read_csv():
    df = pd.read_csv("C:/Users/Andrea/PycharmProjects/datascience/data/StackOverflow2020/survey_results_public.csv")  # df with answers
    schema = pd.read_csv("C:/Users/Andrea/PycharmProjects/datascience/data/StackOverflow2020/survey_results_schema.csv")  # df with questions
    return df, schema
df, schema = read_csv()

# get description of a column (desc) as string by providing the column name (column_name)
def get_description(column_name, schema=schema):
    desc = schema.set_index('Column').loc[f'{column_name}']['QuestionText']
    return desc

def get_multiple_choice_answers():
    DevTypeAnswers = ["Academic researcher",
                    "Data or business analyst",
                    "Data scientist or machine learning specialist",
                    "Database administrator",
                    "Designer",
                    "Developer, back-end",
                    "Developer, desktop or enterprise applications",
                    "Developer, embedded applications or devices",
                    "Developer, front-end",
                    "Developer, full-stack",
                    "Developer, game or graphics",
                    "Developer, mobile",
                    "Developer, QA or test",
                    "DevOps specialist",
                    "Educator",
                    "Engineer, data",
                    "Engineer, site reliability",
                    "Engineering manager",
                    "Marketing or sales professional",
                    "Product manager",
                    "Scientist",
                    "Senior Executive (C-Suite, VP, etc.)",
                    "System administrator"]
    NEWJobHuntAnswers = ['Just because',
                        'Having a bad day (or week or month) at work',
                        'Wanting to share accomplishments with a wider network',
                        'Curious about other opportunities',
                        'Better compensation',
                        'Trouble with my teammates',
                        'Trouble with my direct manager',
                        'Trouble with leadership at my company',
                        'Better work/life balance',
                        'Wanting to work with new technologies',
                        'Growth or leadership opportunities',
                        'Looking to relocate']
    JobFactorsAnswers = ['Remote work options',
                        'Office environment or company culture',
                        'Financial performance or funding status of the company or organization',
                        'Opportunities for professional development',
                        'Diversity of the company or organization',
                        'How widely used or impactful my work output would be',
                        'Industry that I’d be working in',
                        'Specific department or team I’d be working on',
                        'Flex time or a flexible schedule',
                        'Languages, frameworks, and other technologies I’d be working with',
                        'Family friendliness']
    return DevTypeAnswers, NEWJobHuntAnswers, JobFactorsAnswers

def dummy_multiple_choice(df, column, answer_list):
    for x in answer_list:
        sername = column + '_' + str(answer_list.index(x))
        ser = df[column].str.contains(x).rename(sername).to_frame()*1
        df = pd.concat([df, ser], axis=1)
    return df

# get the percentage of NaN values (perc_nan) and the number of columns with more than x percent of values missing (perc_nan_over_x) in df
def get_nan_perc(df, threshold):
    # columns with corresponding percentage of nan values
    perc_nan = df.isnull().sum()/len(df)
    # columns with more than x percent of nan values
    perc_nan_over_x = (df.isnull().sum()/len(df)) > threshold
    # number of columns
    number_over_x = np.sum(perc_nan > threshold)
    return perc_nan, perc_nan_over_x, number_over_x

# create dummy columns for all non-numerical columns, dropping the original column
def create_dummy_df(df, dummy_na=True):
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        try:
            dummy_df = pd.get_dummies(df[col].astype('category'), prefix=col, prefix_sep='_', drop_first=False, dummy_na=dummy_na)
            df = pd.concat([df.drop(col, axis=1), dummy_df], axis=1)
        except:
            continue
    return df
