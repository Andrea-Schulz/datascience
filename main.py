import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import custom_functions as func
import analysis_functions as afunc
import prep_data as prep
import plot_functions as plot


####################################################
#### load data
####################################################
df_raw, schema = func.read_csv()
df = df_raw.copy(deep=True)
color1, color2, color3 = plot.color_cycles()


####################################################
#### prep data
####################################################
#### raw dataframe overview
no_rows, no_cols = df.shape
perc_nan, perc_nan_over_x, number_over_x = func.get_nan_perc(df, 0.75)
# search for specific column description
desc = func.get_description('MainBranch')

#### add gender dummy columns
df['Gender_men'] = (df.Gender == 'Man')
df['Gender_women'] = (df.Gender == 'Woman')
df['Gender_other'] = (df.Gender != 'Woman') & (df.Gender != 'Man')

#### add employment dummy column
df['Employment_employed'] = (df.Employment == "Independent contractor, freelancer, or self-employed")|\
                            (df.Employment == "Employed full-time")|\
                            (df.Employment == "Employed part-time")

#### add MainBranch dummy columns
df['MainBranch_prof'] = (df.MainBranch == 'I am a developer by profession')
df['MainBranch_occasion'] = (df.MainBranch == 'I am not primarily a developer, but I write code sometimes as part of my work')

#### create salary bins (yearly salary in USD)
# df.ConvertedComp.describe(); df.ConvertedComp.quantile(.99) # -->99% are below 126k
df['ConvertedComp_bins'] = pd.cut(df['ConvertedComp'], bins=[i * 1000 for i in [0,25,50,75,100,125,150,200,2000]])

#### create age bins
# df.Age.describe(); df.Age.quantile(.99) # -->99% are below 61 years, ignore outliers over 100 years
df['Age_bins'] = pd.cut(df['Age'], bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 99])

#### add dummies for multiple choice columns
DevTypeAnswers, NEWJobHuntAnswers, JobFactorsAnswers = func.get_multiple_choice_answers()
df = func.dummy_multiple_choice(df, 'DevType', DevTypeAnswers)
df = func.dummy_multiple_choice(df, 'NEWJobHunt', NEWJobHuntAnswers)
df = func.dummy_multiple_choice(df, 'JobFactors', JobFactorsAnswers)

#### drop all missing values from target column: job satisfaction
# save df with and without happy columns
df_full = df.copy(deep=True)
df = prep.dropna_subset(df, 'JobSat')
# make 3 simplified categories for target column (excl. NaN)
df, happy_index, happy_index_bins = prep.new_happy_bins(df)

####################################################
#### analyze data: data overview & leading questions
####################################################

####################################################
#### EMPLOYMENT
# ...OVERALL
job_seeking = afunc.feature_overview(df_full, 'JobSeek', dropna=False)
employment_status = afunc.feature_overview(df_full, 'Employment', dropna=False)
main_branch = afunc.feature_overview(df_full, 'MainBranch', dropna=False)
# ...FOR THOSE EMPLOYED AND INCLUDED IN JOBSAT
job_seeking = afunc.feature_overview(df, 'JobSeek', dropna=False)
employment_status = afunc.feature_overview(df, 'Employment', dropna=False)
main_branch = afunc.feature_overview(df, 'MainBranch', dropna=False)


####################################################
#### HAPPINESS OVERALL
# job satisfaction applies only to those with a job --> 30% nan
happy = afunc.feature_overview(df, 'JobSat', dropna=True)
happy.index = happy.index.fillna(value='not answered')
happy = happy.reindex(happy_index)
# plot
happy_pie, happy_ax = plot.pie_chart(happy, 'share',
                                     title=f'Overall Job Satisfaction Among Currently Employed\n'
                                           f'Stackoverflow 2020 Survey Participants',
                                     filename='satisfaction_pie',
                                     colors=['#3EA607', '#5F9343', '#868686', '#93435F', '#A6073E'])


####################################################
#### JOB SEEKING OVERALL (AMONG THOSE WITH A JOB)
seek = afunc.feature_overview(df, 'JobSeek', dropna=True)
# plot
seek_pie, seek_ax = plot.pie_chart(seek, 'share',
                                     title=f'What is Your Current Job-Seeking Status?\n'
                                           f'(Among Developers Currently Employed)',
                                     filename='job_seek_pie',
                                     colors=['#597BB3', '#6459B3', '#9159B3'])


####################################################
#### MOST IMPORTANT PUSH AND PULL FACTORS OVERALL
# PUSH: What drives you to look for a new job?
df_jobhunt = df.filter(regex='NEWJobHunt_', axis=1).dropna().sum().rename('count')
share = df_jobhunt/df.shape[0]
df_jobhunt = pd.concat([df_jobhunt, share.rename('share'), pd.Series(NEWJobHuntAnswers, index=df_jobhunt.index, name='answers')], axis=1).set_index('answers')
# PULL: 3 most important factors in a job apart from salary and benefits
df_factors = df.filter(regex='JobFactors_', axis=1).dropna().sum().rename('count')
share = df_factors/df.shape[0]
df_factors = pd.concat([df_factors, share.rename('share'), pd.Series(JobFactorsAnswers, index=df_factors.index, name='answers')], axis=1).set_index('answers')
# plots
jobhunt_bar, jobhunt_ax = plot.horizontal_bars(df_jobhunt, 'share', percentage=True,
                                               xlabel='Number of mentions in %',
                                               title='What Drives You to Look for a New Job in General?\n'
                                                     '(Select All That Apply)',
                                               filename='jobhunt_bar')
factors_bar, factors_ax = plot.horizontal_bars(df_factors, 'share', percentage=True,
                                               xlabel='Number of mentions in %',
                                               title=f'What Are Your Most Important Factors in Deciding for a Job Offer?\n'
                                                     f'(Apart from Salary, Location and Benefits - Select 3 Most Important)',
                                               filename='factors_bar')


####################################################
#### MOST IMPORTANT PUSH AND PULL FACTORS BY X
#### Developer Type
df_dev = df.filter(regex='DevType_', axis=1).dropna().sum().rename('count')
share = df_dev/df.shape[0]
df_dev = pd.concat([df_dev, share.rename('share'), pd.Series(DevTypeAnswers, index=df_dev.index, name='answers')], axis=1).set_index('answers')


####################################################
#### HAPPINESS FACTORS
#### correlations for categorical columns


####...by MainBranch
# job satisfaction applies only to those with a job (=developer or code as part of work)
branch_count, branch_share = afunc.feature_by_x(df, 'JobSat', 'MainBranch')
branch_share = branch_share.reindex(happy_index)
branch_share_diff = branch_share.sub(happy['share'], axis=0)
# branch_share_diff.plot(kind='bar')
# plt.tight_layout()
# developers by profession (with 90% of those currently working by far the larger group) are well represented, slightly more happy
# proportion of people "very satisfied" is almost 25% less for those who only work with code as part of their work


####...by salary
sal_count, sal_share = afunc.feature_by_x(df, 'JobSat', 'ConvertedComp_bins')
sal_share = sal_share.reindex(happy_index)
sal_share_diff = sal_share.sub(happy['share'], axis=0)  # get diffs concerning overall satisfaction
# sal_share_diff.plot(kind='bar')
# plt.tight_layout()
# people seem to tend more towards being "very satisfied" with increasing salary
# proportion of people with lower salary which are "very satisfied" is >12% less than the general group
# for the other ratings there is no clear trend in terms of more salary - more satisfaction


####...by JobSeek status (developer position)
# DevType only for (former) professional developers
job_count, job_share = afunc.feature_by_x(df, 'JobSat', 'JobSeek')
job_share = job_share.reindex(happy_index)
job_share_diff = job_share.sub(happy['share'], axis=0)
# job_share_diff.plot(kind='bar')
# plt.tight_layout()
# very happy people not interested / very unhappy people looking more than general group
# slightly dissatisfied/satisfied people are looking/open to opportunities



####...by NEWOvertime
working_count, working_share = afunc.feature_by_x(df, 'JobSat', 'NEWOvertime')
working_share = working_share.reindex(happy_index)
working_share_diff = working_share.sub(happy['share'], axis=0)
# working_share_diff.plot(kind='bar')
# plt.tight_layout()

####...by Hobbyist
hobby_count, hobby_share = afunc.feature_by_x(df, 'JobSat', 'Hobbyist')
hobby_share = hobby_share.reindex(happy_index)
hobby_share_diff = hobby_share.sub(happy['share'], axis=0)
# hobby_share_diff.plot(kind='bar')
# plt.tight_layout()

####...by NEWLearn
learn_count, learn_share = afunc.feature_by_x(df, 'JobSat', 'NEWLearn')
learn_share = learn_share.reindex(happy_index)
learn_share_diff = learn_share.sub(happy['share'], axis=0)
# learn_share_diff.plot(kind='bar')
# plt.tight_layout()
# similar distribution: no clear correlation

####...by PurchaseWhat status (developer position)
purchase_count, purchase_share = afunc.feature_by_x(df, 'JobSat', 'PurchaseWhat')
purchase_share = purchase_share.reindex(happy_index)
purchase_share_diff = purchase_share.sub(happy['share'], axis=0)
# purchase_share_diff.plot(kind='bar')
# plt.tight_layout()
# seems to be a major boost for people to be "Very satisfied" (+12% compared to people with no/little influence)


####################################################
#### JOB PUSH AND PULL FACTORS

df_type = df.filter(regex='DevType_', axis=1)
df_factors = df.filter(regex='JobFactors_', axis=1)
df_hunt = df.filter(regex='NEWJobHunt_', axis=1)

df_type.sum().sort_values()
df_factors.sum().sort_values()
df_hunt.sum().sort_values()


####...by Gender
happy_gender = afunc.happiness_by_gender(df, happy_index)
happy_gender.plot(kind='bar')



print('done')
print('done')