import pandas as pd
import p1_custom_functions as func
import p1_analysis_functions as afunc
import p1_prep_data as prep
import p1_plot_functions as plot


####################################################
#### load data
####################################################
df_raw, schema = func.read_csv()
df = df_raw.copy(deep=True)
df = df.rename(columns={"PurchaseWhat": "Influence_On_Purchases", "JobSat": "Job_Satisfaction"})

####################################################
#### prep data
####################################################
#### raw dataframe overview
no_rows, no_cols = df.shape
perc_nan, perc_nan_over_x, number_over_x = func.get_nan_perc(df, 0.75)
desc = func.get_description('MainBranch')

#### create salary bins (yearly salary in USD)
# df.ConvertedComp.describe(); df.ConvertedComp.quantile(.99) # -->99% are below 126k
# df['Salary_Group'] = pd.cut(df['ConvertedComp'], bins=[i * 1000 for i in [0,25,50,75,100,125,150,200,2000]])
df['Salary_Group'] = pd.cut(df['ConvertedComp'], bins=[0, df.ConvertedComp.median(), 2000000], labels=["below median", "above median"])

#### add dummies for multiple choice columns to extract number of individual mentions
DevTypeAnswers, NEWJobHuntAnswers, JobFactorsAnswers = func.get_multiple_choice_answers()
df = func.dummy_multiple_choice(df, 'DevType', DevTypeAnswers)
df = func.dummy_multiple_choice(df, 'NEWJobHunt', NEWJobHuntAnswers)
df = func.dummy_multiple_choice(df, 'JobFactors', JobFactorsAnswers)

#### drop all missing values from target column: job satisfaction
# save df with and without happy columns
df_full = df.copy(deep=True)
df = prep.dropna_subset(df, 'Job_Satisfaction')
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
# ...FOR THOSE EMPLOYED AND INCLUDED IN Job_Satisfaction
job_seeking = afunc.feature_overview(df, 'JobSeek', dropna=False)
employment_status = afunc.feature_overview(df, 'Employment', dropna=False)
main_branch = afunc.feature_overview(df, 'MainBranch', dropna=False)


####################################################
#### HAPPINESS OVERALL
# job satisfaction applies only to those with a job --> 30% nan
happy = afunc.feature_overview(df, 'Job_Satisfaction', dropna=True)
happy.index = happy.index.fillna(value='not answered')
happy = happy.reindex(happy_index)
# plot
happy_pie, happy_ax = plot.pie_chart(happy, 'share',
                                     title=f'Overall Job Satisfaction Among Currently Employed\n'
                                           f'Stackoverflow 2020 Survey Participants',
                                     filename='satisfaction_pie',
                                     colors=['#3EA607', '#5F9343', '#868686', '#93435F', '#A6073E'])


####################################################
#### JOB SEEKING STATUS (AMONG THOSE WITH A JOB)
seek = afunc.feature_overview(df, 'JobSeek', dropna=True)
# plot
seek_pie, seek_ax = plot.pie_chart(seek, 'share',
                                     title=f'What is Your Current Job-Seeking Status?\n'
                                           f'(Among Developers Currently Employed)',
                                     filename='job_seek_pie',
                                     colors=['#597BB3', '#6459B3', '#9159B3'])


####################################################
#### MOST IMPORTANT PUSH AND PULL FACTORS OVERALL

jobhunt = df['NEWJobHunt'].astype('string').str.get_dummies(sep=';')
jobhunt_count = jobhunt.sum().rename('count')
df_jobhunt = pd.concat([jobhunt_count, (jobhunt_count/jobhunt.shape[0]).rename('share')], axis=1)

factors = df['JobFactors'].astype('string').str.get_dummies(sep=';')
factors_count = factors.sum().rename('count')
df_factors = pd.concat([factors_count, (factors_count/factors.shape[0]).rename('share')], axis=1)



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

#### by MainBranch
# filter for MainBranch
df_professionals = df.loc[df.MainBranch == 'I am a developer by profession']
df_occasionals = df.loc[df.MainBranch == 'I am not primarily a developer, but I write code sometimes as part of my work']
# get push/pull factors by MainBranch
df_jobhunt_prof = (afunc.get_multiple_choice(df_professionals, 'NEWJobHunt_', NEWJobHuntAnswers)).add_prefix('prof_')
df_jobhunt_occ = (afunc.get_multiple_choice(df_occasionals, 'NEWJobHunt_', NEWJobHuntAnswers)).add_prefix('occ_')
df_factors_prof = (afunc.get_multiple_choice(df_professionals, 'JobFactors_', JobFactorsAnswers)).add_prefix('prof_')
df_factors_occ = (afunc.get_multiple_choice(df_occasionals, 'JobFactors_', JobFactorsAnswers)).add_prefix('occ_')
# add to df
df_jobhunt = pd.concat([df_jobhunt_prof['prof_share'], df_jobhunt_occ['occ_share']], axis=1)
df_factors = pd.concat([df_factors_prof['prof_share'], df_factors_occ['occ_share']], axis=1)
# plots
df_factors = df_factors[['share', 'prof_share', 'occ_share']].plot(kind='bar')
# plot
factors_bar, factors_ax = plot.horizontal_bars(df_factors, 'share', percentage=True,
                                               xlabel='Number of mentions in %',
                                               title=f'What Are Your Most Important Factors in Deciding for a Job Offer?\n'
                                                     f'(Apart from Salary, Location and Benefits - Select 3 Most Important)',
                                               filename='factors_bar')

####################################################
#### HAPPINESS FACTORS

####...by MainBranch
# job satisfaction applies only to those with a job (=developer or code as part of work)
branch_count, branch_share = afunc.feature_by_x(df, 'Job_Satisfaction', 'MainBranch')
branch_share = branch_share.reindex(happy_index)
branch_share_diff = branch_share.sub(happy['share'], axis=0)
# plot
fig, ax = plot.horizontal_bars_df(branch_share_diff, title='Job Satisfaction Ratings by Job Branch in [%] Compared to Overall Job Satisfaction',
                                  color=['#8C979F', '#9E8C9F'], filename='happiness_by_branch', percentage=True)

#### by self-employment
# filter for employment status
df_selfemp = df.loc[df.Employment == "Independent contractor, freelancer, or self-employed"]
df_emp = df.loc[(df.Employment == "Employed full-time")|(df.Employment == "Employed part-time")]
# get job satisfaction ratings for groups
happy_selfemp = (afunc.feature_overview(df_selfemp, 'Job_Satisfaction', dropna=True)).add_prefix('selfemp_')
happy_emp = (afunc.feature_overview(df_emp, 'Job_Satisfaction', dropna=True)).add_prefix('emp_')
# compare to average
df_employment = pd.concat([(happy_emp['emp_share'].sub(happy['share'], axis=0)).rename('employed'),
                           (happy_selfemp['selfemp_share'].sub(happy['share'], axis=0)).rename('self-employed')],
                          axis=1)
# plot
fig, ax = plot.horizontal_bars_df(df_employment.reindex(happy_index),
                                  title='Job Satisfaction Ratings by Employment Status in [%] Compared to Overall Job Satisfaction',
                                  color=['#9F948C', '#9E8C9F'], filename='happiness_by_employment', percentage=True)

####...by Influence On Purchases (developer position)
purchase_count, purchase_share = afunc.feature_by_x(df, 'Job_Satisfaction', 'Influence_On_Purchases')
purchase_share = purchase_share.reindex(happy_index)
purchase_share_diff = purchase_share.sub(happy['share'], axis=0)
# plot
fig, ax = plot.horizontal_bars_df_multi(purchase_share_diff.reindex(happy_index).dropna(),
                                        title='Job Satisfaction Ratings by Technical Decisionmaking Competency in [%]\n'
                                              'Compared to Overall Job Satisfaction',
                                        color=['#A26B61', '#6198A2', '#9E8C9F'],
                                        filename='happiness_by_purchase', percentage=True)

####...by salary
sal_count, sal_share = afunc.feature_by_x(df, 'Job_Satisfaction', 'Salary_Group')
sal_share = sal_share.reindex(happy_index)
sal_share_diff = sal_share.sub(happy['share'], axis=0)  # get diffs concerning overall satisfaction
sal_share_diff.index = sal_share_diff.index.rename('Job Satisfaction')
# plot
fig, ax = plot.horizontal_bars_df(sal_share_diff.reindex(happy_index),
                                  title='Job Satisfaction Ratings by Salary in [%] Compared to Overall Job Satisfaction',
                                  color=['#A26B61', '#6198A2'], filename='happiness_by_salary', percentage=True)


print('done')