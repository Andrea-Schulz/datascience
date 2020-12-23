import pandas as pd


def happiness_by_gender(df, happy_index):
    men = df.loc[df.Gender == 'Man']
    women = df.loc[df.Gender == 'Woman']
    other = df.loc[df.Gender.astype('string').str.contains('Non-binary, genderqueer, or gender non-conforming')]

    happy_male = (men['Job_Satisfaction'].value_counts(dropna=False)) / men.shape[0]
    happy_female = (women['Job_Satisfaction'].value_counts(dropna=False)) / women.shape[0]
    happy_other = (other['Job_Satisfaction'].value_counts(dropna=False)) / other.shape[0]

    happy_gender = pd.DataFrame({'happy_male': happy_male, 'happy_female': happy_female, 'happy_other': happy_other})
    happy_gender.index = happy_gender.index.fillna(value='not answered')
    happy_gender = happy_gender.reindex(happy_index)

    return happy_gender

def feature_overview(df, feature_column, dropna=False):
    val_counts = df[feature_column].value_counts(dropna=dropna)
    overview_df = pd.DataFrame({'count': val_counts, 'share': val_counts / val_counts.sum()})
    return overview_df

def feature_by_x(df, feature, x_column):
    # get feature counts per x_column & unstack multi-index
    counts = df.groupby(x_column)[feature].value_counts().unstack(level=0)
    # get feature percentages per x_column & unstack multi-index (get x by feature dataframe)
    shares = counts.apply(lambda x: x/x.sum())
    return counts, shares

def get_multiple_choice(df, column_regex, answerlist_mc):
    df_mc = df.filter(regex=column_regex, axis=1).dropna().sum().rename('count')
    share = df_mc / df.shape[0]
    df_mc = pd.concat\
        ([df_mc, share.rename('share'), pd.Series(answerlist_mc, index=df_mc.index, name='answers')], axis=1)\
        .set_index('answers')
    return df_mc
