
def dropna_subset(df, subset):
    # drop all missing values from target column
    df = df.dropna(subset=[subset]).copy(deep=True)
    return df

def new_happy_bins(df):
    # make 3 new categories for target column (excl. NaN)
    df['Job_Satisfaction_bins'] = df['Job_Satisfaction'].replace({'Very satisfied': 'satisfied',
                                              'Slightly satisfied': 'satisfied',
                                              'Very dissatisfied': 'dissatisfied',
                                              'Slightly dissatisfied': 'dissatisfied',
                                              'Neither satisfied nor dissatisfied': 'neither'})
    happy_index = ['Very satisfied', 'Slightly satisfied', 'Neither satisfied nor dissatisfied',
                   'Slightly dissatisfied', 'Very dissatisfied', 'not answered']
    happy_index_bins = ['satisfied', 'dissatisfied', 'neither']
    return df, happy_index, happy_index_bins
