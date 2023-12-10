import pandas as pd
import os

def get_data_frame(file):

    # current directory
    current_dir = os.path.dirname('src')

    # directory of data relative to current irectory
    data_file_path = os.path.join(current_dir, '..', 'data', file)

    # read the csv file
    df = pd.read_csv(data_file_path)

    df = remove_outliers(df)

    df.rename(columns={
        'alpha': 'right_ascension',
        'delta': 'declination',
        'u': 'ultraviolet_filter',
        'g' : 'green_filter',
        'r' : 'red_filter',
        'i' : 'near_infrared_filter',
        'z' : 'infrared_filter'
        }, inplace=True)

    return df

def remove_outliers(df):
    outlier_indices = []

    # Iterate over each numeric column to find outliers
    for i in df.select_dtypes(include='number').columns:

        # Find interquartile range
        qt1 = df[i].quantile(0.25)
        qt3 = df[i].quantile(0.75)
        iqr = qt3 - qt1

        # Get upper and lower bounds
        lower = qt1 - (1.5 * iqr)
        upper = qt3 + (1.5 * iqr)

        # Find indices of outliers in the current column
        column_outliers = df[(df[i] < lower) | (df[i] > upper)].index

        # Append these indices to the list
        outlier_indices.extend(column_outliers)

    # Find unique indices
    unique_outlier_indices = list(set(outlier_indices))

    # Drop the rows with indices that are identified as outliers
    df_cleaned = df.drop(unique_outlier_indices)

    return df_cleaned
