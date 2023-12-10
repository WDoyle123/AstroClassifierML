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

    return df

def remove_outliers(df):

    for i in df.select_dtypes(include = 'number').columns:
        qt1 = df[i].quantile(0.25)
        qt3 = df[i].quantile(0.75)
        iqr = qt3 - qt1
        lower = qt1 - (1.5 * iqr)
        upper = qt3 + (1.5 * iqr)
        min_index = df[df[i] < lower].index
        max_index = df[df[i] > upper].index
        df.drop(min_index, inplace=True)
        df.drop(max_index, inplace=True)

        return df

