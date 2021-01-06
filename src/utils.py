from configs import WORKING_DIR, INPUT_DIR
import os
import pandas as pd
import numpy as np

NUMERICS_LIST = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def get_file_path(fname=None):
    input_dir = os.path.join(WORKING_DIR, 'input')
    if fname is None:
        return input_dir
    else:
        return os.path.join(WORKING_DIR, 'input', fname)

def write_df(df, fname='output_df', dir=None):
    if dir is not None:
        file_path = os.path.join(dir, fname)
    else:
        file_path=fname
    print(f'writing to the path: {file_path}')
    df.to_csv(file_path)


def read_df(fname='output_df', dir=None):
    if dir is not None:
        file_path = os.path.join(dir, fname)
    else:
        file_path=fname
    print(f'reading the dataframe: {file_path}')
    return pd.read_csv(file_path)


def select_cols(df, dtype='NUMERIC'):
    if dtype in ['NUMERIC', 'CATEGORICAL']:
        return df.select_dtypes(include=NUMERICS_LIST)
    elif dtype == ['ALL']:
        return df


def isNumeric(df):
    return df.iloc[:,0].dtype in NUMERICS_LIST
        # df.columns.values == df.select_dtypes(include=NUMERICS_LIST).columns.values


def get_columns_with_missing_values(df):
    nas = df.isna().sum()
    nas_df = pd.DataFrame(nas, columns=['null_count'])
    return np.array(nas_df[nas_df.null_count>0].index)


def missing_values_table(df):
        # Total missing values
        # print(self.df.head())


        mis_val = df.isnull().sum()
        n_rows = df.shape[0]
        # print(f'row_count - {n_rows}')

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / n_rows

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns