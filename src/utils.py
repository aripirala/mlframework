from configs import WORKING_DIR, INPUT_DIR
import os
import pandas as pd

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
