import glob
import numpy as np
import pandas as pd

if __name__=='__main__':
    files = glob.glob('../model_preds/*.csv')
    preds_df=pd.DataFrame()
    for f in files:
        df= pd.read_csv(f)

