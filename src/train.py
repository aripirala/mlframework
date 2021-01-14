import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from utils import select_dtype_columns
from metrics import RegressionMetrics

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f','--fold', required=True)
# parser.add_argument('-o','--ofile', required=True)
args = parser.parse_args()

FOLD = int(args.fold)

import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
TARGET = os.environ.get('TARGET')


# print(f'Test data: {TEST_DATA}')
# print(f'Train data: {TRAINING_DATA}')
# print(f'Fold is {FOLD}')

# sys.exit()

# FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
SCALAR = os.environ.get("SCALAR")

# print(f'Model is {MODEL}')
# print(f'SCALAR is {SCALAR}')
# print(f'FOLLD is {FOLD}')
# print(f'Target is {TARGET}')


FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)


    df = df.rename(columns={TARGET:'target'})
    df_test = df_test.rename(columns={TARGET: 'target'})


    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["Id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["Id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = {}
    for c in select_dtype_columns(train_df, dtype='CATEGORICAL', sub_dtype='object'):
        lbl = preprocessing.LabelEncoder()
        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(train_df[c].values.tolist() +
                valid_df[c].values.tolist() +
                df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # data is ready to train

    if MODEL in ['linear_regression', 'knn', 'logistic']:
        SCALAR = os.environ.get("SCALAR")
        scalar_model = dispatcher.NORMALIZER[SCALAR]
        train_df = pd.DataFrame(scalar_model.fit_transform(train_df), columns=train_df.columns)
        valid_df = pd.DataFrame(scalar_model.transform(valid_df), columns=valid_df.columns)
        joblib.dump(scalar_model, f"../models/{SCALAR}_{FOLD}_normalizer.pkl")

    clf = dispatcher.MODELS[MODEL]
    # print(train_df.head())
    # print(train_df.shape)
    # print(ytrain.shape)

    clf.fit(train_df, ytrain)
    preds = clf.predict(valid_df)


    reg_metrics = RegressionMetrics()
    metric = reg_metrics(yvalid, preds, 'rmse')
    print(f"Fold: {FOLD} - metric: {metric}")

    joblib.dump(train_df, f"../input/train_df_{FOLD}.pkl")
    joblib.dump(valid_df, f"../input/valid_df_{FOLD}.pkl")

    joblib.dump(clf, f"../models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"../models/{MODEL}_{FOLD}_columns.pkl")
