######
# 1. Randomly get column combinations
# 2. Filter by column types
# 3. Numerical columns - Feature engineering
# 4. Categorical columns - Feature engineering
# 5. Combine all the features
# 6. Fit a model across for each fold
# 7. calculate the accuracy - rmse
# 8. minimize rmse
# 9. save whenever it beats the previous best


import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
# from sklearn import metrics
import joblib
from utils import select_dtype_columns

from metrics import RegressionMetrics
import optuna

import sys


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
TARGET = os.environ.get('TARGET')


# print(f'Test data: {TEST_DATA}')
# print(f'Train data: {TRAINING_DATA}')
# print(f'Fold is {FOLD}')

# sys.exit()

# FOLD = int(os.environ.get("FOLD"))
# MODEL = os.environ.get("MODEL")
SCALAR = os.environ.get("SCALAR")
SEED = 42

np.random.seed(SEED)

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

best_error = 100000

def objective(trial):

    global best_error

    regressor_name = trial.suggest_categorical("regressor", ["extratrees", "randomforest"])
    if regressor_name == 'randomforest':
        n_estimators = trial.suggest_int("rf_trees", 50, 1500)
        max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        max_features = trial.suggest_float('rf_max_features', 0.01, 1)

        clf = ensemble.RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                              max_depth=max_depth, max_features=max_features)
    elif regressor_name == 'extratrees':
        n_estimators = trial.suggest_int("ex_trees", 50, 1500)
        max_depth = trial.suggest_int("ex_max_depth", 2, 32, log=True)
        # max_features = trial.suggest_float('rf_max_features', 0.01, 1)

        clf = ensemble.RandomForestRegressor(n_estimators=n_estimators,n_jobs=-1,
                                               max_depth=max_depth)
    errors = []
    for fold in df.kfold.unique():
        train_df = df[df.kfold.isin(FOLD_MAPPPING.get(fold))].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(["Id", "target", "kfold"], axis=1)
        valid_df = valid_df.drop(["Id", "target", "kfold"], axis=1)

        valid_df = valid_df[train_df.columns]

        clf.fit(train_df, ytrain)
        preds = clf.predict(valid_df)

        reg_metrics = RegressionMetrics()
        metric = reg_metrics(yvalid, preds, 'rmse')
        errors.append(metric['rmse'])

        # joblib.dump(train_df, f"../input/train_df_optuna_{fold}.pkl")
        # joblib.dump(valid_df, f"../input/valid_df_optuna_{fold}.pkl")

    if best_error > np.mean(errors):
        best_error = np.mean(errors)
        print(f'best_accuracy is {best_error}')

        clf.fit(df.drop(columns=["Id", "target", "kfold"], axis=1), df.target.values)
        joblib.dump(clf, f"../models/optuna/median_{regressor_name}_{int(best_error)}.pkl")
    print(errors)
    return np.mean(errors)



if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)


    df = df.rename(columns={TARGET:'target'})
    df_test = df_test.rename(columns={TARGET: 'target'})


    label_encoders = {}
    for c in select_dtype_columns(df, dtype='CATEGORICAL', sub_dtype='object'):
        lbl = preprocessing.LabelEncoder()
        df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
        # valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(df[c].values.tolist() +
                # valid_df[c].values.tolist() +
                df_test[c].values.tolist())
        df.loc[:, c] = lbl.transform(df[c].values.tolist())
        # valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    # data is ready to train

    # if MODEL in ['linear_regression', 'knn', 'logistic']:
    #     SCALAR = os.environ.get("SCALAR")
    #     scalar_model = dispatcher.NORMALIZER[SCALAR]
    #     train_df = pd.DataFrame(scalar_model.fit_transform(train_df), columns=train_df.columns)
    #     valid_df = pd.DataFrame(scalar_model.transform(valid_df), columns=valid_df.columns)
    #     joblib.dump(scalar_model, f"../models/{SCALAR}_{FOLD}_normalizer.pkl")

    best_error = 100000

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(objective, n_trials=100, timeout=600)

