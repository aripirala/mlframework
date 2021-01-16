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
from sklearn import svm, linear_model
import lightgbm as lgb

# from sklearn import metrics
import joblib
from utils import select_dtype_columns, select_dtype_data, select_not_dtype_data
from utils import write_df

from metrics import RegressionMetrics
import optuna

import sys
import dispatcher


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

    regressor_name = trial.suggest_categorical("regressor", ['lightgbm']) #['elastinet','ridge', "extratrees", "randomforest", 'svm'])
    if regressor_name == 'randomforest':
        n_estimators = trial.suggest_int("rf_trees", 50, 1500)
        max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        max_features = trial.suggest_float('rf_max_features', 0.01, 1)

        reg = ensemble.RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                              max_depth=max_depth, max_features=max_features)
    elif regressor_name == 'extratrees':
        n_estimators = trial.suggest_int("ex_trees", 50, 1500)
        max_depth = trial.suggest_int("ex_max_depth", 2, 32, log=True)
        # max_features = trial.suggest_float('rf_max_features', 0.01, 1)

        reg = ensemble.RandomForestRegressor(n_estimators=n_estimators,n_jobs=-1,
                                               max_depth=max_depth)
    elif regressor_name == 'svm':
        kernel= trial.suggest_categorical("kernel", ['linear', 'rbf'])
        cost = trial.suggest_float('cost', 0.01, 100)

        reg = svm.SVR(kernel=kernel, C=cost, )

    elif regressor_name == 'ridge':
        alpha = trial.suggest_float('alpha', 0.01, 100)
        reg= linear_model.Ridge(alpha=alpha)

    elif regressor_name == 'elastinet':
        alpha = trial.suggest_float('alpha', 0.01, 100)
        l1_ratio = trial.suggest_float('l1_ratio', 0.01, 1.0)
        reg = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    elif regressor_name == 'lightgbm':
        param = {
            "objective": "regression",
            "metric": {'l2', 'l1'},
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int('num_leaves', 15, 50),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        # Add a callback for pruning.
        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "l2")


    errors = []
    preds_df = pd.DataFrame()
    for fold in df.kfold.unique():
        train_df = df[df.kfold.isin(FOLD_MAPPPING.get(fold))].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(["Id", "target", "kfold"], axis=1)
        valid_df = valid_df.drop(["Id", "target", "kfold"], axis=1)

        valid_df = valid_df[train_df.columns]


        if regressor_name in ['linear_regression', 'knn', 'svm', 'ridge', 'elastinet']:
            SCALAR = os.environ.get("SCALAR")
            scalar_model = dispatcher.NORMALIZER[SCALAR]

            train_num_df = select_dtype_data(train_df, dtype='NUMERIC')
            valid_num_df = select_dtype_data(valid_df, dtype='NUMERIC')

            train_rest_df = select_not_dtype_data(train_df, dtype='NUMERIC')
            valid_rest_df = select_not_dtype_data(valid_df, dtype='NUMERIC')

            train_num_df = pd.DataFrame(scalar_model.fit_transform(train_num_df), columns=train_num_df.columns)
            valid_num_df = pd.DataFrame(scalar_model.transform(valid_num_df), columns=valid_num_df.columns)

            train_df = pd.concat([train_num_df, train_rest_df], axis=1)
            valid_df = pd.concat([valid_num_df, valid_rest_df], axis=1)

            joblib.dump(scalar_model, f"../models/optuna/median_trial3_{SCALAR}_{fold}_normalizer.pkl")

        if regressor_name == 'lightgbm':
            lgb_train = lgb.Dataset(train_df, ytrain)
            lgb_valid = lgb.Dataset(valid_df, yvalid, reference=lgb_train)

            reg = lgb.train(
                param, lgb_train, valid_sets=[lgb_valid], verbose_eval=False, callbacks=[pruning_callback]
            )
            preds = reg.predict(valid_df, num_iteration=reg.best_iteration)
        else:
            reg.fit(train_df, ytrain)
            preds = reg.predict(valid_df)

        fold_preds_df = pd.DataFrame(preds, columns=[f'preds'])
        fold_preds_df['kfold']= fold
        fold_preds_df['Id'] = df[df.kfold == fold].reset_index(drop=True)['Id'].values
        preds_df = pd.concat([preds_df, fold_preds_df], axis=0)


        reg_metrics = RegressionMetrics()
        metric = reg_metrics(yvalid, preds, 'rmse')
        errors.append(metric['rmse'])

        # joblib.dump(train_df, f"../input/train_df_optuna_{fold}.pkl")
        # joblib.dump(valid_df, f"../input/valid_df_optuna_{fold}.pkl")

    if best_error > np.mean(errors):
        best_error = np.mean(errors)
        print(f'best_accuracy is {best_error}')
        if regressor_name != 'lightgbm':
            reg.fit(df.drop(columns=["Id", "target", "kfold"], axis=1), df.target.values)
        preds_df.columns=[f'preds_{regressor_name}_{int(best_error)}', 'kfold', 'Id']
        write_df(preds_df, f'../model_preds/preds_{regressor_name}_{int(best_error)}.csv')

        joblib.dump(reg, f"../models/optuna/cat_trial3_{regressor_name}_{int(best_error)}.pkl")
    # print(errors)
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

    study.optimize(objective, n_trials=200, timeout=600)

