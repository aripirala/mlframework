import glob
import numpy as np
import pandas as pd
from utils import write_df, read_df
from metrics import RegressionMetrics
from functools import partial

from scipy.optimize import fmin

PREDS_DIR = '../model_preds'
type='weighted'
metric_type='rmse'

class OptimizeBlending:
    def __init__(self, metric='rmse', problem_type='regression', disp=False):
        self.weight_arr = 0
        self.metric = metric
        self.problem_type = problem_type
        self.disp = disp

    def _metric(self, weight_arr, X, y):
        pred_arr = X @ weight_arr / sum(weight_arr)
        metric_dict = calculate_metrics(pred_arr, y, self.metric, self.problem_type)
        return metric_dict[self.metric]

    def fit(self, X, y):
        partial_loss = partial(self._metric, X=X, y=y)
        init_weight_arr = np.random.dirichlet(np.ones([X.shape[1]])).reshape(X.shape[1], -1)
        self.weight_arr = fmin(partial_loss, init_weight_arr, disp=self.disp)

    def predict(self, X):
        pred_arr = X @ self.weight_arr / sum(self.weight_arr)
        return pred_arr

def blending(df, type='weighted', weights=[1,1,1]):

    preds_cols = [col for col in df.columns if 'pred' in col]
    if type == 'weighted':
        weight_arr = np.array(weights).reshape(-1, 1)

        df_np = df[preds_cols].values
        pred_arr = df_np@weight_arr/sum(weight_arr)
        return pred_arr
    if type == 'rank_average':
        pass

    if type=='optimize':
        folds = df.kfold.unique()
        fold_preds_df = pd.DataFrame()
        optimal_preds_df = pd.DataFrame()
        weights = []
        for fold in folds:
            train_df = df.drop(columns=['kfold', 'Id'], axis=1)[df.kfold != fold]
            valid_df = df.drop(columns=['kfold', 'Id'], axis=1)[df.kfold == fold]

            optimal_enc = OptimizeBlending(metric='rmse')
            train_X, train_y = train_df.drop('target', axis=1).values, train_df.target.values
            # print(f'train_shape: {train_X.shape}\ntrain_y shape: {train_y.shape}')
            optimal_enc.fit(train_X, train_y)
            weights.append(optimal_enc.weight_arr)
        # print(f'weights: {np.array(weights)}')
        optimal_weights = np.mean(np.array(weights), axis=0)
        df_np = df[preds_cols].values
        pred_arr = df_np @ optimal_weights / sum(optimal_weights)

        return pred_arr, optimal_weights

def calculate_metrics(preds, target, metric='rmse', problem_type='regression'):
    # print(f'Calculating metric - {metric}')
    if problem_type=='regression':
        reg_metrics = RegressionMetrics()
        metric = reg_metrics(target, preds, metric)
        return metric

if __name__=='__main__':
    files = glob.glob('../model_preds/*.csv')
    preds_df=pd.DataFrame()

    FIRST = True
    for f in files:
        # print(f)
        if f.split('/')[2] in ['preds_lightgbm_27493.csv', 'preds_randomforest_27804.csv', 'preds_ridge_33672.csv']:
            df= pd.read_csv(f)
            if FIRST:
                preds_df = df.copy()
                FIRST = False
            else:
                preds_df = preds_df.merge(df, on=['Id'], how='left')

    true_label_df = read_df('train_cat_trial3_5fold.csv', '../input')
    true_label_df.rename(columns={"SalePrice":"target"}, inplace=True)

    preds_df = preds_df.merge(true_label_df[['Id','target']], on='Id', how='left')
    preds_cols = [col for col in preds_df.columns if 'preds' in col]

    sub_cols = preds_cols + ['Id', 'kfold', 'target']

    write_df(preds_df[sub_cols], 'preds_blending.csv', PREDS_DIR)
    # preds_df.to_csv('../model_preds/preds_blending.csv', index=False)
    target = preds_df.target.values
    for col in preds_cols:
        metric_value = calculate_metrics(preds_df[col].values, target, metric='rmse')
        print(f'Accuracy for {col}: {metric_value}')

    blended_preds = blending(preds_df[preds_cols], weights=[2, 4, 1], type=type)
    metric_value = calculate_metrics(blended_preds, target, metric=metric_type)
    print(f'Accuracy for blended type - {type}: - {metric_value}')

    sub_cols = preds_cols + ['Id', 'target', 'kfold']
    blended_preds, optimal_weights = blending(preds_df[sub_cols], type='optimize')
    print(blended_preds.shape)

    metric_value = calculate_metrics(blended_preds, target, metric=metric_type)
    print(f'Accuracy for blended type - {type} and optimal weights - {optimal_weights} is \n\t {metric_value}')

