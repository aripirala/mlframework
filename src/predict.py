import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

num_folds = os.environ.get(NUM_FOLDS)
MODEL = os.environ.get("MODEL")
SCALAR = os.environ.get("SCALAR")

def predict(test_data_path, model_type, model_path, scalar_type=None, num_folds=5):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(num_folds):
        df = pd.read_csv(test_data_path)
        # encoders = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        # for c in encoders:
        #     lbl = encoders[c]
        #     df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
        #     df.loc[:, c] = lbl.transform(df[c].values.tolist())

        df = df[cols]

        if MODEL in ['linear_regression', 'knn', 'logistic'] and scalar_type is not None:
            scalar = joblib.load(os.path.join(model_path, f"{scalar_type}_{FOLD}_normalizer.pkl"))
            df = scalar.transform(df)
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        preds = clf.predict(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= num_folds

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub

if __name__ == "__main__":
    submission = predict(test_data_path="input/test_mean_imputed.csv",
                         model_type=MODEL,
                         scalar_type=SCALAR,
                         model_path="models/")
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"models/rf_submission.csv", index=False)
