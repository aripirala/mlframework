import pandas as pd
from sklearn import model_selection
from configs import TRAIN_DATA, TEST_DATA, WORKING_DIR, INPUT_DIR
from utils import get_file_path, write_df, read_df

class CreateCVFolds:
    def __init__(self, df, fold_type='kfold', n_splits=5, target='target', shuffle=False, output_fname='train_stratified_folds.csv', random_state=42):
        self.fold_type = fold_type
        self.n_splits = n_splits
        self.target=target
        self.shuffle= shuffle
        self.output_fname = output_fname
        self.random_state = random_state
        self.df = df.copy()

        #create a kfold column and save a dummy value in it
        self.df["kfold"] = -1
        self.df = df.sample(frac=1).reset_index(drop=True)  # shuffle the sample just to create randomness
        # self.create_folds()

    def create_folds(self):
        if self.fold_type=='kfold':
            self.create_kfold(self.df, n_splits=self.n_splits,
                                         shuffle=self.shuffle,
                                         output_fname=self.output_fname,
                                         random_state=self.random_state
                                         )
        elif self.fold_type=='stratified_kfold':
            self.create_stratified_kfold(self.df, n_splits=self.n_splits,
                                         target=self.target, shuffle=self.shuffle,
                                         output_fname=self.output_fname,
                                         random_state=self.random_state
                                         )

    @staticmethod
    def create_stratified_kfold(df, n_splits=5, target='target', shuffle=False, output_fname='train_stratified_folds.csv', random_state=42):

        kf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target].values)):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, 'kfold'] = fold

        write_df(df, output_fname, INPUT_DIR)
        # self.df.to_csv("../input/train_folds.csv", index=False)

    @staticmethod
    def create_kfold(df, n_splits=5, output_fname='train_folds.csv', shuffle=False, random_state=42):
        kf = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, 'kfold'] = fold

        write_df(df, output_fname, INPUT_DIR)


if __name__ == "__main__":
    train_path = get_file_path('train.csv')
    test_path = get_file_path('test.csv')

    train_df = read_df(train_path)
    test_df = read_df(test_path)

    cv = CreateCVFolds(train_df,
                       fold_type='kfold',
                       n_splits=5,
                       shuffle=True,
                       output_fname='train_folds.csv',
                       random_state=42)
    cv.create_folds()
