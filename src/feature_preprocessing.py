from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from knn_impute import KNNImputation


from configs import TRAIN_DATA, TEST_DATA, WORKING_DIR, INPUT_DIR
from utils import get_file_path, write_df, read_df, select_dtype_df
import pandas as pd
import numpy as np

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)
    
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
        
        else:
            raise Exception("Encoding type not understood")


class NumericalFeatures:
    def __init__(self, df, numerical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of column names, e.g. ["ord_1", "nom_0"......]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.num_feats = numerical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.num_feats:
                self.df.loc[:, c] = self.df.loc[:, c].fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.num_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe
        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)
        else:
            raise Exception("Encoding type not understood")

class Transformation:
    def __init__(self, df, ft_transform=None):
        # self.pipeline = pipeline
        self.ft_transforms=ft_transform
        if df is not None:
            self.df= df.copy()
        else:
            raise Exception ("DataFrame cannot be empty")
        self.transformed_cols = []

    def get_df(self):
        return self.df

    def get_transformed_cols(self):
        return self.transformed_cols

    def compose(self, ft_transforms=None):
        if ft_transforms:
            self.ft_transforms = ft_transforms

    def apply_transforms(self, df=None, ft_transforms=None):
        if df is not None:
            self.df=df.copy()

        if self.df is None:
            raise Exception ("df cannot be empty")

        if ft_transforms is not None:
            self.ft_transforms = ft_transforms

        if self.ft_transforms is None:
            raise Exception("ft_transforms cannot be empty")

        for col in self.ft_transforms.keys():
            (fn, kwargs) = self.ft_transforms[col]

            print(f'col = {col}, function:{fn}, kwargs:{kwargs}')
            transform_type = str(fn).split()[1]
            output_col = f'{col}_{transform_type}'

            if col not in self.df.columns:
                raise Exception (f"{col} not in the dataframe")
            self.df[output_col] = self.apply_transformation(self.df[col].values, fn, **kwargs)
            self.transformed_cols.append(output_col)

    @staticmethod
    def apply_transformation(data_arr, fn, *args, **kwargs):
        print(f'kwargs:{kwargs}')
        print(f"args:{args}")
        # print(type(data_arr))
        # print(data_arr)
        return fn(data_arr, *args, **kwargs)

class MissingValueImputer:
    def __init__(self, imputation_type='non_time_series', strategy='IMPUTE',
                 strategy_numeric='mean', strategy_categorical='mode',
                 feature_type='ALL',
                 drop_feature_threshold=0.4, fill_value=-9999):

        self.df_imputed = None
        self.imputation_type=imputation_type
        self.strategy = strategy # can be IMPUTE/DROP_FEATURES
        self.strategy_numeric = strategy_numeric
        self.strategy_categorical = strategy_categorical
        self.feature_type = feature_type
        # self.df_original = df.copy()
        self.df = None
        self.imputer_numeric = None
        self.imputer_categorical = None
        self.feature_missing_table = None
        self.drop_feature_threshold = drop_feature_threshold

        self.df_categorical_imputed = pd.DataFrame()
        self.df_numeric_imputed = pd.DataFrame()
        self.remaining_non_cat_cols = None
        self.remaining_non_numeric_cols = None
        self.fill_value = fill_value

    def _numeric_fit_transform(self, df):
        self.df_numeric = select_dtype_df(df.copy(), 'NUMERIC')
        print(f'Started Numerical {self.strategy_numeric} imputation')

        self.numeric_columns = self.df_numeric.columns
        self.remaining_non_numeric_cols = set(np.array(df.columns)) - set(np.array(self.numeric_columns))

        if self.strategy_numeric in ['mean', 'most_frequent', 'median', 'constant']:
            self.imputer_numeric = SimpleImputer(strategy=self.strategy_numeric, fill_value=self.fill_value)
            self.df_numeric_imputed = pd.DataFrame(self.imputer_numeric.fit_transform(self.df_numeric),
                                           columns=self.numeric_columns)
            final_df = pd.concat([df[self.remaining_non_numeric_cols], self.df_numeric_imputed], axis=1)
            print(f'Numerical Imputer is fit and completed the transformation')
            return final_df

    def _categorical_fit_transform(self, df):
        self.df_categorical = select_dtype_df(df.copy(), 'CATEGORICAL')
        self.categorical_columns = self.df_categorical.columns

        self.remaining_non_cat_cols = set(np.array(df.columns)) - set(np.array(self.categorical_columns))

        if self.strategy_categorical in ['mode']:
            self.categorical_feature_modes = dict(zip(self.categorical_columns,
                                                      [self.df_categorical[col].mode()[0] for col in
                                                       self.categorical_columns])
                                                  )
            # print(f'Categorical columns : {self.categorical_columns}')

            for col in self.categorical_columns:
                df_categorical_feature_imputed = self.df_categorical[col].fillna(self.categorical_feature_modes[col])
                self.df_categorical_imputed = pd.concat([self.df_categorical_imputed, df_categorical_feature_imputed],
                                                        axis=1)
            final_df = pd.concat([self.df_categorical_imputed, df[self.remaining_non_cat_cols]], axis=1)[df.columns]
            return final_df

        if self.strategy_categorical == 'constant':
            self.imputer_categorical = SimpleImputer(strategy=self.strategy_categorical, fill_value=self.fill_value)

            df_categorical_imputed = pd.DataFrame(self.imputer_categorical.fit_transform(self.df_categorical),
                                                   columns=self.categorical_columns)
            # final_df = pd.concat([df[self.remaining_non_numeric_cols], self.df_numeric_imputed], axis=1)
            final_df = pd.concat([df_categorical_imputed, df[self.remaining_non_cat_cols]], axis=1)[df.columns]
            return final_df

    def _numeric_transform(self, df):
        df_numeric = df.copy()[self.numeric_columns]

        if self.strategy_numeric in ['mean', 'most_frequent', 'median', 'constant']:
            df_numeric_imputed = pd.DataFrame(self.imputer_numeric.transform(df_numeric),
                                           columns=self.numeric_columns)
            final_df = pd.concat([df[self.remaining_non_numeric_cols], df_numeric_imputed], axis=1)
            return final_df

    def _categorical_transform(self, df):
        df_categorical = df[self.categorical_columns]
        if self.strategy_categorical in ['mode']:
            # print(f'Categorical columns : {self.categorical_columns}')
            df_categorical_imputed = pd.DataFrame()
            for col in self.categorical_columns:
                df_categorical_feature_imputed = df_categorical[col].fillna(self.categorical_feature_modes[col])
                df_categorical_imputed = pd.concat([df_categorical_imputed, df_categorical_feature_imputed],
                                                        axis=1)

        if self.strategy_categorical == 'constant':
            df_categorical_imputed = pd.DataFrame(self.imputer_categorical.transform(df_categorical),
                                           columns=self.categorical_columns)

        final_df = pd.concat([df_categorical_imputed, df[self.remaining_non_cat_cols]], axis=1)[df.columns]
        return final_df

    def fit_transform(self, df):
        # TODO handle numerical imputation
        # - Mean, Median, Mode - Imputer - Done
        # - Model based (KNN Regressor)
        # - Expectation Maximization
        # TODO handle categorical Imputation
        # - Most frequent, least frequent
        # - Create a new category called UNK
        # - Model based (KNN)
        # TODO handle time series Imputation
        # - Intepolate methods
        # TODO date time Imputation
        # - Interpolate methods
        # TODO General Methods
        # - create a column identifying null value
        if self.imputation_type == 'time_series':
            pass
        else:
            if self.feature_type=='NUMERIC':
                if self.strategy_numeric in ['mean', 'most_frequent', 'median', 'constant']:
                    final_df = self._numeric_fit_transform(df)
                    return final_df
                elif self.strategy_numeric == 'knn_impute':
                    #TODO: integrate the KNNImputer class and this class
                    pass
            elif self.feature_type=='CATEGORICAL':
                if self.strategy_categorical=='mode':
                    final_df = self._categorical_fit_transform(df)
                    return final_df
                elif self.strategy_categorical=='knn_impute':
                    pass #TODO need to code
            elif self.feature_type == 'BOTH':
                if self.strategy_numeric in ['mean', 'most_frequent', 'median', 'constant']:
                    final_df = self._numeric_fit_transform(df)
                    print(f'df shape is {final_df.shape}')
                if self.strategy_categorical in ['mode', 'constant']:
                    final_df = self._categorical_fit_transform(final_df)
                    print(f'df shape is {final_df.shape}')
                    return final_df
                elif self.strategy_numeric == 'knn_impute':
                    #TODO: integrate the KNNImputer class and this class
                    pass

                elif self.strategy_categorical == 'knn_impute':
                    pass  # TODO need to code

            elif self.feature_type == 'ALL':
                if self.strategy == 'DROP_FEATURES':
                    final_df = self._drop_fit_transform(df)
                    return final_df
                elif self.strategy == 'IMPUTE':
                    pass

    def transform(self, df):
        # print(len(df_select.columns))
        if self.strategy == 'DROP_FEATURES':
            return self._drop_transform(df)

        elif self.strategy == 'IMPUTE':
            if self.feature_type == 'NUMERIC':
                if self.strategy_numeric in ['mean', 'most_frequent', 'median', 'constant']:
                    final_df = self._numeric_transform(df)
                    return final_df
                elif self.strategy_numeric == 'knn_impute':
                    # TODO: integrate the KNNImputer class and this class
                    pass
            elif self.feature_type == 'CATEGORICAL':
                if self.strategy_categorical == 'mode':
                    final_df = self._categorical_transform(df)
                    return final_df
                elif self.strategy_categorical == 'knn_impute':
                    pass  # TODO need to code
            elif self.feature_type == 'BOTH':
                if self.strategy_numeric in ['mean', 'most_frequent', 'median', 'constant']:
                    final_df = self._numeric_transform(df)
                    print(f'df shape is {final_df.shape}')
                if self.strategy_categorical in ['mode', 'constant']:
                    final_df = self._categorical_transform(final_df)
                    return final_df

    def _drop_fit_transform(self, df):
        self.feature_missing_table = self.missing_values_table(df)

        features_greater_than_threshold_percent_missing = \
            self.feature_missing_table[(self.feature_missing_table['% of Total Values'] >
                                        self.drop_feature_threshold * 100.0)].index.values

        print(f'Features not meeting the threshold {self.drop_feature_threshold*100} - '
              f'\n {features_greater_than_threshold_percent_missing}')

        self.percent_missing_columns = [col for col in df.columns
                                        if col not in features_greater_than_threshold_percent_missing]
        return df[self.percent_missing_columns]

    def _drop_transform(self, df):
        return df[self.percent_missing_columns]

    @staticmethod
    def missing_values_table(df):
        # Total missing values

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


def add(np_arr, n=5,*args, **kwargs):
    print(f'n is {n}')
    # print(type(np_arr))
    # print(np_arr[:5])
    return np_arr+n

def square(arr):
    return arr**2

if __name__ == "__main__":
    # import pandas as pd
    # from sklearn import linear_model
    train_path = get_file_path(TRAIN_DATA)
    test_path = get_file_path(TEST_DATA)

    train_df = read_df(train_path)
    test_df = read_df(test_path)

    # sample = pd.read_csv("../input/sample_submission.csv")

    train_len = len(train_df)

    print(train_len)


    # ft_transforms = {
    #     'LotArea':(add,{'n':100}),
    #     'MoSold': (add, {'n': 7}),
    #     'LotArea_add':(square,{})
    #                  }
    # train_transformer = Transformation(train_df)
    # train_transformer.compose(ft_transforms=ft_transforms)
    # train_transformer.apply_transforms()
    # train_transformed_df = train_transformer.get_df()
    # write_df(train_transformed_df, "train_transformed.csv", INPUT_DIR)
    # print(train_transformer.get_transformed_cols())

    # df_test["target"] = -1
    # full_data = pd.concat([df, df_test])
    #
    # cols = [c for c in df.columns if c not in ["id", "target"]]
    # cat_feats = CategoricalFeatures(full_data,
    #                                 categorical_features=cols,
    #                                 encoding_type="ohe",
    #                                 handle_na=True)
    # full_data_transformed = cat_feats.fit_transform()
    #
    # X = full_data_transformed[:train_len, :]
    # X_test = full_data_transformed[train_len:, :]
    #
    # clf = linear_model.LogisticRegression()
    # clf.fit(X, df.target.values)
    # preds = clf.predict_proba(X_test)[:, 1]
    #
    # sample.loc[:, "target"] = preds
    # sample.to_csv("submission.csv", index=False)


