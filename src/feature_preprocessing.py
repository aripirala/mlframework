from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from impute import knn_impute

from configs import TRAIN_DATA, TEST_DATA, WORKING_DIR, INPUT_DIR
from utils import get_file_path, write_df, read_df, select_cols
import pandas as pd

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
    def __init__(self, df, imputation_type, strategy='mean', feature_type='NUMERIC'):

        self.df_imputed = None
        self.imputation_type=imputation_type
        self.strategy=strategy
        self.feature_type=feature_type
        self.df_original = df.copy()
        self.df = select_cols(df.copy(), self.feature_type)
        # print(len(self.df))
        print(self.df.columns)
        self.imputer = None


    def fit_transform(self, **kwargs):
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

        if self.feature_type=='NUMERIC':
            if self.strategy in ['mean', 'most_frequent', 'median', 'constant']:
                self.imputer = SimpleImputer(strategy=self.strategy, **kwargs)
                self.df_imputed = pd.DataFrame(self.imputer.fit_transform(self.df), columns=self.df.columns)
                return self.df_imputed
            elif self.strategy == 'knn_impute':
                missing_values_cols = self.missing_values_table(self.df).index.values
                for col in missing_values_cols:
                    print(f'Model based imputation for column - {col}')
                    target = self.df[col].values #should be a numpy array
                    attributes = self.df.drop(col, axis=1)
                    print(f'aggregation_method = {aggregation_method}')
                    print(f'Neighbhors = {n_neighbors}')
                    target_imputed = knn_impute(target, attributes, n_neighbors, aggregation_method=aggregation_method)
                    self.df[col+'_imputed'] = target_imputed.iloc[:,0].values
        elif self.feature_type=='CATEGORICAL':
            pass

    def transform(self, df):
        df_select = select_cols(df.copy(), self.feature_type)
        print(len(df_select.columns))
        return pd.DataFrame(self.imputer.transform(df_select), columns=df_select.columns)

    @staticmethod
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


