import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers
from sklearn.impute import SimpleImputer
from utils import get_columns_with_missing_values
from sklearn.preprocessing import MinMaxScaler

class KNNImputation:
    def __init__(self, strategy='mean', aggregation_method='mean', numerical_distance='euclidean', categorical_distance='jaccard'):
        self.numeric_columns = None
        self.categorical_columns = None
        self.strategy=strategy
        self.aggregation_method = aggregation_method
        self.numerical_distance=numerical_distance
        self.categorical_distance=categorical_distance
        self.transform_metrics = None
        self.standardizer = None
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.scalar = MinMaxScaler()
        self.X_numeric_imputed = None
        self.categorical_feature_modes = []

    @staticmethod
    def weighted_hamming(data):
        """ Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
            the values between point A and point B are different, else it is equal the relative frequency of the
            distribution of the value across the variable. For multiple variables, the harmonic mean is computed
            up to a constant factor.
            @params:
                - data = a pandas data frame of categorical variables
            @returns:
                - distance_matrix = a distance matrix with pairwise distance for all attributes
        """
        categories_dist = []

        for category in data:
            X = pd.get_dummies(data[category])
            X_mean = X * X.mean()
            X_dot = X_mean.dot(X.transpose())
            X_np = np.asarray(X_dot.replace(0, 1, inplace=False))
            categories_dist.append(X_np)
        categories_dist = np.array(categories_dist)
        distances = hmean(categories_dist, axis=0)
        return distances

    def fit(self, X):
        X = X.copy()
        X_knn =X.copy()
        is_numeric = [all(isinstance(n, numbers.Number) for n in X.iloc[:, i]) for i, x in enumerate(X)]
        is_all_numeric = sum(is_numeric) == len(is_numeric)
        is_all_categorical = sum(is_numeric) == 0
        is_mixed_type = not is_all_categorical and not is_all_numeric
        self.numeric_columns = X.iloc[:, is_numeric].columns
        self.categorical_columns = X.iloc[:, ~np.array(is_numeric)].columns
        X_numeric =  X[[self.numeric_columns]]
        X_categorical = X[[self.categorical_columns]]

        #impute the numerical columns with simple imputation
        print(f'imputing numerical columns with the strategy -{self.strategy}')
        X_numeric_imputed = pd.DataFrame(self.imputer.fit_transform(X_numeric), columns=X_numeric.columns)
        self.X_numeric_imputed = pd.DataFrame(self.scalar.fit_transform(X_numeric_imputed), columns=X_numeric.columns)

        #impute categorical data with mode or most frequent
        self.categorical_feature_modes = [X_categorical[col].mode()[0] for col in self.categorical_columns]

        for col in self.categorical_columns:
            self.X_categorical_imputed = X_categorical[col].fillna(X_categorical[col].mode()[0])

        self.X_imputed = pd.concat([self.X_numeric_imputed, self.X_categorical_imputed], axis=1)


        missing_value_columns = get_columns_with_missing_values(X)
        remaining_cols = set(np.array(X.columns)) - set(missing_value_columns)
        imputed_df = pd.DataFrame()
        for col in missing_value_columns:
            target = X_knn[col].values
            attributes = self.X_imputed.drop(col)
            target_imputed = self.knn_impute(target, attributes, k_neighbors=5, aggregation_method='mean')
            target_imputed.columns = [col]
            imputed_df = pd.concat([imputed_df, target_imputed], axis=1)

        final_df = pd.concat([X[[remaining_cols]], imputed_df], axis=1)[[X.columns]]

        return final_df

    def distance_matrix(self, X1, X2=None, is_train=True):
        """ Compute the pairwise distance attribute by attribute in order to account for different variables type:
            - Continuous
            - Categorical
            For ordinal values, provide a numerical representation taking the order into account.
            Categorical variables are transformed into a set of binary ones.
            If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
            variables are all normalized in the process.
            If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.

            Note: If weighted-hamming distance is chosen, the computation time increases a lot since it is not coded in C
            like other distance metrics provided by scipy.
            @params:
                - X1, X2                  = pandas dataframe to compute distances on.
                - numeric_distances     = the metric to apply to continuous attributes.
                                          "euclidean" and "cityblock" available.
                                          Default = "euclidean"
                - categorical_distances = the metric to apply to binary attributes.
                                          "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                          available. Default = "jaccard"
            @returns:
                - the distance matrix
        """
        possible_continuous_distances = ["euclidean", "cityblock"]
        possible_binary_distances = ["euclidean", "jaccard", "hamming", "weighted-hamming"]
        number_of_variables = X1.shape[1]
        number_of_observations = X1.shape[0]
        X1_len = len(X1)

        if not is_train:
            X=pd.concat([X1, X2], axis=0)
        else:
            X = X1

        # Get the type of each attribute (Numeric or categorical)
        is_numeric = len(self.numeric_columns) > 0
        is_all_numeric = len(self.numeric_columns) == X1.shape[1]
        is_all_categorical = is_numeric is False
        is_mixed_type = not is_all_categorical and not is_all_numeric
        number_of_numeric_var = len(self.numeric_columns)
        number_of_categorical_var = len(X.columns) - number_of_numeric_var

        # Check the content of the distances parameter
        if self.numeric_distance not in possible_continuous_distances:
            print("The continuous distance " + self.numeric_distance + " is not supported.")
            return None
        elif self.categorical_distance not in possible_binary_distances:
            print("The binary distance " + self.categorical_distance + " is not supported.")
            return None

        # # Separate the data frame into categorical and numeric attributes and normalize numeric data
        # if is_mixed_type:
        #     number_of_numeric_var = sum(is_numeric)
        #     number_of_categorical_var = number_of_variables - number_of_numeric_var
        #     X_numeric = X.iloc[:, is_numeric]
        #     # X_numeric = (X_numeric - X_numeric.mean()) / (X_numeric.max() - X_numeric.min())
        #     X_categorical = X.iloc[:, [not x for x in is_numeric]]
        #
        # # Replace missing values with column mean for numeric values and mode for categorical ones. With the mode, it
        # # triggers a warning: "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
        # # but the value are properly replaced
        #
        # # X1 and X2 should be mean/median or mode imputed (mode for categorical variables)
        # # it should be normalized using min-max scalar
        #
        # if is_mixed_type:
        #     X_numeric.fillna(data_numeric.mean(), inplace=True)
        #     for x in data_categorical:
        #         data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)
        # elif is_all_numeric:
        #     data.fillna(data.mean(), inplace=True)
        # else:
        #     for x in data:
        #         data[x].fillna(data[x].mode()[0], inplace=True)

        # "Dummifies" categorical variables in place
        if not is_all_numeric and not (self.categorical_distance in ['hamming', 'weighted-hamming']):
            if is_mixed_type:
                X_categorical = pd.get_dummies(X[[self.categorical_columns]])
            else:
                X = pd.get_dummies(X)
        elif not is_all_numeric and self.categorical_distance == 'hamming':
            if is_mixed_type:
                X_categorical = pd.DataFrame(
                    [pd.factorize(X[[self.categorical_columns]][col])[0] for col in X[[self.categorical_columns]]]).transpose()
            else:
                X = pd.DataFrame([pd.factorize(X[col])[0] for col in X]).transpose()
        if is_all_numeric:
            if is_train:
                result_matrix = cdist(X, X, metric=self.numeric_distance)
            else:
                result_matrix = cdist(X.iloc[:X1_len, :], X.iloc[X1_len:, :], metric=self.numeric_distance)
        elif is_all_categorical:
            if self.categorical_distance == "weighted-hamming":
                result_matrix = self.weighted_hamming(X)
            else:
                if is_train:
                    result_matrix = cdist(X, X, metric=self.categorical_distance)
                else:
                    result_matrix = cdist(X.iloc[:X1_len, :], X.iloc[X1_len:, :], metric=self.categorical_distance)
        else:
            X_numeric = X[[self.numeric_columns]]
            if is_train:
                result_numeric = cdist(X_numeric, X_numeric, metric=self.numerical_distance)
            else:
                result_numeric = cdist(X_numeric.iloc[:X1_len, :], X_numeric.iloc[X1_len:, :],
                                       metric=self.numerical_distance)

            if self.categorical_distance == "weighted-hamming":
                result_categorical = self.weighted_hamming(X_categorical)
            else:
                if is_train:
                    result_categorical = cdist(X_categorical, X_categorical, metric=self.categorical_distance)
                else:
                    result_categorical = cdist(X_categorical.iloc[:X1_len, :], X_categorical.iloc[X1_len:, :],
                                           metric=self.categorical_distance)

            result_matrix = np.array([[1.0 * (result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] *
                                              number_of_categorical_var) / number_of_variables for j in
                                       range(number_of_observations)] for i in range(number_of_observations)])

        # Fill the diagonal with NaN values
        np.fill_diagonal(result_matrix, np.nan)

        return pd.DataFrame(result_matrix)


    def knn_impute(self, target, attributes, k_neighbors=5, aggregation_method="mean",
                   missing_neighbors_threshold=0.5, is_train=True):
        """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
            attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
            remains missing. If there is a problem in the parameters provided, returns None.
            If to many neighbors also have missing values, leave the missing value of interest unchanged.
            @params:
                - target                        = a vector of n values with missing values that you want to impute. The length has
                                                  to be at least n = 3.
                - attributes                    = a data frame of attributes with n rows to match the target variable
                - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                                  value between 1 and n.
                - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                                  Default = "mean"
                - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                                  the correct value. Default = 0.5
            @returns:
                target_completed        = the vector of target values with missing value replaced. If there is a problem
                                          in the parameters, return None
        """

        # Get useful variables
        possible_aggregation_method = ["mean", "median", "mode"]
        number_observations = len(target)
        is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

        # Check for possible errors
        if number_observations < 3:
            print("Not enough observations.")
            return None
        if attributes.shape[0] != number_observations:
            print("The number of observations in the attributes variable is not matching the target variable length.")
            return None
        if k_neighbors > number_observations or k_neighbors < 1:
            print("The range of the number of neighbors is incorrect.")
            return None
        if aggregation_method not in possible_aggregation_method:
            print("The aggregation method is incorrect.")
            return None
        if not is_target_numeric and aggregation_method != "mode":
            print("The only method allowed for categorical target variable is the mode.")
            return None

        # Make sure the data are in the right format
        target = pd.DataFrame(target).copy()
        attributes = pd.DataFrame(attributes).copy()
        target = target.fillna(np.nan)

        # Get the distance matrix and check whether no error was triggered when computing it
        if is_train:
            distances = self.distance_matrix(X1=attributes, is_train=True)
        else:
            distances = self.distance_matrix(X1=self.X_train, X2=attributes, is_train=False)

        print(f"Computed Distances -  distances shape is {distances.shape}")
        if distances is None:
            return None

        # Get the closest points and compute the correct aggregation method
        for i, value in enumerate(target.iloc[:, 0]):
            if pd.isnull(value):
                order = distances.iloc[i, :].values.argsort()[:k_neighbors]
                closest_to_target = target.iloc[order, :]
                missing_neighbors = [x for x in closest_to_target.isnull().iloc[:, 0]]

                # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
                if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
                    continue
                elif aggregation_method == "mean":
                    target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target, np.isnan(closest_to_target)))
                elif aggregation_method == "median":
                    target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target, np.isnan(closest_to_target)))
                else:

                    # print(f'Hi i: {i} -  {closest_to_target[~np.array(missing_neighbors)]}')
                    # print(stats.mode(closest_to_target[~np.array(missing_neighbors)], nan_policy='omit'))
                    target.iloc[i] = stats.mode(closest_to_target[~np.array(missing_neighbors)], nan_policy='omit')[0][0]

        return target

    def transform(df, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
                   categorical_distance="jaccard", missing_neighbors_threshold=0.5):
        """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
            attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
            remains missing. If there is a problem in the parameters provided, returns None.
            If to many neighbors also have missing values, leave the missing value of interest unchanged.
            @params:
                - target                        = a vector of n values with missing values that you want to impute. The length has
                                                  to be at least n = 3.
                - attributes                    = a data frame of attributes with n rows to match the target variable
                - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                                  value between 1 and n.
                - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                                  Default = "mean"
                - numeric_distances             = the metric to apply to continuous attributes.
                                                  "euclidean" and "cityblock" available.
                                                  Default = "euclidean"
                - categorical_distances         = the metric to apply to binary attributes.
                                                  "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                                  available. Default = "jaccard"
                - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                                  the correct value. Default = 0.5
            @returns:
                target_completed        = the vector of target values with missing value replaced. If there is a problem
                                          in the parameters, return None
        """

        # Get useful variables
        possible_aggregation_method = ["mean", "median", "mode"]
        number_observations = len(target)
        is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

        # Check for possible errors
        if number_observations < 3:
            print("Not enough observations.")
            return None
        if attributes.shape[0] != number_observations:
            print("The number of observations in the attributes variable is not matching the target variable length.")
            return None
        if k_neighbors > number_observations or k_neighbors < 1:
            print("The range of the number of neighbors is incorrect.")
            return None
        if aggregation_method not in possible_aggregation_method:
            print("The aggregation method is incorrect.")
            return None
        if not is_target_numeric and aggregation_method != "mode":
            print("The only method allowed for categorical target variable is the mode.")
            return None

        # Make sure the data are in the right format
        target = pd.DataFrame(target).copy()
        attributes = pd.DataFrame(attributes).copy()
        target = target.fillna(np.nan)

        # Get the distance matrix and check whether no error was triggered when computing it
        distances = distance_matrix(attributes, numeric_distance, categorical_distance)
        print(f"Computed Distances -  distances shape is {distances.shape}")
        if distances is None:
            return None

        # Get the closest points and compute the correct aggregation method
        for i, value in enumerate(target.iloc[:, 0]):
            if pd.isnull(value):
                order = distances.iloc[i, :].values.argsort()[:k_neighbors]
                closest_to_target = target.iloc[order, :]
                missing_neighbors = [x for x in closest_to_target.isnull().iloc[:, 0]]

                # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
                if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
                    continue
                elif aggregation_method == "mean":
                    target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target, np.isnan(closest_to_target)))
                elif aggregation_method == "median":
                    target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target, np.isnan(closest_to_target)))
                else:

                    # print(f'Hi i: {i} -  {closest_to_target[~np.array(missing_neighbors)]}')
                    # print(stats.mode(closest_to_target[~np.array(missing_neighbors)], nan_policy='omit'))
                    target.iloc[i] = stats.mode(closest_to_target[~np.array(missing_neighbors)], nan_policy='omit')[0][0]

        return target #, attributes, distances
