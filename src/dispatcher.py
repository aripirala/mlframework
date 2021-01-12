from sklearn import ensemble, linear_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

MODELS = {
    "randomforest": ensemble.RandomForestRegressor(n_estimators=50, n_jobs=-1, verbose=0),
    "extratrees": ensemble.ExtraTreesRegressor(n_estimators=50, n_jobs=-1, verbose=0),
    "linear_regression": linear_model.LinearRegression()
}

NORMALIZER = {
    'min_max_scalar': MinMaxScaler(),
    'standard_scalar':StandardScaler(),
}