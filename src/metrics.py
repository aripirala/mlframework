from sklearn import metrics as skmetrics

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "auc": self._auc,
            "logloss": self._logloss
        }
    
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metric not implemented")
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for AUC")
        elif metric == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for logloss")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)


class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "explained_variance_score": self._explained_variance_score,
            "mean_absolute_error": self._mae,
            "mean_squared_error": self._mse,
            "median_absolute_error": self._median_absolute_error,
            'rmse': self._rmse,
            "r2_score": self._r2_score,
            "msle": self._msle,
        }
        self.y_true = None
        self.y_pred = None

    def __call__(self, y_true, y_pred, metric):
        self.y_true = y_true
        self.y_pred = y_pred

        if metric not in self.metrics and metric != 'all':
            raise Exception("Metric not implemented")
        if metric != 'all':
            # print(f'{metric} is being calculated')
            metric_dict={}
            metric_value =self.metrics[metric]()
            metric_dict[metric] = metric_value
            # print(f'Metric {metric}: {metric_value}')
            return metric_dict
        if metric == 'all':
            metric_dict = {}
            for metric, func in self.metrics.items():
                metric_value = func()
                metric_dict[metric] = metric_value
                # print(f'Metric {metric}: {metric_value}')
                return metric_dict

    def _explained_variance_score(self):
        return skmetrics.explained_variance_score(y_true=self.y_true, y_pred=self.y_pred)

    def _rmse(self):
        return skmetrics.mean_squared_error(y_true=self.y_true, y_pred=self.y_pred, squared=False)

    def _mae(self):
        return skmetrics.mean_absolute_error(y_true=self.y_true, y_pred=self.y_pred)

    def _median_absolute_error(self):
        return skmetrics.median_absolute_error(y_true=self.y_true, y_pred=self.y_pred)

    def _mse(self):
        return skmetrics.mean_squared_error(y_true=self.y_true, y_pred=self.y_pred)

    def _msle(self):
        return skmetrics.mean_squared_log_error(y_true=self.y_true, y_pred=self.y_pred)

    def _r2_score(self):
        return skmetrics.r2_score(y_true=self.y_true, y_pred=self.y_pred)
