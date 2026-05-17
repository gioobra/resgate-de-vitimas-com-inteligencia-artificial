from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import numpy as np


class Testes:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])

    # Regressão
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)