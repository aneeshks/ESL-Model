import  numpy as np
from .utils import lazy_method
from numpy import linalg
from scipy.linalg import svd


class MathCollection:
    def __init__(self):
        self.inv = linalg.inv
        self.sum = np.sum
        self.svd = svd

    def __repr__(self):
        return 'Math Collection'

mathcollection = MathCollection()




class Result:
    def __init__(self, y_hat:np.ndarray, y:np.ndarray):
        self.y_hat = y_hat
        self.y = y

    @property
    @lazy_method
    def mse(self):
        return mathcollection.sum((self.y - self.y_hat)**2)/ self.y.shape[0]

    @lazy_method
    def z_score(self):
        pass







class BaseStatModel:

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray, features_name=None):
        self.train_x =  self._raw_train_x = train_x
        self.train_y = train_y
        self.features_name = features_name



    def standardize(self, x, axis=0, with_mean=True, with_std=True):
        if getattr(self, '_x_std_', None) is None or getattr(self, '_x_mean_', None) is None:
            self._x_mean_ = x.mean(axis=0)
            self._x_std_ = x.std(axis=0, ddof=1)

        return (x-self._x_mean_) / self._x_std_


    @property
    def N(self):
        return self._raw_train_x.shape[0]

    @property
    def p(self):
        """
        number of features exclude intercept one
        :return:
        """
        return self._raw_train_x.shape[1]


    def pre_processing_x(self, x):
        return x

    def pre_processing_y(self, y):
        return y

    def pre_processing(self):
        self.train_x = self.pre_processing_x(self.train_x)
        self.train_y = self.pre_processing_y(self.train_y)

    def train(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def test(self, x, y):
        y_hat = self.predict(x)
        return Result(y_hat, y)

    @property
    @lazy_method
    def y_hat(self):
        return self.predict(self._raw_train_x)


    @property
    def math(self):
        return mathcollection





