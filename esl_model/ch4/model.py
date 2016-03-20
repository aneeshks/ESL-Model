from math import log

from ..base import BaseStatModel
from ..ch3.model import LeastSquareModel, LinearModel
import numpy as np
from ..utils import lazy_method
from functools import partial




class ErrorRateMixin:
    @property
    @lazy_method
    def error_rate(self):
        return (1 - np.sum((self._raw_train_y == self.y_hat))/ self.N)



class LinearRegressionIndicatorMatrix(ErrorRateMixin, LeastSquareModel):
    def __init__(self, *args, **kwargs):
        self.K = kwargs.pop('K', 1)
        super().__init__(*args, **kwargs)

    def _pre_processing_y(self, y):
        iy = y.flatten()
        N = y.shape[0]
        if self.K > 1:
            Y = np.zeros((N, self.K))
            for i in range(N):
                k = iy[i]
                # k starts from 1
                Y[i, k-1] = 1
        else:
            return super()._pre_processing_y(y)

        return Y

    def predict(self, x):
        Y_hat = super().predict(x)
        y = (Y_hat.argmax(axis=1)).reshape((-1,1)) + 1

        return y


class LDAModel(ErrorRateMixin, LinearModel):
    """
    Linear Discriminant Analysis
    from page 106
    """
    def __init__(self, *args, **kwargs):
        self.K = kwargs.pop('K', 1)
        super().__init__(*args, **kwargs)

    def _pre_processing_x(self, x):
        x = self.standardize(x)
        return x

    def train(self):
        X = self.train_x
        y = self.train_y
        K = self.K
        p = self.p

        self.Mu = np.zeros((K, p))
        self.Pi = np.zeros((K, 1))
        self.Sigma_hat = np.zeros((p, p))

        for k in range(K):
            mask = (y == k+1)
            N_k = sum(mask)

            X_k = X[mask.flatten(), :]

            self.Pi[k] = N_k / self.N
            self.Mu[k] = np.sum(X_k, axis=0).reshape((1, -1)) / N_k
            self.Sigma_hat = self.Sigma_hat + ((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / (self.N - K)




    def linear_discriminant_func(self, x, k):
        """
        linear discriminant function.
        Define by (4.10)
        :return: delta_k(x)
        """
        mu_k = self.Mu[k]
        pi_k = self.Pi[k]
        sigma_inv = self.math.pinv(self.Sigma_hat)
        result = mu_k @ sigma_inv @ x.T - (mu_k @ sigma_inv @ mu_k.T)/2 + log(pi_k)

        return result

    def predict(self, x):
        X = self._pre_processing_x(x)
        N = X.shape[0]
        Y = np.zeros((N, self.K))

        for k in range(self.K):
            # delta_k is (N x 1)
            delta_k = self.linear_discriminant_func(X, k)
            Y[:, k] = delta_k

        # make the k start from 1
        y_hat = Y.argmax(axis=1).reshape((-1, 1)) + 1

        return y_hat


class QDAModel(LDAModel):

    def _pre_processing_x(self, x):
        return self.standardize(x)

    def train(self):
        X = self.train_x
        y = self.train_y
        K = self.K
        p = self.p

        self.Mu = np.zeros((K, p))
        self.Pi = np.zeros((K, 1))
        self.Sigma_hat = []

        for k in range(K):
            mask = (y == k+1)
            N_k = sum(mask)

            X_k = X[mask.flatten(), :]

            self.Pi[k] = N_k / self.N
            self.Mu[k] = np.sum(X_k, axis=0).reshape((1, -1)) / N_k
            # We div by N_k instead of (N-K)
            self.Sigma_hat.append(((X_k - self.Mu[k]).T @ (X_k - self.Mu[k])) / (N_k))



    def quadratic_discriminant_func(self, x, k):
        mu_k = self.Mu[k]
        pi_k = self.Pi[k]
        sigma_k = self.Sigma_hat[k]
        pinv = self.math.pinv

        result = -(np.log(np.linalg.det(sigma_k)))/2 - \
                 ((x - mu_k) @ pinv(sigma_k, rcond=0) @ (x - mu_k).T)/2 + \
                 log(pi_k)
        return result



    def predict(self, x):
        X = self._pre_processing_x(x)
        N = X.shape[0]
        Y = np.zeros((N, self.K))

        rg = reversed(list(range(self.p + 1)))

        for k in rg:
            # delta_k is (N x 1)
            # delta_k_func = partial(self.linear_discriminant_func, k=k)
            # delta_k = np.apply_along_axis(delta_k_func, 1, X)

            # d is NxN
            d = self.quadratic_discriminant_func(X, k)
            Y[:, k] = d.diagonal()
        # make the k start from 1
        y_hat = Y.argmax(axis=1).reshape((-1, 1)) + 1

        return y_hat
